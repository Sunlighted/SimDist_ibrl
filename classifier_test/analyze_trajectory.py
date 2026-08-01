"""Roll out ONE trajectory (source policy in the source env) and visualize,
step by step, where the discriminator calls it source vs target.

A source-policy / source-env trajectory is "all source" by construction, so an
ideal dynamics discriminator should output P(target) < 0.5 everywhere. Plotting
P(target) over time shows exactly which states the discriminator mislabels and
how that lines up with the contact phase.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/Chenyang/ibrl"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import train_rl  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.domain_shift import DomainShiftConfig  # noqa: E402
from common_utils import ibrl_utils as utils  # noqa: E402
from discriminator import DynamicsDiscriminator  # noqa: E402
from data_utils import needle_ring_dist  # noqa: E402


# Dynamics-only target shift (matches CLAUDE.md §12 / the target dataset).
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
    mask_hole_offset_in_obs=False, ring_size_scale=0.85, ring_friction_scale=2.0,
)


def _rollout(env, agent, seed, state_key="state"):
    s_list, a_list, ns_list, r_list = [], [], [], []
    with torch.no_grad(), utils.eval_mode(agent):
        np.random.seed(seed)
        obs, _ = env.reset()
        prev = obs[state_key].clone()
        terminal = False
        while not terminal:
            action = agent.act(obs, eval_mode=True)
            obs, reward, terminal, _, _ = env.step(action)
            s_list.append(prev.cpu().numpy())
            a_list.append(action.cpu().numpy())
            ns_list.append(obs[state_key].cpu().numpy())
            r_list.append(float(reward))
            prev = obs[state_key].clone()
    return (np.stack(s_list), np.stack(a_list), np.stack(ns_list), np.array(r_list))


def collect_one(weight, seed, shift="none", require="any", max_tries=40, state_key="state"):
    """Roll out trajectories until one matches `require` (any/success/fail)."""
    agent, _, env_params = train_rl.load_model(weight, "cuda")
    env_params = dict(env_params)
    env_params["domain_shift"] = (
        DomainShiftConfig(mode="none") if shift == "none"
        else DomainShiftConfig(**TARGET_SHIFT))
    env = PixelRobosuite(**env_params)
    for t in range(max_tries):
        sd = seed + t
        traj = _rollout(env, agent, sd, state_key)
        succeeded = bool((traj[3] > 0).any())
        if require == "any" or (require == "success" and succeeded) or (require == "fail" and not succeeded):
            print(f"[traj] using seed={sd} (success={succeeded})")
            return traj, sd
    print(f"[traj] WARNING: no '{require}' trajectory in {max_tries} tries; using last (seed={sd})")
    return traj, sd


def load_disc(path, device):
    ck = torch.load(path, map_location=device)
    d = DynamicsDiscriminator(ck["state_dim"], ck["action_dim"], ck["hidden"], ck["dropout"]).to(device)
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def score(d, s, a, ns, device):
    S, A, NS = (torch.as_tensor(x, dtype=torch.float32, device=device) for x in (s, a, ns))
    with torch.no_grad():
        sa_prob, sas_prob = d.prob_sas(S, A, NS)
        log_r = d.log_dynamics_ratio(S, A, NS)
    return (sa_prob[:, 0].cpu().numpy(),   # P(target) from d_sa
            sas_prob[:, 0].cpu().numpy(),  # P(target) from d_sas
            log_r.cpu().numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--disc", default=f"{os.path.dirname(__file__)}/discriminator.pt")
    p.add_argument("--disc_contact", default=f"{os.path.dirname(__file__)}/discriminator_contact.pt")
    p.add_argument("--contact_thresh", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shift", choices=["none", "target"], default="none",
                   help="env to roll out in: none=source, target=dynamics-only shift")
    p.add_argument("--require", choices=["any", "success", "fail"], default="any",
                   help="keep searching seeds until a success/fail trajectory is found")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/trajectory_analysis.png")
    args = p.parse_args()
    device = "cuda"

    # truth label of this trajectory's env: 1 = source, 0 = target.
    truth_target = (args.shift == "target")
    env_name = "TARGET" if truth_target else "SOURCE"
    ideal = "P(target)>0.5" if truth_target else "P(target)<0.5"

    print(f"[traj] collecting one trajectory: policy={os.path.basename(os.path.dirname(args.weight))}, "
          f"env={env_name}, require={args.require}")
    (s, a, ns, r), used_seed = collect_one(args.weight, args.seed, args.shift, args.require)
    T = len(s)
    dist = needle_ring_dist(s)
    contact = dist < args.contact_thresh
    success_step = int(np.argmax(r > 0)) if (r > 0).any() else -1
    print(f"[traj] len={T}, success at step {success_step}, contact steps={int(contact.sum())}")

    d = load_disc(args.disc, device)
    pt_sa, pt_sas, logr = score(d, s, a, ns, device)
    # contact discriminator, only meaningful on contact steps
    have_contact_disc = os.path.exists(args.disc_contact)
    if have_contact_disc:
        dc = load_disc(args.disc_contact, device)
        pt_sa_c, pt_sas_c, logr_c = score(dc, s, a, ns, device)

    # ---- summary numbers ----
    # truth=source -> mislabel when P(target)>0.5; truth=target -> when <0.5.
    mis = (pt_sas < 0.5) if truth_target else (pt_sas > 0.5)
    wrong_label = "SOURCE" if truth_target else "TARGET"
    print(f"[traj] full disc: {mis.mean()*100:.1f}% of steps mislabeled as {wrong_label}")
    if contact.sum():
        print(f"       contact steps mislabeled: {mis[contact].mean()*100:.1f}%, "
              f"free-space mislabeled: {mis[~contact].mean()*100:.1f}%")

    # ---- plot ----
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, pt_sas, label="P(target) d_sas (joint)", color="C3", lw=2)
    ax.plot(t, pt_sa, label="P(target) d_sa (s,a only)", color="C0", lw=1.2, alpha=0.8)
    if have_contact_disc:
        ax.plot(t, pt_sas_c, label="P(target) contact-disc", color="C2", lw=1.0, ls="--", alpha=0.8)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.fill_between(t, 0, 1, where=mis, color="red", alpha=0.10, label=f"mislabeled→{wrong_label.lower()}")
    ax.set_ylabel("P(target)")
    ax.set_ylim(0, 1)
    pol = os.path.basename(os.path.dirname(args.weight))
    ax.set_title(f"{pol} in {env_name} env (truth={env_name} → ideal {ideal} everywhere) | "
                 f"seed={used_seed}, success={'yes' if success_step>=0 else 'NO'}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(t, logr, color="C4", lw=2, label="log dyn-ratio (tgt/src) full")
    if have_contact_disc:
        ax.plot(t, logr_c, color="C2", lw=1.0, ls="--", alpha=0.8, label="log dyn-ratio contact-disc")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_ylabel("log p_tgt/p_src")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(t, dist, color="C1", lw=2, label="needle–ring dist")
    ax.axhline(args.contact_thresh, color="C1", ls=":", lw=1, label=f"contact thresh {args.contact_thresh}")
    ax.fill_between(t, 0, dist.max() * 1.05, where=contact, color="C1", alpha=0.12, label="contact phase")
    if success_step >= 0:
        for a_ in axes:
            a_.axvline(success_step, color="green", lw=1.5, ls="-", alpha=0.7)
        ax.text(success_step, dist.max() * 0.9, " success", color="green", fontsize=9)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("env step")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

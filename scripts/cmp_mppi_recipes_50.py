"""Headless 50-episode A/B of the MPPI-ADAPTED SimDist model under two planner
recipes, to solidify the "broad-search wanders in free space" hypothesis:
  - broad   = apply_original_recipe (init_std 1.0, 100/350 policy seeds)
  - trust   = ibrl-retuned defaults (mixture 0.8, init_std 0.15)
Same adapted ckpt, same 50 seeds, TARGET env, sigma 0.1. No rendering.
For each episode records success / min needle-ring dist / ring shove, then
classifies failures as free-space wander (minD > FREE_THRESH) vs contact-phase
(minD <= FREE_THRESH). Prints a comparison table.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.eval_simdist_policy_only as ev
from rl.simdist_planner import apply_original_recipe, MppiPlanner
from env.robosuite_wrapper import PixelRobosuite

STAGE1 = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt"
ADAPTED = os.environ.get("ADAPTED", "exps/mimicgen/threading_simdist_adapt_seed1/model0.pt")
N_EP = int(os.environ.get("N_EP", "50"))
MAX_STEPS = 400
SEED0 = 100                 # seeds 100..100+N_EP-1 (superset of the 20-viz seeds)
FREE_THRESH = 0.10         # minD > 0.10 m => needle never got near the ring


def build_env():
    _, ws = ev.build(STAGE1, use_planner=1)
    params = dict(ws.eval_env_params)
    params["hand_init_perturb_joint_sigma"] = 0.1
    params["device"] = "cuda"
    env = PixelRobosuite(**params)
    raw = env.env
    nid = raw.sim.model.geom_name2id("needle_obj_needle")
    ring = [raw.sim.model.geom_name2id(n) for n in raw.sim.model.geom_names
            if "tripod_obj_ring" in n and "vis" not in n]
    return env, raw, nid, ring, ws


def build_agent(recipe):
    agent, _ = ev.build(STAGE1, use_planner=1)
    agent.load_state_dict(torch.load(ADAPTED))
    agent = agent.to("cuda")
    agent.cfg.use_planner = 1
    if recipe == "broad":
        apply_original_recipe(agent.planner.cfg)
        agent.planner = MppiPlanner(agent.model, agent.planner.cfg, agent.action_dim, device="cuda")
    p = agent.planner.cfg
    print(f"[{recipe}] num_samples={p.num_samples} base_override={p.num_base_override} "
          f"init_std={p.init_std} iters={p.iterations} temp={p.temperature} "
          f"mixture={p.mixture_coef:.3f}", flush=True)
    return agent


def run(recipe, env, raw, nid, ring):
    def ring_center():
        return raw.sim.data.geom_xpos[ring].mean(0).copy()
    agent = build_agent(recipe)
    rows = []
    for ep in range(N_EP):
        np.random.seed(SEED0 + ep)
        obs, _ = env.reset()
        ring0 = ring_center()
        succ, min_d, max_disp = False, 9.0, 0.0
        t = 0
        for t in range(MAX_STEPS):
            a = agent.act(obs, eval_mode=True)
            obs, r, term, success, _ = env.step(a)
            d = float(np.linalg.norm(raw.sim.data.geom_xpos[nid] - ring_center()))
            min_d = min(min_d, d)
            max_disp = max(max_disp, float(np.linalg.norm(ring_center() - ring0)))
            succ = succ or bool(success)
            if term:
                break
        rows.append((ep, succ, min_d, max_disp, t + 1))
        print(f"[{recipe}] ep{ep:2d}: succ={succ}  minD={min_d*1000:6.1f}mm  "
              f"ringD={max_disp*1000:6.1f}mm  steps={t+1}", flush=True)
    return rows


def summarize(name, rows):
    n = len(rows)
    succ = [r for r in rows if r[1]]
    fail = [r for r in rows if not r[1]]
    free = [r for r in fail if r[2] > FREE_THRESH]
    contact = [r for r in fail if r[2] <= FREE_THRESH]
    print(f"\n=== {name}: {len(succ)}/{n} success ({100*len(succ)/n:.0f}%) ===")
    print(f"  failures            : {len(fail)}")
    print(f"  free-space wander   : {len(free)}  (minD > {FREE_THRESH*1000:.0f}mm, needle never approached)")
    print(f"  contact-phase fail  : {len(contact)}  (minD <= {FREE_THRESH*1000:.0f}mm, reached ring, no insert)")
    return dict(n=n, succ=len(succ), free=len(free), contact=len(contact))


def main():
    env, raw, nid, ring, _ = build_env()
    print(f"\n### ADAPTED ckpt: {ADAPTED} | {N_EP} episodes | seeds {SEED0}..{SEED0+N_EP-1}\n")
    rows_broad = run("broad", env, raw, nid, ring)
    rows_trust = run("trust", env, raw, nid, ring)
    b = summarize("BROAD-SEARCH (apply_original_recipe)", rows_broad)
    t = summarize("TRUST-PROPOSAL (ibrl defaults)", rows_trust)
    print("\n########## SUMMARY ##########")
    print(f"{'recipe':<16}{'success':>10}{'free-wander':>14}{'contact-fail':>14}")
    print(f"{'broad-search':<16}{b['succ']}/{b['n']:>8}{b['free']:>14}{b['contact']:>14}")
    print(f"{'trust-proposal':<16}{t['succ']}/{t['n']:>8}{t['free']:>14}{t['contact']:>14}")
    print(f"\nsuccess delta   : {t['succ']-b['succ']:+d} episodes ({100*(t['succ']-b['succ'])/t['n']:+.0f}pt)")
    print(f"free-wander delta: {t['free']-b['free']:+d}  (broad {b['free']} -> trust {t['free']})")
    np.savez("/tmp/claude-1003/-home-Chenyang-ibrl/c75bb058-182a-404a-9624-e7bbc5232759/scratchpad/cmp50.npz",
             broad=np.array(rows_broad, dtype=object), trust=np.array(rows_trust, dtype=object))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

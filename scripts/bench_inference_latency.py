"""Per-control-step inference latency (appendix: Computational Cost, part C).

Measures obs -> action wall-clock for ONE environment (batch 1), the deployment
setting, including host->device of the obs and device->host of the action (the
`.cpu()` in agent.act is also the synchronization point, so wall-clock is exact).

Arms:
  simdist_mppi   : SimDist world model + MPPI, paper recipe (apply_original_recipe:
                   350 = 100 base + 250 random, 3 iters, 64 elites, H=5), K heads
  policy_head    : the same world model's policy head, no planning
  ibrl           : IBRL actor + prior + critic argmax (train_rl.load_model)
  pixel_*        : r18@224 pixel world model built from its yaml with RANDOM
                   weights (latency does not depend on weight values)

Also sweeps the MPPI cost knobs one at a time around the paper recipe
(K / num_samples / iterations / horizon) so the cost structure is visible.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_inference_latency.py --out <json>
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.simdist_planner import MppiPlanner, apply_original_recipe  # noqa: E402

STATE_S1 = "exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt"
PIXEL_YAML = "release/cfgs/mimicgen/threading_d0_pixel_simdist_r18_224.yaml"
IBRL_CKPTS = {
    # sim2sim IBRL baseline: prior = source expert fine-tuned on D_t (an IBRL actor)
    "ibrl_sim2sim": "exps/mimicgen/threading_ibrl_exp2_nosimdist_seed1/model0.pt",
}


def time_fn(fn, n_warm, n_iter):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        fn()                      # returns a CPU tensor => implicit sync
        ts[i] = (time.perf_counter() - t0) * 1e3
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99)), "mean_ms": float(ts.mean()),
            "n": n_iter}


def load_state_simdist(path):
    """Build the agent without constructing an env (Workspace would build one)."""
    import pyrallis
    from train_simdist import MainConfig
    from rl.simdist_agent import SimDistAgent
    cfg = pyrallis.load(MainConfig, open(os.path.join(os.path.dirname(path), "cfg.yaml")))
    sd = torch.load(path, map_location="cuda")
    obs_dim = sd["model.encoder.0.weight"].shape[1]   # first encoder Linear
    agent = SimDistAgent(True, (obs_dim,), None, 7, cfg.rl_camera, cfg.simdist, device="cuda")
    agent.load_state_dict(sd)
    agent.eval()
    return agent, obs_dim


def load_pixel_simdist(yaml_path):
    import pyrallis
    from train_simdist import MainConfig
    from rl.simdist_agent import SimDistAgent
    cfg = pyrallis.load(MainConfig, open(yaml_path))
    S = cfg.rl_image_size
    agent = SimDistAgent(False, (3, S, S), (9,), 7, cfg.rl_camera, cfg.simdist, device="cuda")
    agent.eval()
    return agent, cfg


def set_heads(model, heads_all, priors_all, k):
    model.dynamics_extra = nn.ModuleList(list(heads_all)[: k - 1])
    if len(priors_all) > 0:
        model.dynamics_prior = nn.ModuleList(list(priors_all)[:k])


def planner_cfg(base, **kw):
    p = copy.deepcopy(base)
    apply_original_recipe(p)
    for k, v in kw.items():
        if k == "num_samples":        # keep the paper's 100:250 base:random ratio
            p.num_base_override = int(round(v * 100 / 350))
        setattr(p, k, v)
    return p


def bench_simdist(agent, make_obs, tag, results, args, heads_all, priors_all):
    base = agent.cfg.planner
    Kmax = 1 + len(heads_all)
    obs = make_obs()

    def run(pcfg, k):
        set_heads(agent.model, heads_all, priors_all, k)
        agent.cfg.use_planner = 1
        agent.planner = MppiPlanner(agent.model, pcfg, agent.action_dim, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        r = time_fn(lambda: agent.act(obs, eval_mode=True), args.warm, args.iters)
        r["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
        return r

    # headline arms
    paper = planner_cfg(base)
    for k in sorted({1, Kmax}):
        r = run(paper, k)
        results.append({"arm": f"{tag}_mppi_K{k}", "K": k, "samples": paper.num_samples,
                        "iters": paper.iterations, "H": paper.horizon, **r})
        print(results[-1], flush=True)
    set_heads(agent.model, heads_all, priors_all, Kmax)
    agent.cfg.use_planner = 0
    r = time_fn(lambda: agent.act(obs, eval_mode=True), args.warm, args.iters)
    results.append({"arm": f"{tag}_policy_head", **r})
    print(results[-1], flush=True)
    if not args.sweep:
        return
    # one-knob-at-a-time sweeps around the paper recipe (K = Kmax)
    sweeps = {"K": [1, 2, 3, 5, 10], "num_samples": [100, 350, 1000, 3000, 10000],
              "iterations": [1, 3, 6, 12], "horizon": [1, 3, 5, 8]}
    for knob, vals in sweeps.items():
        for v in vals:
            if knob == "K":
                if v > Kmax:
                    continue
                pc, k = paper, v
            else:
                pc, k = planner_cfg(base, **{knob: v}), Kmax
            r = run(pc, k)
            results.append({"arm": f"{tag}_sweep_{knob}", knob: v, "K": k,
                            "samples": pc.num_samples, "iters": pc.iterations,
                            "H": pc.horizon, **r})
            print(results[-1], flush=True)
    set_heads(agent.model, heads_all, priors_all, Kmax)


def dump(path, meta, results):
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--sweep", type=int, default=1)
    ap.add_argument("--arms", default="state,pixel,ibrl")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cpu": os.popen("lscpu | grep 'Model name'").read().strip(),
            "iters": args.iters, "warm": args.warm,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    print(meta, flush=True)
    results = []
    arms = args.arms.split(",")

    if "state" in arms:
        agent, obs_dim = load_state_simdist(STATE_S1)
        m = agent.model
        heads_all, priors_all = list(m.dynamics_extra), list(getattr(m, "dynamics_prior", []))
        bench_simdist(agent, lambda: {"state": torch.randn(obs_dim)}, "state", results, args,
                      heads_all, priors_all)
        del agent
        torch.cuda.empty_cache()

    if "pixel" in arms:
        agent, cfg = load_pixel_simdist(PIXEL_YAML)
        S = cfg.rl_image_size
        cams = agent.rl_cameras

        def make_obs():
            o = {c: torch.randint(0, 255, (3, S, S), dtype=torch.uint8) for c in cams}
            o["prop"] = torch.randn(9)
            return o
        m = agent.model
        heads_all, priors_all = list(m.dynamics_extra), list(getattr(m, "dynamics_prior", []))
        a2 = copy.copy(args)
        a2.sweep = 0
        bench_simdist(agent, make_obs, "pixel_r18_224", results, a2, heads_all, priors_all)
        del agent
        torch.cuda.empty_cache()

    dump(args.out, meta, results)
    if "ibrl" in arms:
        import train_rl
        for tag, w in IBRL_CKPTS.items():
            agent, env, _ = train_rl.load_model(w, "cuda")
            agent.eval()
            obs_dim = env.observation_shape[0]
            x = torch.randn(obs_dim)
            # QAgent.act wants device tensors and unsqueezes the dict IN PLACE, so
            # build a fresh dict per call; the H2D copy is timed, as in SimDist.act
            r = time_fn(lambda: agent.act({"state": x.to("cuda")}, eval_mode=True),
                        args.warm, args.iters)
            results.append({"arm": tag, "act_method": agent.cfg.act_method, **r})
            print(results[-1], flush=True)
            del agent, env
            torch.cuda.empty_cache()

    dump(args.out, meta, results)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

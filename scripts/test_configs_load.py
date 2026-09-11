"""pyrallis-load every SimDist config so a placement error cannot reach the queue.

A YAML that parses fine can still be WRONG: `simdist.planner` and
`simdist.world_model` both have a `horizon: 5`, so a knob anchored on that line
can land in the wrong block. pyrallis then rejects it -- but only at job startup,
after the queue wait. This does that load locally in seconds.
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyrallis
import train_simdist

pats = sys.argv[1:] or ["release/cfgs/mimicgen/threading_d0_*simdist*.yaml"]
files = sorted({f for p in pats for f in glob.glob(p)})
# These configs target OTHER entry points (train_residual_simdist / the IBRL-on-
# frozen-base trainer), so they carry fields MainConfig does not have and are
# expected not to load here. Excluded so the gate stays a clean pass/fail.
_OTHER_ENTRYPOINTS = ("_ibrl_simdist.yaml", "_residual_simdist.yaml")
files = [f for f in files if not f.endswith(_OTHER_ENTRYPOINTS)]
assert files, f"no configs matched {pats}"
bad = []
for f in files:
    try:
        cfg = pyrallis.load(train_simdist.MainConfig, open(f))  # type: ignore
        wm = cfg.simdist.world_model
        print(f"OK   {os.path.basename(f):<52} enc={wm.encoder_type:<7} "
              f"heads={wm.num_dynamics_heads} H={wm.horizon} "
              f"aug=shift{wm.aug_shift_pad}/jit{wm.aug_color_jitter}/"
              f"blur{wm.aug_blur_sigma_max}/prop{wm.aug_prop_noise} "
              f"demo_off={cfg.target_demo_offset} rl_img={cfg.rl_image_size}")
    except Exception as e:
        bad.append((f, f"{type(e).__name__}: {str(e)[:160]}"))
        print(f"FAIL {os.path.basename(f)}\n     {type(e).__name__}: {str(e)[:200]}")
print(f"\n{len(files)-len(bad)}/{len(files)} configs load")
if bad:
    sys.exit(1)

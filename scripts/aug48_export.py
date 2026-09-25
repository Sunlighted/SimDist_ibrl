"""Export every §48 run's raw numbers into the repo (docs/aug48_*.csv).

WHY: all §48 runs live ONLY under /scratch, which the cluster purges on an
inactivity timer (docs/setup.md:140), and nothing went to wandb (`--use_wb 0`).
The checkpoints are ~11 GB, but the numbers the conclusions rest on -- every
in-training eval point and every independent n=500 result -- are a few hundred
KB. This writes them to version control so §48 stays auditable after a purge.

Every field is read from the run itself (cfg.yaml / train.log / eval artifact),
never from the launch scripts, so the manifest records what actually ran.
"""
import csv, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
_s = importlib.util.spec_from_file_location(
    "R", os.path.join(os.path.dirname(os.path.abspath(__file__)), "aug48_report.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

ROOT = "exps/mimicgen"
EVALDIR = "/scratch/yifanr/simdist/eval_staging"
CELLS = [("A", "aug48_A_s*"), ("B", "aug48_B_s*"), ("C", "aug48_C_s*"),
         ("K", "aug48k_C_s*"), ("L", "aug48k_B_s*"), ("SW", "aug48sw_lam*_s*")]

def cfg(d, key):
    try:
        m = re.search(rf"^{key}:\s*(.+)$", open(f"{d}/cfg.yaml").read(), re.M)
        return m.group(1).strip() if m else ""
    except OSError:
        return ""

def stage1_aug(stage1):
    """The augmentation of the stage-1 this run adapted from, read from ITS cfg."""
    d = os.path.dirname(stage1)
    try:
        t = open(f"{d}/cfg.yaml").read()
    except OSError:
        return "?"
    g = lambda k: (re.search(rf"^\s*{k}:\s*([0-9.]+)", t, re.M) or [None, "0"])[1]
    return (f"shift{g('aug_shift_pad')}/jit{g('aug_color_jitter')}/"
            f"blur{g('aug_blur_sigma_max')}/prop{g('aug_prop_noise')}")

def indep(tag):
    hits = sorted(glob.glob(f"{EVALDIR}/{tag}_*.txt"), key=os.path.getmtime)
    for f in reversed(hits):
        t = open(f, errors="ignore").read()
        m = re.findall(r"success rate:\s*[0-9.]+%\s*\((\d+)/(\d+)\)", t)
        e = re.findall(r"^\[EVAL ENV\].*$", t, re.M)
        if m:
            return int(m[-1][0]), int(m[-1][1]), f, (e[-1] if e else "")
    return "", "", "", ""

runs, curves = [], []
for cell, pat in CELLS:
    for d in sorted(glob.glob(f"{ROOT}/{pat}")):
        tag = os.path.basename(d)
        s1 = cfg(d, "stage1_ckpt")
        w = R.window(d)
        k, n, f, env = indep(tag)
        runs.append(dict(
            cell=cell, run=tag, run_dir=os.path.realpath(d),
            stage1=s1, stage1_heads_H=cfg(os.path.dirname(s1), "") or "",
            stage1_aug=stage1_aug(s1),
            bc_anchor_coef=cfg(d, "bc_anchor_coef"),
            demo_offset=cfg(d, "target_demo_offset"),
            max_transitions=cfg(d, "max_transitions"),
            reached=(w[3] if w else ""), window_complete=(w[4] if w else ""),
            window_mean=(round(w[0], 4) if w else ""), window_sd=(round(w[1], 4) if w else ""),
            window_n=(w[2] if w else ""),
            indep_success=k, indep_episodes=n,
            indep_rate=(round(k / n, 4) if n else ""), eval_file=f, eval_env=env))
        for t, sc, po in R.curve(d):
            curves.append(dict(run=tag, transitions=t, target_score=sc,
                               policy_only=("" if po != po else po)))

os.makedirs("docs", exist_ok=True)
for name, rows in (("docs/aug48_runs.csv", runs), ("docs/aug48_curves.csv", curves)):
    with open(name, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")
envs = {r["eval_env"] for r in runs if r["eval_env"]}
print(f"runs={len(runs)}  complete windows={sum(1 for r in runs if r['window_complete'] is True)}"
      f"  with indep eval={sum(1 for r in runs if r['indep_episodes'])}  distinct [EVAL ENV]={len(envs)}")

"""Keep one checkpoint per eval point, so any of them can be evaluated later.

Why this exists: `TopkSaver` keeps exactly ONE weights file per run and
OVERWRITES it (iron law #14), so by the time a run finishes, every intermediate
checkpoint that was ever written is gone and the surviving `model0.pt` is the
argmax over in-training eval points -- selection-biased and unreproducible.

This watcher does not touch the trainer. It polls `<save_dir>/train.log` and,
for every `[eval @ <phase> step N]` line it has not seen, copies the CURRENT
`latest.pt` to `snapshots/step<N>.pt` plus a sidecar recording the step, the
score printed for it, and a sha256.

Why that copy is the right weights: `log_and_save()` writes `latest.pt` BEFORE
it prints the eval line (train_simdist.py:1078 vs :1091), and eval points are
`log_per_step` apart (~2 h at stage-1 speed), so a poll that sees the line for
step N is copying exactly the weights that scored at N.

  setsid python scripts/snapshot_ckpts.py \
      --dirs exps/.../run_a exps/.../run_b \
      --out /user_data/Chenyang/ckpt_snapshots \
      </dev/null >logs/snapshot.log 2>&1 & disown
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import time

EVAL_RE = re.compile(r"\[eval @ (\w+) step (\d+)\][^\n]*?target_score=([0-9.]+)")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def log(msg):
    print(f"[snap {time.strftime('%F %T')}] {msg}", flush=True)


def _eval_steps(log_path):
    if not os.path.exists(log_path):
        return set()
    try:
        with open(log_path, "r", errors="replace") as f:
            return {int(m[1]) for m in EVAL_RE.findall(f.read())}
    except OSError:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="run save_dirs to watch")
    ap.add_argument("--out", required=True, help="snapshot root (use a big filesystem)")
    ap.add_argument("--src", default="latest.pt", help="file to copy at each eval point")
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls")
    ap.add_argument("--until_step", type=int, default=0,
                    help="exit once every dir has a snapshot at >= this step (0 = never)")
    ap.add_argument("--max_hours", type=float, default=48.0)
    ap.add_argument("--min_free_gb", type=float, default=40.0,
                    help="stop copying (but keep watching) below this much free space")
    ap.add_argument("--also_model0", type=int, default=1,
                    help="on startup, also preserve the current model0.pt -- TopkSaver "
                         "will overwrite it as soon as a better eval lands")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seen = {}
    for d in args.dirs:
        tag = os.path.basename(os.path.normpath(d))
        os.makedirs(os.path.join(args.out, tag), exist_ok=True)
        seen[d] = {int(m.group(1)) for m in
                   (re.match(r"step(\d+)\.pt$", f)
                    for f in os.listdir(os.path.join(args.out, tag))) if m}
        if seen[d]:
            log(f"{tag}: already have steps {sorted(seen[d])}")
        # ⚠️ DO NOT BACK-FILL. `latest.pt` holds ONE step -- the most recent eval
        # point -- because TopkSaver overwrote everything before it (that is the
        # whole reason this watcher exists). Copying it once per historical eval
        # line produces N byte-identical files under N different step names, i.e.
        # fabricated provenance. Measured the first time this ran: 8 snapshots per
        # arm, all sha256-identical. So every eval already in train.log at startup
        # is marked SEEN, except the newest, which `latest.pt` genuinely is.
        hist = _eval_steps(os.path.join(d, "train.log"))
        if hist:
            newest = max(hist)
            skipped = sorted(x for x in hist if x != newest and x not in seen[d])
            seen[d] |= set(skipped)
            if skipped:
                log(f"{tag}: NOT back-filling {skipped} -- those weights no longer "
                    f"exist on disk (TopkSaver overwrote them). Snapshots start at "
                    f"the newest eval point, {newest}.")

    if args.also_model0:
        for d in args.dirs:
            tag = os.path.basename(os.path.normpath(d))
            src = os.path.join(d, "model0.pt")
            dst = os.path.join(args.out, tag, "model0_at_start.pt")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                json.dump(dict(note="TopkSaver best-so-far at watcher start; it is the "
                                    "ARGMAX over in-training eval points, so it is "
                                    "selection-biased (iron law #14) -- do not report it "
                                    "as a headline number",
                               src=src, mtime=os.path.getmtime(src),
                               sha256=sha256(dst)),
                          open(dst.replace(".pt", ".json"), "w"), indent=1)
                log(f"{tag}: preserved model0.pt -> {dst}")

    t0 = time.time()
    while time.time() - t0 < args.max_hours * 3600:
        done = 0
        for d in args.dirs:
            tag = os.path.basename(os.path.normpath(d))
            lp = os.path.join(d, "train.log")
            if not os.path.exists(lp):
                continue
            try:
                with open(lp, "r", errors="replace") as f:
                    txt = f.read()
            except OSError:
                continue
            evals = {int(s): (ph, float(sc)) for ph, s, sc in EVAL_RE.findall(txt)}
            for step in sorted(evals):
                if step in seen[d]:
                    continue
                src = os.path.join(d, args.src)
                if not os.path.exists(src):
                    continue
                if free_gb(args.out) < args.min_free_gb:
                    log(f"ONLY {free_gb(args.out):.0f} GB free on {args.out} -- "
                        f"NOT copying {tag} step {step}. Free space or lower --min_free_gb.")
                    break
                dst = os.path.join(args.out, tag, f"step{step}.pt")
                tmp = dst + ".partial"
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
                ph, sc = evals[step]
                json.dump(dict(step=step, phase=ph, target_score=sc, run=d,
                               src=args.src, src_mtime=os.path.getmtime(src),
                               copied_at=time.time(), sha256=sha256(dst)),
                          open(dst.replace(".pt", ".json"), "w"), indent=1)
                seen[d].add(step)
                log(f"{tag}: step {step} (score {sc:.3f}) -> {dst}  "
                    f"[{free_gb(args.out):.0f} GB free]")
            if args.until_step and seen[d] and max(seen[d]) >= args.until_step:
                done += 1
        if args.until_step and done == len(args.dirs):
            log("every run reached --until_step; exiting")
            return
        time.sleep(args.poll)
    log("max_hours reached; exiting")


if __name__ == "__main__":
    main()

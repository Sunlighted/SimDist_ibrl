"""Which policy actually generated a demo dataset?

`TopkSaver` OVERWRITES model0.pt whenever a better checkpoint appears, so a
dataset collected mid-run records a `source_weight` path that no longer points
at the network that produced it. env_cfg.json stores no fingerprint, and a
squashed git history cannot recover the script state either -- so provenance
has to be established EMPIRICALLY.

Feed each recorded state back to the expert (eval_mode=True => deterministic)
and compare the action it would emit against the action actually recorded.
The newly-collected set (known to come from the current model0.pt in ibrl mode)
is the positive control: it must match to ~0.
"""
import os, sys
_REPO="/home/Chenyang/ibrl"; sys.path.insert(0,_REPO)
os.environ.setdefault("MUJOCO_GL","egl")
import dataclasses, h5py, numpy as np, torch
import common_utils
from common_utils import ibrl_utils as utils
import train_rl

def log(*a): print(*a,file=sys.stderr,flush=True)
EXPERT="exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt"
SETS={"ORIGINAL sigma0.02 (n=20)":("release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5",20),
      "NEW sigma0.02 (pos ctrl)":("release/data/mimicgen/threading_d0_target_sig002_extra/dataset.hdf5",20),
      "sigma0.1 pool (expert_n500)":("release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5",20)}

agent,_e,env_params=train_rl.load_model(os.path.join(_REPO,EXPERT),"cuda")
log(f"[expert] act_method={agent.cfg.act_method} bc_policies={len(getattr(agent,'bc_policies',[]))}")
for lab,(path,n) in SETS.items():
    with h5py.File(os.path.join(_REPO,path),"r") as f:
        g=f["data"]; ks=sorted(g.keys(),key=lambda k:int(k.split("_")[1]))[:n]
        S=[np.array(g[k]["obs"]["state"]) for k in ks]; A=[np.array(g[k]["actions"]) for k in ks]
    for mode in ("ibrl","rl"):
        agent.cfg.act_method=mode
        errs=[]
        with torch.no_grad(), utils.eval_mode(agent):
            for s,a in zip(S,A):
                st=torch.as_tensor(s[:len(a)],dtype=torch.float32,device="cuda")
                for t in range(len(a)):
                    pred=agent.act({"state":st[t].clone()},eval_mode=True).cpu().numpy()
                    errs.append(np.abs(pred-a[t]).max())
        errs=np.asarray(errs)
        log(f"  {lab:<34} act_method={mode:>4}: max|a_pred - a_rec|  mean={errs.mean():.4f} "
            f"median={np.median(errs):.4f}  frac<0.01={np.mean(errs<0.01)*100:5.1f}%  n={len(errs)}")
os._exit(0)

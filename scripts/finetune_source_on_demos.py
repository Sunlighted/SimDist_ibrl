"""EXP2 helper: BC-finetune the SOURCE QAgent's actor on the 20 TARGET demos, save
as a new QAgent ckpt (+ copy source cfg.yaml so load_frozen_agent can rebuild it).
Produces the 'source-finetuned-on-20-demos' policy used as the IBRL prior in the
no-SimDist ablation. Run:
  python scripts/finetune_source_on_demos.py --steps 5000 \
    --out exps/mimicgen/threading_source_ft20_seed1
"""
import os, sys, shutil, argparse
os.environ.setdefault("MUJOCO_GL","egl")
import torch, pyrallis, numpy as np
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from rl.q_agent import QAgent
from rl import replay
import train_rl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", default="exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    ap.add_argument("--demos", default="release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    ap.add_argument("--num_demo", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    src_cfg=pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(a.source),"cfg.yaml")))
    # a minimal SOURCE env just for shapes (state-based)
    env=PixelRobosuite(env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96, camera_names=[], rl_cameras=[],
        use_state=True, obs_stack=1, state_stack=1, prop_stack=1,
        domain_shift=DomainShiftConfig(mode="none"))
    agent=QAgent(True, env.observation_shape, env.prop_shape, env.action_dim, src_cfg.rl_camera, src_cfg.q_agent)
    agent.load_state_dict(torch.load(a.source)); agent=agent.to("cuda"); agent.train(True)

    # load the 20 TARGET demos into a bc replay
    rb=replay.ReplayBuffer(1,0.99,1,400,100000,use_bc=True)
    replay.add_demos_to_replay(rb, a.demos, num_data=a.num_demo, rl_cameras=[], use_state=1,
        obs_stack=1, state_stack=1, prop_stack=1, reward_scale=1, record_sim_state=False, is_demo=True)
    print(f"[ft] {rb.num_success} success demos; BC-finetuning source actor {a.steps} steps")
    for i in range(a.steps):
        b=rb.sample_bc(256,"cuda:0")
        m=agent.pretrain_actor_with_bc(b)
        if (i+1)%max(a.steps//5,1)==0: print(f"[ft] step {i+1}: bc_loss={m['pretrain/loss']:.4f}")
    os.makedirs(a.out, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(a.out,"model0.pt"))
    shutil.copy(os.path.join(os.path.dirname(a.source),"cfg.yaml"), os.path.join(a.out,"cfg.yaml"))
    print(f"[ft] saved -> {a.out}/model0.pt (+cfg.yaml)")
    sys.stdout.flush(); os._exit(0)

if __name__=="__main__": main()

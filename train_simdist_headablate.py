"""SimDist head-ablation: is DYNAMICS or VALUE the bottleneck for target adaptation?

Both experiments START from the SAME adapted ckpt (threading_simdist_adapt_seed1:
target-BC policy + target-adapted dynamics + FROZEN source encoder/reward/value),
re-initialize EXACTLY ONE head, and train ONLY that head OFFLINE on the 100k
TARGET diverse dataset. Everything else stays frozen at the adapt ckpt, so the
ONLY variable is the ablated head. Every `eval_every` steps we (a) run MPPI on
the TARGET env (`num_eval_episode`) and (b) log the head's OWN held-out quality
(so a null MPPI result is interpretable: was the head actually well-trained?).

  --head dynamics : reinit dynamics, train latent-consistency on TARGET (s,a,s')
                    segments; value stays the FROZEN source value.
  --head value    : reinit value, regress V(encode(s)) -> the STORED target-expert
                    critic Q (`values` field); dynamics stays the adapt dynamics.

Reference bar R = the loaded adapt ckpt's OWN MPPI-on-target success (measured
once at step 0, BEFORE any reinit). Interpretation:
  Exp1(dynamics) > R, Exp2(value) ~ R  => dynamics was the bottleneck.
  Exp2(value)    > R, Exp1 ~ R         => value was the bottleneck.
  both ~ R (with head quality verified good) => encoder/policy/MPPI is the cap.

Usage:
  python train_simdist_headablate.py --head dynamics \
      --init_ckpt exps/mimicgen/threading_simdist_adapt_seed1/latest.pt \
      --data_path release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5 \
      --save_dir exps/mimicgen/threading_simdist_ablate_dyn_seed1
  python train_simdist_headablate.py --head value ... \
      --save_dir exps/mimicgen/threading_simdist_ablate_val_seed1
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrallis
import torch
import h5py

import common_utils
from evaluate import run_eval, run_eval_mp
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer
from train_simdist import MainConfig, Workspace


@dataclass
class AblateConfig(common_utils.RunConfig):
    # arch source: a MainConfig cfg.yaml to REBUILD the agent + envs (the stage-1
    # dir; adapt_seed1 was derived from it, so its weights load onto this arch).
    base_cfg_dir: str = "exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1"
    # weights to load (the ADAPTED ckpt both experiments start from)
    init_ckpt: str = "exps/mimicgen/threading_simdist_adapt_seed1/latest.pt"
    save_dir: str = "exps/mimicgen/threading_simdist_ablate_dyn_seed1"
    # "dynamics" | "value" (single-head ablation), or the JOINT arms:
    # "dynamics_value" (dynamics + value, encoder frozen) and
    # "dynamics_value_encoder" (dynamics + value + encoder trainable). The joint
    # arms use the (s,a,s') sequence buffer and MC return-to-go value targets.
    head: str = "dynamics"
    # value-loss weight for the joint arms (dynamics uses consistency_coef).
    value_coef: float = 1.0
    # warm_start=0 (default): reinit the head -> train from scratch (the original
    #   "is this head, learned in isolation, the bottleneck" probe).
    # warm_start=1: KEEP the head's loaded (adapt/stage-1) weights and just train
    #   it MORE on target -> preserves the cross-head co-adaptation, the cleaner
    #   "does further training of this head help beyond R" probe. Step-0 MPPI == R.
    warm_start: int = 0
    seed: int = 1
    # OFFLINE target data
    data_path: str = "release/data/mimicgen/threading_d0_target_diverse_v2/dataset.hdf5"
    num_data: int = -1               # -1 = all 100k trajectories
    val_frac: float = 0.02           # held-out episode fraction for head-quality
    # training
    num_steps: int = 400000
    eval_every: int = 10000
    batch_size: int = 256
    lr: float = 3e-4
    # eval (MPPI on TARGET)
    num_eval_episode: int = 200
    mp_eval: int = 1
    num_eval_proc: int = 10
    original_mppi: int = 1
    eval_hand_init_perturb_joint_sigma: Optional[float] = None
    use_wb: int = 1

    def __post_init__(self):
        assert self.head in (
            "dynamics", "value", "dynamics_value", "dynamics_value_encoder"
        ), self.head


class AblateWorkspace:
    def __init__(self, cfg: AblateConfig):
        self.cfg = cfg
        common_utils.set_all_seeds(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        # --- rebuild agent + TARGET eval env from the base (stage-1) cfg ---
        base_cfg_path = os.path.join(cfg.base_cfg_dir, "cfg.yaml")
        self.base_cfg: MainConfig = pyrallis.load(MainConfig, open(base_cfg_path, "r"))  # type: ignore
        self.base_cfg.simdist.use_planner = 1  # MPPI for eval
        if cfg.original_mppi:
            from rl.simdist_planner import apply_original_recipe
            apply_original_recipe(self.base_cfg.simdist.planner)
            print("[ablate] using ORIGINAL SimDist MPPI recipe for eval")
        if cfg.eval_hand_init_perturb_joint_sigma is not None:
            self.base_cfg.eval_hand_init_perturb_joint_sigma = \
                cfg.eval_hand_init_perturb_joint_sigma
        self.ws = Workspace(self.base_cfg, from_main=False)
        self.agent = self.ws.agent
        print(f"[ablate] loading adapt ckpt (weights): {cfg.init_ckpt}")
        self.agent.load_state_dict(torch.load(cfg.init_ckpt))
        self.agent = self.agent.to("cuda")
        self.eval_env_params = self.ws.eval_env_params
        self.H = self.base_cfg.simdist.world_model.horizon

        # --- OFFLINE target data (train + held-out val) ---
        self._load_data()

        self.cfg_dict = {**cfg.__dict__}
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        self.stat = common_utils.MultiCounter(
            cfg.save_dir, bool(cfg.use_wb), wb_exp_name=cfg.wb_exp,
            wb_run_name=cfg.wb_run, wb_group_name=cfg.wb_group, config=self.cfg_dict,
        )
        self.saver = common_utils.TopkSaver(save_dir=cfg.save_dir, topk=1)

    # ------------------------------------------------------------ data
    def _load_data(self):
        cfg = self.cfg
        # dynamics + the JOINT arms need (s,a,s') sequences (dynamics consistency
        # + MC return-to-go value targets from the same episodes).
        if cfg.head in ("dynamics", "dynamics_value", "dynamics_value_encoder"):
            # (s,a,s') sequence buffer; split episodes into train / val.
            full = SequenceReplayBuffer(max_episodes=200000, device="cpu")
            load_source_replay_to_sequence_buffer(
                full, cfg.data_path, num_data=cfg.num_data,
                reward_scale=self.base_cfg.env_reward_scale)
            eps = full.episodes
            rng = np.random.default_rng(cfg.seed)
            rng.shuffle(eps)
            n_val = max(1, int(len(eps) * cfg.val_frac))
            self.val_buf = SequenceReplayBuffer(max_episodes=n_val + 1, device="cpu")
            self.train_buf = SequenceReplayBuffer(max_episodes=len(eps) + 1, device="cpu")
            self.val_buf.episodes = eps[:n_val]
            self.train_buf.episodes = eps[n_val:]
            self.val_buf.num_episode = n_val
            self.train_buf.num_episode = len(eps) - n_val
            print(f"[ablate|dyn] train {self.train_buf.size()} / val {self.val_buf.size()} episodes")
        else:
            # flat (state, STORED target-expert value) pairs; split train / val.
            states, values = _load_flat_states_values(cfg.data_path, cfg.num_data)
            n = states.shape[0]
            rng = np.random.default_rng(cfg.seed)
            perm = rng.permutation(n)
            n_val = max(256, int(n * cfg.val_frac))
            vi, ti = perm[:n_val], perm[n_val:]
            dev = "cuda"
            self.val_states = states[vi].to(dev); self.val_values = values[vi].to(dev)
            self.train_states = states[ti].to(dev); self.train_values = values[ti].to(dev)
            print(f"[ablate|val] train {ti.shape[0]} / val {vi.shape[0]} states "
                  f"| target-value range [{values.min():.3f},{values.max():.3f}] mean {values.mean():.3f}")

    # ------------------------------------------------------------ eval
    def eval_mppi(self, seed: int) -> float:
        random_state = np.random.get_state()
        try:
            if self.cfg.mp_eval:
                scores = run_eval_mp(
                    env_params=self.eval_env_params, agent=self.agent,
                    num_proc=self.cfg.num_eval_proc, num_game=self.cfg.num_eval_episode,
                    seed=seed, verbose=False)
            else:
                scores = run_eval(
                    env_params=self.eval_env_params, agent=self.agent,
                    num_game=self.cfg.num_eval_episode, seed=seed,
                    record_dir=None, verbose=False)
        finally:
            np.random.set_state(random_state)
        return float(np.mean(scores))

    @torch.no_grad()
    def head_quality(self) -> dict[str, float]:
        """Held-out quality of the trained head (so a null MPPI result is
        interpretable). dynamics: 1-step + H-step latent-consistency MSE.
        value: MSE(predict_value(encode(s)), stored target value)."""
        m = self.agent.model
        head = self.cfg.head
        seq_based = head in ("dynamics", "dynamics_value", "dynamics_value_encoder")
        joint = head in ("dynamics_value", "dynamics_value_encoder")
        if seq_based:
            out: dict[str, float] = {}
            sample = self.val_buf.sample(
                self.cfg.batch_size, self.H, device="cuda",
                return_value=joint, discount=self.base_cfg.discount)
            if joint:
                s, a, _, v_tgt = sample
            else:
                s, a, _ = sample
            z = m.encode(s[:, 0])
            step1 = None
            for t in range(self.H):
                z_next_enc = m.encode(s[:, t + 1])
                z_pred = m.next(z, a[:, t])
                mse = torch.nn.functional.mse_loss(z_pred, z_next_enc).item()
                if t == 0:
                    step1 = mse
                z = z_pred
            hstep = torch.nn.functional.mse_loss(z, m.encode(s[:, self.H])).item()
            out["quality/dyn_1step_mse"] = step1
            out["quality/dyn_Hstep_mse"] = hstep
            if joint:
                B = s.shape[0]
                z_all = m.encode(s.reshape(B * (self.H + 1), -1))
                pred = m.predict_value(z_all)
                tgt = v_tgt.reshape(-1)
                out["quality/val_mse"] = \
                    torch.nn.functional.mse_loss(pred, tgt).item()
                out["quality/val_pred_mean"] = float(pred.mean())
                out["quality/val_tgt_mean"] = float(tgt.mean())
            return out
        else:
            idx = torch.randint(0, self.val_states.shape[0], (4096,), device="cuda")
            z = m.encode(self.val_states[idx])
            pred = m.predict_value(z)
            tgt = self.val_values[idx]
            return {
                "quality/val_mse": torch.nn.functional.mse_loss(pred, tgt).item(),
                "quality/val_pred_mean": float(pred.mean()),
                "quality/val_tgt_mean": float(tgt.mean()),
            }

    # ------------------------------------------------------------ train
    def _train_step(self):
        head = self.cfg.head
        if head == "dynamics":
            s, a, _ = self.train_buf.sample(self.cfg.batch_size, self.H, device="cuda")
            return self.agent.update_dynamics_only(s, a, lr=self.cfg.lr)
        elif head in ("dynamics_value", "dynamics_value_encoder"):
            s, a, _, v = self.train_buf.sample(
                self.cfg.batch_size, self.H, device="cuda",
                return_value=True, discount=self.base_cfg.discount)
            fn = (self.agent.update_dynamics_value
                  if head == "dynamics_value"
                  else self.agent.update_dynamics_value_encoder)
            return fn(s, a, v, lr=self.cfg.lr, value_coef=self.cfg.value_coef)
        else:
            idx = torch.randint(0, self.train_states.shape[0],
                                (self.cfg.batch_size,), device="cuda")
            return self.agent.update_value_only(
                self.train_states[idx], self.train_values[idx], lr=self.cfg.lr)

    def run(self):
        cfg = self.cfg
        # reference bar R = the ADAPT ckpt's own MPPI (BEFORE reinit)
        R = self.eval_mppi(seed=0)
        print(f"[ablate] reference R (adapt ckpt MPPI on target) = {R:.3f}")
        self.stat.reset()
        self.stat["ablate/reference_R"].append(R)
        self.stat["score/mppi_target"].append(R)  # x=0 point
        self.stat.summary(0)

        # reinit (from scratch) OR warm-start (keep adapt weights, train more)
        if cfg.warm_start:
            print(f"[ablate] WARM-START head '{cfg.head}' (keep adapt weights, train more)")
        else:
            # reinit the trainable value/dynamics sub-heads from scratch. The
            # encoder (shared) is NEVER reinitialized -- that would be a full
            # distillation restart; the encoder arm only ever FINE-TUNES it,
            # so warm_start=1 is the intended setting for the encoder arm.
            if cfg.head in ("dynamics", "value"):
                self.agent.reinit_head(cfg.head)
            else:
                self.agent.reinit_head("dynamics")
                self.agent.reinit_head("value")
                if cfg.head == "dynamics_value_encoder":
                    print("[ablate] NOTE: encoder is fine-tuned from adapt weights "
                          "even under warm_start=0 (not reinitialized)")
        self.agent.train()

        eval_at = set(range(cfg.eval_every, cfg.num_steps + 1, cfg.eval_every))
        for step in range(1, cfg.num_steps + 1):
            m = self._train_step()
            for k, v in m.items():
                self.stat[k].append(v)
            if step in eval_at:
                score = self.eval_mppi(seed=step)
                q = self.head_quality()
                self.stat["score/mppi_target"].append(score)
                for k, v in q.items():
                    self.stat[k].append(v)
                saved = self.saver.save(self.agent.state_dict(), score, save_latest=True)
                self.stat.summary(step)
                self.stat.reset()
                qs = " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in q.items())
                print(f"[{cfg.head}] step {step}: MPPI_target={score:.3f} (R={R:.3f})  "
                      f"{qs}  saved?={saved}", flush=True)


def _load_flat_states_values(data_path: str, num_data: int):
    """Read the diverse hdf5 -> (states[N,obs], values[N]) flat tensors. `values`
    is the STORED per-state target-expert critic Q (min-ensemble Q(s, pi(s)))."""
    f = h5py.File(data_path, "r")
    g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))
    if num_data is not None and num_data > 0:
        ids = ids[:num_data]
    print(f"[flat] loading states+values from {len(ids)} trajectories")
    all_s, all_v = [], []
    for ep_id in ids:
        ep = g[ep_id]
        if "values" not in ep:
            raise KeyError(f"{ep_id} has no `values` field (need the stored critic Q)")
        s = np.array(ep["obs"]["state"], dtype=np.float32)
        v = np.array(ep["values"], dtype=np.float32)
        n = min(s.shape[0], v.shape[0])
        all_s.append(s[:n]); all_v.append(v[:n])
    states = torch.from_numpy(np.concatenate(all_s, axis=0))
    values = torch.from_numpy(np.concatenate(all_v, axis=0))
    return states, values


def main():
    cfg = pyrallis.parse(config_class=AblateConfig)  # type: ignore
    ws = AblateWorkspace(cfg)
    ws.run()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()

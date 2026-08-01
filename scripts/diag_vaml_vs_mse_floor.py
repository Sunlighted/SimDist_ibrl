"""VAML vs MSE: dynamics 的 VALUE 预测有没有比 STATE 预测更低的不可约下界?

用户猜想:如果到 adapt 阶段才把 loss 从 MSE 换成 VAML 收益不大,是不是因为 world model 的
latent/dynamics 从 source 蒸馏时就是为"重建 state(MSE)"组织的,而非为"预测 value(VAML)"组织?
=> 如果从头用 VAML 训 dynamics,它能不能在【value 预测】上达到比 MSE 更低的 held-out 下界?

这直接判定"换 loss 有没有救":
- VAML 的 value-MAE << MSE 的 value-MAE  => dynamics 能牺牲 state 精度换 value 精度, 有优化空间,
  用户猜想(source 就该用 VAML)值得投入。
- 两者 value-MAE 相近                    => value 预测下界由数据/接触物理决定, 与 loss 无关,
  换 loss 无救(与今天"VAML vs MSE 无显著差异"的成绩结论一致)。

设置:raw-state dynamics(encoder_type=identity,无 learned latent 混淆,干净隔离物理下界)。
两个 arm 除 loss 外完全相同(同数据 v5 / 同 split / 同架构 / 同步数 / 同 lr)。
  MSE  : 多步 latent(=z-scored state) rollout MSE   —— 预测 next-state
  VAML : 多步 rollout 后 V_hat(decode(z_H)) 对真值 V_hat(s_H_real) 的 MSE —— 预测 value
V_hat = target 专家 critic 的 min-ensemble Q(s, pi(s)) (raw state 输入, 冻结)。
held-out 同时报 state-MAE 和 value-MAE, 两个 arm 可比。
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pyrallis

sys.path.insert(0, "/home/Chenyang/ibrl")
os.environ.setdefault("MUJOCO_GL", "egl")

from rl.world_model import WorldModel, WorldModelConfig  # noqa: E402
from rl.sequence_replay import SequenceReplayBuffer, load_source_replay_to_sequence_buffer  # noqa: E402
from train_dynamics import split_train_val  # noqa: E402

ERR = sys.stderr


def log(*a):
    print(*a, file=ERR, flush=True)


def load_value_agent(path, obs_dim, action_dim):
    import train_rl
    from rl.q_agent import QAgent
    vcfg = pyrallis.load(train_rl.MainConfig, open(os.path.join(os.path.dirname(path), "cfg.yaml")))  # type: ignore
    # 用 env-free 的 shape (raw state); QAgent state-path 只需要 obs_shape/action_dim
    ag = QAgent(True, (obs_dim,), None, action_dim, vcfg.rl_camera, vcfg.q_agent)
    ag.load_state_dict(torch.load(path))
    ag.train(False)
    for p in ag.parameters():
        p.requires_grad_(False)
    return ag.to("cuda")


def build_model(obs_dim, action_dim, train_buf, args):
    cfg = WorldModelConfig(
        encoder_type="identity", mlp_hidden_dim=args.hidden_dim, num_layer=args.num_layer,
        layer_norm=1, horizon=args.horizon, rho=args.rho, activation="relu", grad_clip=20.0,
    )
    m = WorldModel(obs_dim, action_dim, cfg).to("cuda")
    mean, std = train_buf.state_stats()
    m.set_normalization(mean, std)
    return m


def mse_loss(model, states, actions, rho):
    H = states.shape[1] - 1
    z = model.encode(states[:, 0]); loss = wsum = 0.0
    for t in range(H):
        w = rho ** t; wsum += w
        with torch.no_grad():
            z_next = model.encode(states[:, t + 1])
        z = model.next(z, actions[:, t])
        loss = loss + w * F.mse_loss(z, z_next)
    return loss / wsum


def value_grad(vagent, s):
    """可导的 V(s) = min_k Q_k(s, pi_rl(s))。不能用 vagent.value()(有 @torch.no_grad)。
    critic/actor 参数冻结,但对输入 s 的梯度要保留(VAML 要经 s 传回 dynamics)。"""
    a = vagent.actor({"state": s}, 0.0).mean          # 冻结 actor, 但对 s 可导
    q = vagent.critic.forward_k(s, a)                 # [B, num_q]
    return q.min(-1)[0].view(-1)


def vaml_loss(model, vagent, states, actions, rho):
    """rollout 到每一步, decode 回 raw state, 用冻结 V_hat 评估, 对真值 V_hat(真实 state) MSE。"""
    H = states.shape[1] - 1
    z = model.encode(states[:, 0]); loss = wsum = 0.0
    for t in range(H):
        w = rho ** t; wsum += w
        z = model.next(z, actions[:, t])
        s_pred = model.decode(z)                                # 反归一化 -> raw state (可导)
        v_pred = value_grad(vagent, s_pred)
        with torch.no_grad():
            v_true = value_grad(vagent, states[:, t + 1])
        loss = loss + w * F.mse_loss(v_pred, v_true)
    return loss / wsum


@torch.no_grad()
def evaluate(model, vagent, states, actions):
    """held-out: 逐步 rollout, 报 state-MAE 和 value-MAE (1 步 & H 步)。"""
    H = states.shape[1] - 1
    z = model.encode(states[:, 0])
    smae, vmae = [], []
    for t in range(H):
        z = model.next(z, actions[:, t])
        s_pred = model.decode(z)
        smae.append((s_pred - states[:, t + 1]).abs().mean().item())
        vp = value_grad(vagent, s_pred)
        vt = value_grad(vagent, states[:, t + 1])
        vmae.append((vp - vt).abs().mean().item())
    return smae, vmae


def train_arm(name, loss_kind, model, vagent, train, val, args):
    opt = torch.optim.Adam(model.dynamics.parameters(), lr=args.lr)
    for step in range(1, args.num_steps + 1):
        s, a, _ = train.sample(args.batch_size, args.horizon, "cuda")
        loss = (mse_loss(model, s, a, args.rho) if loss_kind == "mse"
                else vaml_loss(model, vagent, s, a, args.rho))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 20.0)
        opt.step()
        if step % args.log_every == 0 or step == args.num_steps:
            vs, va, _ = val.sample(args.val_batch, args.horizon, "cuda")
            smae, vmae = evaluate(model, vagent, vs, va)
            log(f"  [{name}] step {step}: loss {loss.item():.5f} | "
                f"state-MAE 1={smae[0]:.4f} H={smae[-1]:.4f} | "
                f"★value-MAE 1={vmae[0]:.4f} H={vmae[-1]:.4f}")
    return smae, vmae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="release/data/mimicgen/threading_d0_target_diverse_v5/dataset.hdf5")
    ap.add_argument("--value_agent", default="exps/mimicgen/threading_ibrl_sig01_snapshots_seed1/model0.pt")
    ap.add_argument("--num_data", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_batch", type=int, default=4096)
    ap.add_argument("--num_steps", type=int, default=60000)
    ap.add_argument("--log_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    full = SequenceReplayBuffer(max_episodes=10_000_000)
    load_source_replay_to_sequence_buffer(full, args.data_path, num_data=args.num_data)
    train, val = split_train_val(full, 0.05, args.seed); del full
    obs_dim = train.episodes[0]["state"].shape[1]
    action_dim = train.episodes[0]["action"].shape[1]
    log(f"[data] train {train.size()} ep / {train.num_transitions()} tr | val {val.size()} ep | "
        f"obs={obs_dim} act={action_dim}")

    vagent = load_value_agent(args.value_agent, obs_dim, action_dim)
    log(f"[value] {args.value_agent}")

    # V_hat 在真实 val 状态上的 spread (给 value-MAE 一个参照尺度)
    vs, va, _ = val.sample(4096, args.horizon, "cuda")
    with torch.no_grad():
        vspread = vagent.value({"state": vs[:, -1]}).std().item()
    log(f"[scale] V_hat 在真实 s_H 上的 std = {vspread:.4f} (value-MAE 应与之比较)\n")

    log("===== ARM MSE (预测 next-state) =====")
    m_mse = build_model(obs_dim, action_dim, train, args)
    smae_m, vmae_m = train_arm("MSE", "mse", m_mse, vagent, train, val, args)

    log("\n===== ARM VAML (预测 value) =====")
    torch.manual_seed(args.seed); np.random.seed(args.seed)  # 同初始化/同采样序列
    m_vaml = build_model(obs_dim, action_dim, train, args)
    smae_v, vmae_v = train_arm("VAML", "vaml", m_vaml, vagent, train, val, args)

    log("\n===== 判读 (held-out, H 步) =====")
    log(f"  V_hat spread (真实 s_H)   : {vspread:.4f}")
    log(f"  MSE  arm:  state-MAE {smae_m[-1]:.4f}   value-MAE {vmae_m[-1]:.4f}")
    log(f"  VAML arm:  state-MAE {smae_v[-1]:.4f}   value-MAE {vmae_v[-1]:.4f}")
    log(f"  value-MAE 比 (VAML/MSE) = {vmae_v[-1]/max(vmae_m[-1],1e-9):.2f}")
    if vmae_v[-1] < 0.7 * vmae_m[-1]:
        log("  ⇒ VAML 的 value 预测下界明显更低 ⇒ 换 loss(甚至 source 就用 VAML)有优化空间")
    else:
        log("  ⇒ 两者 value-MAE 相近 ⇒ value 预测下界由数据/接触物理决定, 与 loss 无关, 换 loss 无救")
    os._exit(0)


if __name__ == "__main__":
    main()

"""Tests for SequenceReplayBuffer's PIXEL extension.

The headline test is ALIGNMENT. Frames are stamped with their timestep index and
`state` is stamped with the same index, so any off-by-one between the image and
state windows shows up as a hard failure rather than as a mysteriously bad
world model 20 GPU-hours later. This matters because the two offline loaders
disagree about the terminal step: `load_source_replay_to_sequence_buffer`
duplicates s_{T-1} as s_T (rl/sequence_replay.py:328-329) while the online path
stores the true s_T -- so an image loader that pads differently desynchronises
exactly at the step where the only reward lives.

Also checks: uint8 is preserved (float32 would be 4x the memory for nothing),
legacy call sites are untouched, and the transition-based eviction works.

Run:  python scripts/test_replay_images.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.sequence_replay import SequenceReplayBuffer  # noqa: E402

CAMS = ("agentview", "robot0_eye_in_hand")
OBS, ACT, C, S = 37, 7, 3, 8       # small frames: this tests indexing, not vision
FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def stamped_obs(t, cam_offset=0):
    """Frame filled with the value (t + per-camera offset), mod 256."""
    return {c: torch.full((C, S, S), (t + i * 100 + cam_offset) % 256, dtype=torch.uint8)
            for i, c in enumerate(CAMS)}


def fill(buf, n_ep=6, T=12):
    for e in range(n_ep):
        # state[t][0] == t, so state and frames carry the SAME timestep stamp
        buf.new_episode(torch.full((OBS,), 0.0), obs=stamped_obs(0))
        for t in range(1, T + 1):
            buf.add(torch.full((OBS,), float(t)), torch.zeros(ACT), 1.0 if t == T else 0.0,
                    terminal=(t == T), success=True, next_obs=stamped_obs(t))
    return buf


def main():
    print("=== 1. alignment: obs[:, t] must be the same timestep as states[:, t] ===")
    buf = fill(SequenceReplayBuffer(max_episodes=100, image_keys=CAMS))
    np.random.seed(0)
    H = 3
    st, ac, rw, obs = buf.sample(64, H, device="cpu", return_obs=True)
    check(tuple(st.shape) == (64, H + 1, OBS), f"states {tuple(st.shape)}")
    for i, cam in enumerate(CAMS):
        check(tuple(obs[cam].shape) == (64, H + 1, C, S, S), f"{cam} {tuple(obs[cam].shape)}")
        check(obs[cam].dtype == torch.uint8, f"{cam} dtype {obs[cam].dtype} (must stay uint8)")
        # states[b,t,0] is the timestep; the frame is that timestep + camera offset
        want = (st[:, :, 0].long() + i * 100) % 256
        got = obs[cam][:, :, 0, 0, 0].long()
        check(bool(torch.equal(want, got)),
              f"{cam}: frame stamp == state stamp for all (b,t)"
              + ("" if torch.equal(want, got) else
                 f"  [state {st[0, :, 0].tolist()} vs frame {got[0].tolist()}]"))

    print("\n=== 2. an off-by-one IS detected (negative control) ===")
    # Shift one episode's frames by +1 and confirm the same assertion fires --
    # otherwise the test above proves nothing.
    bad = fill(SequenceReplayBuffer(max_episodes=100, image_keys=CAMS), n_ep=1)
    for k in CAMS:
        bad.episodes[0]["obs"][k] = torch.roll(bad.episodes[0]["obs"][k], 1, dims=0)
    st2, _, _, obs2 = bad.sample(16, H, device="cpu", return_obs=True)
    want2 = st2[:, :, 0].long() % 256
    got2 = obs2[CAMS[0]][:, :, 0, 0, 0].long()
    check(not bool(torch.equal(want2, got2)),
          "shifted frames are flagged as misaligned (test has teeth)")

    print("\n=== 3. legacy call sites unchanged (image_keys=()) ===")
    leg = SequenceReplayBuffer(max_episodes=100)
    for e in range(4):
        leg.new_episode(torch.zeros(OBS))
        for t in range(1, 9):
            leg.add(torch.full((OBS,), float(t)), torch.zeros(ACT), 0.0, terminal=(t == 8))
    out = leg.sample(8, 3, device="cpu")
    check(isinstance(out, tuple) and len(out) == 3, f"plain sample returns a 3-tuple (got {len(out)})")
    out4 = leg.sample(8, 3, device="cpu", return_mask=True)
    check(len(out4) == 4, f"return_mask -> 4-tuple (got {len(out4)})")
    out5 = leg.sample(8, 3, device="cpu", return_mask=True, return_value=True)
    check(len(out5) == 5, f"mask+value -> 5-tuple (got {len(out5)})")

    print("\n=== 4. transition-capped eviction ===")
    cap = SequenceReplayBuffer(max_episodes=10_000, image_keys=CAMS, max_transitions=40)
    fill(cap, n_ep=10, T=12)          # 120 transitions offered, cap 40
    check(cap.num_transitions() <= 40 + 12,
          f"num_transitions {cap.num_transitions()} respects cap 40 (+1 episode slack)")
    check(len(cap.episodes) < 10, f"evicted down to {len(cap.episodes)} episodes")
    st3, _, _, obs3 = cap.sample(8, H, device="cpu", return_obs=True)
    want3 = st3[:, :, 0].long() % 256
    got3 = obs3[CAMS[0]][:, :, 0, 0, 0].long()
    check(bool(torch.equal(want3, got3)), "alignment survives eviction")

    print("\n=== 4b. OFFLINE path (add_episode) -- the loaders use this ===")
    # Regression: add_episode originally attached no ep["obs"] and never touched
    # _n_trans, so sample(return_obs=True) raised KeyError and max_transitions was
    # a silent no-op for offline-loaded data. Tests 1-4 all use the ONLINE path
    # (new_episode/add) and so passed straight through both bugs.
    off = SequenceReplayBuffer(max_episodes=100, image_keys=CAMS, max_transitions=60)
    for e in range(8):
        T = 10
        st = torch.stack([torch.full((OBS,), float(t)) for t in range(T + 1)])
        ob = {c: torch.stack([torch.full((C, S, S), (t + i * 100) % 256, dtype=torch.uint8)
                              for t in range(T + 1)]) for i, c in enumerate(CAMS)}
        off.add_episode(st, torch.zeros(T, ACT), torch.zeros(T), success=True, obs=ob)
    check(off.num_transitions() <= 60 + 10,
          f"add_episode respects max_transitions ({off.num_transitions()} <= 60+slack)")
    st4, _, _, obs4 = off.sample(16, H, device="cpu", return_obs=True)
    check(True, "add_episode episodes are sampleable with return_obs")
    for i, cam in enumerate(CAMS):
        want = (st4[:, :, 0].long() + i * 100) % 256
        got = obs4[cam][:, :, 0, 0, 0].long()
        check(bool(torch.equal(want, got)), f"{cam}: offline-path alignment")

    print("\n=== 5. memory accounting ===")
    per = C * S * S
    print(f"  test frames: {per} B each; real 96x96x3 = {3*96*96} B "
          f"-> 2 cams = {2*3*96*96} B/transition "
          f"({2*3*96*96*400_000/1e9:.1f} GB at 400k transitions)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  - " + f)
        return 2
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Round-trip gate for the JPEG frame store (Phase B1).

Checks, against a RAW-stored control built from the same frames:
  1. legacy raw datasets still load byte-for-byte (no codec attr -> np.uint8);
  2. jpeg datasets survive generator -> merge -> loader -> sample -> decode;
  3. frame[t] still lines up with state[t] (the alignment the world model rides on);
  4. h5py's native shard copy preserves the codec/frame_shape attrs (a manual
     create_dataset walk drops them, which is what made this a test).
Run on a login node: CPU decode, seconds.
"""
import os, sys, tempfile
import h5py, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.sequence_replay import (SequenceReplayBuffer, JpegFrames,
                                read_frames_dataset, decode_jpeg_batch)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_simdist_data.py"))

CAM, T, C, H, W = "agentview", 12, 3, 96, 96
rng = np.random.default_rng(0)


def _smooth_frames(n):
    """Smooth gradients, not white noise: JPEG on noise is a pathological ratio."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((n, C, H, W), dtype=np.uint8)
    for t in range(n):
        base = (np.sin((xs + 3 * t) / 11.0) + np.cos((ys - 2 * t) / 13.0) + 2) * 60
        for c in range(C):
            out[t, c] = np.clip(base + 20 * c, 0, 255).astype(np.uint8)
    return out


def _write(path, frames, quality):
    import torchvision.io as tvio
    with h5py.File(path, "w") as hf:
        g = hf.create_group("data")
        d = g.create_group("demo_0")
        d.create_dataset("obs/state", data=np.arange(T * 4, dtype=np.float32).reshape(T, 4))
        if quality:
            enc = [tvio.encode_jpeg(torch.from_numpy(frames[t]), quality=quality).numpy()
                   for t in range(T)]
            ds = d.create_dataset(f"obs/{CAM}_image", shape=(T,),
                                  dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i, b in enumerate(enc):
                ds[i] = b
            ds.attrs["codec"] = "jpeg"
            ds.attrs["frame_shape"] = np.asarray((C, H, W), dtype=np.int32)
            ds.attrs["jpeg_quality"] = quality
        else:
            d.create_dataset(f"obs/{CAM}_image", data=frames, dtype="uint8")
        d.create_dataset("actions", data=np.zeros((T, 7), np.float32))
        d.create_dataset("rewards", data=np.eye(T, dtype=np.float32)[-1])
        d.create_dataset("expert_flags", data=np.ones(T, np.int8))
        d.attrs["num_samples"] = T
        d.attrs["success"] = True
        g.attrs["num_demos"] = 1
        g.attrs["num_success"] = 1


def main():
    frames = _smooth_frames(T)
    tmp = tempfile.mkdtemp(prefix="jpegtest_")
    raw_p, jpg_p, merged_p = (os.path.join(tmp, n) for n in
                              ("raw.hdf5", "jpg.hdf5", "merged.hdf5"))
    _write(raw_p, frames, 0)
    _write(jpg_p, frames, 95)

    # (1) legacy raw path unchanged
    with h5py.File(raw_p, "r") as f:
        got = read_frames_dataset(f["data/demo_0/obs"][f"{CAM}_image"])
    assert isinstance(got, np.ndarray) and got.dtype == np.uint8, type(got)
    assert np.array_equal(got, frames), "raw path is not byte-for-byte"
    print("[1] raw path byte-for-byte: OK")

    # (2) jpeg round-trip fidelity
    with h5py.File(jpg_p, "r") as f:
        jf = read_frames_dataset(f["data/demo_0/obs"][f"{CAM}_image"])
    assert isinstance(jf, JpegFrames), type(jf)
    dec = decode_jpeg_batch(jf[0:T], "cpu").numpy()
    mae = np.abs(dec.astype(np.int16) - frames.astype(np.int16)).mean()
    raw_b, enc_b = frames.nbytes, jf.nbytes_encoded()
    print(f"[2] jpeg q95: MAE={mae:.2f}/255, {raw_b/enc_b:.1f}x compression "
          f"({enc_b/T:.0f} B/frame vs {raw_b//T} raw)")
    assert mae < 3.0, f"jpeg too lossy: MAE {mae}"

    # (3) h5py native copy preserves the codec attrs (the merge_shards fix)
    with h5py.File(jpg_p, "r") as src, h5py.File(merged_p, "w") as out:
        g = out.create_group("data")
        src["data"].copy("demo_0", g, name="demo_0")
    with h5py.File(merged_p, "r") as f:
        ds = f["data/demo_0/obs"][f"{CAM}_image"]
        assert ds.attrs.get("codec") == "jpeg", "MERGE DROPPED THE CODEC ATTR"
        assert tuple(ds.attrs["frame_shape"]) == (C, H, W)
    print("[3] shard merge preserves codec/frame_shape attrs: OK")

    # (4) buffer -> sample -> decode, and frame[t] vs state[t] alignment
    outs = {}
    for tag, path in (("raw", raw_p), ("jpeg", jpg_p)):
        buf = SequenceReplayBuffer(max_episodes=10, image_keys=(CAM,))
        with h5py.File(path, "r") as f:
            ep = f["data/demo_0"]
            st = np.array(ep["obs"]["state"]).astype(np.float32)
            fr = read_frames_dataset(ep["obs"][f"{CAM}_image"])
            seq = np.concatenate([st, st[-1:]], axis=0)
            fs = (fr.pad_last(1) if isinstance(fr, JpegFrames)
                  else torch.from_numpy(np.concatenate([fr, fr[-1:]], axis=0)))
            buf.add_episode(torch.from_numpy(seq),
                            torch.zeros(T, 7), torch.zeros(T),
                            success=True, obs={CAM: fs})
        torch.manual_seed(0); np.random.seed(0)
        s, a, r, o = buf.sample(4, horizon=5, device="cpu", return_obs=True)
        outs[tag] = (s, o[CAM])
        assert o[CAM].shape == (4, 6, C, H, W), o[CAM].shape
        assert o[CAM].dtype == torch.uint8, o[CAM].dtype
    assert torch.equal(outs["raw"][0], outs["jpeg"][0]), "state sampling diverged"
    d = (outs["raw"][1].float() - outs["jpeg"][1].float()).abs().mean().item()
    print(f"[4] sample(): shapes/dtype match, same segments, frame MAE={d:.2f}/255")
    assert d < 3.0, f"sampled frames diverge too much: {d}"

    # (5) alignment: state[t] encodes t, so decoded frame[t] must be _smooth_frames[t]
    st_batch, fr_batch = outs["jpeg"]
    for b in range(st_batch.shape[0]):
        for t in range(st_batch.shape[1]):
            idx = int(round(st_batch[b, t, 0].item() / 4))     # state row -> timestep
            ref = frames[min(idx, T - 1)]
            err = np.abs(fr_batch[b, t].numpy().astype(np.int16) - ref.astype(np.int16)).mean()
            assert err < 4.0, f"frame/state desync at b={b} t={t} idx={idx}: MAE {err:.1f}"
    print("[5] frame[t] aligned with state[t] across the sampled window: OK")
    print("\nALL JPEG REPLAY CHECKS PASSED")


if __name__ == "__main__":
    main()

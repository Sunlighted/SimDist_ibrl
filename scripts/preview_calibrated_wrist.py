"""Audit saved trajectories and preview a flange-referenced wrist calibration."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import pyrallis
from PIL import Image, ImageDraw
import train_rl
from env.sim2real import restore_model_state, model_state_layout, mat2quat_wxyz
from scripts.rerender_simdist_data import render_episode
from scripts.generate_simdist_data import build_source_env, resolve_domain_shift, _RES_OVERRIDE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', default='figs/sim2real/wrist_calibrated')
    args = p.parse_args()
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {'shards': [], 'preview': []}
    for path in sorted(root.glob('part*.hdf5')):
        with h5py.File(path, 'r') as f:
            g = f['data']
            failures, transitions = [], 0
            layout = json.loads(g.attrs['model_state_layout'])
            model_dim = sum(int(np.prod(shape)) for _, shape in layout)
            for key in g:
                ep = g[key]
                required = ('sim_state', 'model_state', 'actions', 'obs/agentview_image')
                if any(name not in ep for name in required):
                    failures.append(key)
                    continue
                n = len(ep['actions'])
                transitions += n
                if len(ep['sim_state']) != n or len(ep['obs/agentview_image']) != n or ep['model_state'].shape != (model_dim,):
                    failures.append(key)
            row = dict(file=path.name, episodes=len(g), transitions=transitions,
                       model_dim=model_dim, failures=failures)
            report['shards'].append(row)
            print(json.dumps(row), flush=True)
    report['cell_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (root/'cell_snapshot.yaml', Path('release/cfgs/sim2real/threading_real_cell.yaml'))}
    with open('real2sim/scan_pipeline/ground_run2_solved.json') as f:
        transform = np.asarray(json.load(f)['T_ee_zedm'])
    with open('real2sim/zed_capture/scan_run2/intrinsics.json') as f:
        intr = json.load(f)['zedm']
    fovy = float(np.degrees(2*np.arctan(intr['resolution'][1]/(2*intr['left']['fy']))))
    report['camera'] = dict(T_flange_camera_opencv=transform.tolist(), fovy=fovy,
                            note='224 square, centered pinhole; native principal-point offset not modeled')
    with open('exps/mimicgen/threading_realcell_ibrl_seed1/cfg.yaml') as f:
        cfg = pyrallis.load(train_rl.MainConfig, f)
    _RES_OVERRIDE['rl_image_size'] = 224
    cams = ('robot0_eye_in_hand',)
    env, _ = build_source_env(cfg, 0., resolve_domain_shift(argparse.Namespace(domain='expert'), cfg), cams)
    env.reset()
    sim = env.env.sim
    cid = sim.model.camera_name2id(cams[0])
    bid = sim.model.body_name2id('robot0_right_hand')
    pid = int(sim.model.cam_bodyid[cid])
    # The sim hand is rotated -45 degrees from the physical flange (link8).
    # Recover the fixed hand -> reported flange transform from all real poses.
    from robosuite.models.robots import Panda
    from robosuite.utils.binding_utils import MjSim
    from scipy.spatial.transform import Rotation
    fk = MjSim.from_xml_string(Panda(idn=0).get_xml())
    offsets = []
    for state_path in sorted(Path('real2sim/zed_capture/scan_run2').glob('pose_*/state.json')):
        state = json.loads(state_path.read_text())
        for j, q in enumerate(state['joints']):
            fk.data.set_joint_qpos(f'robot0_joint{j+1}', q)
        fk.forward()
        base_id = fk.model.body_name2id('robot0_base')
        hand_id = fk.model.body_name2id('robot0_right_hand')
        br = fk.data.body_xmat[base_id].reshape(3,3)
        hr = br.T @ fk.data.body_xmat[hand_id].reshape(3,3)
        hp = br.T @ (fk.data.body_xpos[hand_id]-fk.data.body_xpos[base_id])
        offset = np.eye(4)
        offset[:3,:3] = hr.T @ Rotation.from_quat(state['ee_quat']).as_matrix()
        offset[:3,3] = hr.T @ (np.asarray(state['ee_pos'])-hp)
        offsets.append(offset)
    hand_to_flange = np.eye(4)
    hand_to_flange[:3,:3] = Rotation.from_matrix(np.stack([x[:3,:3] for x in offsets])).mean().as_matrix()
    hand_to_flange[:3,3] = np.mean([x[:3,3] for x in offsets], axis=0)
    report['camera']['T_sim_hand_reported_flange'] = hand_to_flange.tolist()
    transform = hand_to_flange @ transform
    report['camera']['T_sim_hand_camera_opencv'] = transform.tolist()
    tiles = []
    try:
        for path in (root/'part1.hdf5', root/'part8.hdf5'):
            with h5py.File(path, 'r') as f:
                stored_layout = json.loads(f['data'].attrs['model_state_layout'])
                assert stored_layout == [[name,list(shape)] for name,shape in model_state_layout(env.env)]
                keys = sorted(f['data'], key=lambda k: int(k.split('_')[1]))
                for key in (keys[0], keys[len(keys)//2]):
                    ep = f['data'][key]
                    states = np.asarray(ep['sim_state'])
                    ms = np.asarray(ep['model_state'])
                    indices = sorted(set([0, len(states)//2, len(states)-1]))
                    old = render_episode(env, states[indices], ms, cams)[cams[0]]
                    restore_model_state(env.env, ms)
                    sim.set_state_from_flattened(states[indices[0]])
                    sim.forward()
                    # Convert flange optical pose to the camera parent's local MuJoCo pose.
                    fr = sim.data.body_xmat[bid].reshape(3,3)
                    pr = sim.data.body_xmat[pid].reshape(3,3)
                    pos = sim.data.body_xpos[bid] + fr @ transform[:3,3]
                    sim.model.cam_pos[cid] = pr.T @ (pos-sim.data.body_xpos[pid])
                    sim.model.cam_quat[cid] = mat2quat_wxyz(pr.T @ fr @ transform[:3,:3] @ np.diag([1,-1,-1]))
                    sim.model.cam_fovy[cid] = fovy
                    calibrated = render_episode(env, states[indices], None, cams)[cams[0]]
                    for j,t in enumerate(indices):
                        raw = ep['obs/agentview_image'][t]
                        agent = Image.open(io.BytesIO(np.asarray(raw,dtype=np.uint8).tobytes())).convert('RGB')
                        row = Image.new('RGB', (672,250), 'white')
                        for col,im in enumerate((agent, Image.fromarray(old[j].transpose(1,2,0)), Image.fromarray(calibrated[j].transpose(1,2,0)))):
                            row.paste(im.resize((224,224)), (col*224,26))
                        ImageDraw.Draw(row).text((4,5), f'{path.stem}/{key} t={t} | agentview / default wrist / calibrated wrist', fill='black')
                        row.save(out/f'{path.stem}_{key}_t{t}.png')
                        tiles.append(row)
                    report['preview'].append(dict(shard=path.name, episode=key, indices=indices,
                                                  calibrated_std=float(calibrated.std())))
        sheet = Image.new('RGB', (1344,250*((len(tiles)+1)//2)), 'white')
        for i,tile in enumerate(tiles):
            sheet.paste(tile, ((i%2)*672,(i//2)*250))
        sheet.save(out/'comparison.png')
    finally:
        env.env.close()
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(str(out/'comparison.png'), flush=True)


if __name__ == '__main__':
    main()

"""Eight resumable shard workers; only closed, validated files become final shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--limit', type=int, default=4500)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--gpus', default='1,1,1,1,2,2,3,3')
    # The expert only supplies the env definition (task class + sim2real cell +
    # render size) that the stored states are replayed in. It MUST be the one
    # the shards were collected with -- a Threading expert on Square shards
    # would rebuild the wrong scene.
    p.add_argument('--expert', default='exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt')
    p.add_argument('--num_shards', type=int, default=8)
    a = p.parse_args()
    src, dst = Path(a.source), Path(a.dest)
    dst.mkdir(parents=True, exist_ok=True)
    calibration = Path(a.calibration).read_bytes()
    manifest = dict(source=str(src.resolve()), limit=a.limit, chunk=a.chunk,
                    **({} if a.expert.startswith('exps/mimicgen/threading_realcell_ibrl_seed1/')
                       else dict(expert=a.expert)),
                    calibration_sha256=hashlib.sha256(calibration).hexdigest(),
                    renderer_sha256=hashlib.sha256(Path('scripts/rerender_simdist_data.py').read_bytes()).hexdigest())
    manifest_path = dst/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Resume settings changed'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        (dst/'calibration.json').write_bytes(calibration)
    gpus = a.gpus.split(',')
    started = time.monotonic()

    def validate(path, expected):
        with h5py.File(path, 'r') as f:
            g = f['data']
            assert g.attrs.get('render_complete', False) and len(g) == expected
            assert json.loads(g.attrs['wrist_calibration']) == json.loads(calibration)['camera']
            for ep in g.values():
                n = len(ep['actions'])
                assert all(len(ep[k]) == n for k in ('sim_state','rewards','obs/state','obs/agentview_image','obs/robot0_eye_in_hand_image'))
                assert 'model_state' in ep

    def worker(index):
        source = src/f'part{index}.hdf5'
        final = dst/f'part{index}_2cam.hdf5'
        if final.exists():
            validate(final, a.limit)
            return
        pieces = dst/'chunks'/f'part{index}'
        pieces.mkdir(parents=True, exist_ok=True)
        paths = []
        for start in range(0, a.limit, a.chunk):
            count = min(a.chunk, a.limit-start)
            path = pieces/f'{start:05d}.hdf5'
            if not path.exists():
                command = [sys.executable, 'scripts/rerender_simdist_data.py',
                           '--dataset', str(source), '--out', str(path),
                           '--expert', a.expert,
                           '--cameras', 'robot0_eye_in_hand', '--copy_cameras', 'agentview',
                           '--rl_image_size', '224', '--jpeg_quality', '90',
                           '--start', str(start), '--limit', str(count),
                           '--wrist_calibration', str(dst/'calibration.json')]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[(index-1)%len(gpus)],
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with open(pieces/f'{start:05d}.log','w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate(path, count)
            paths.append(path)
            print(json.dumps(dict(shard=index, completed=start+count, total=a.limit,
                                  elapsed_seconds=round(time.monotonic()-started))), flush=True)
        temp = str(final)+'.partial'
        with h5py.File(temp,'w') as output, h5py.File(source,'r') as original:
            target = output.create_group('data')
            transitions = 0
            for path in paths:
                with h5py.File(path,'r') as f:
                    for name, value in f['data'].attrs.items():
                        target.attrs[name] = value
                    for key, ep in f['data'].items():
                        raw = original['data'][key]
                        for name in ('actions','rewards','sim_state','model_state','obs/state'):
                            assert np.array_equal(ep[name][:],raw[name][:]), (index,key,name)
                        for t in (0,len(ep['actions'])-1):
                            assert np.array_equal(ep['obs/agentview_image'][t],raw['obs/agentview_image'][t])
                        f['data'].copy(key,target)
                        transitions += len(ep['actions'])
            target.attrs['num_demos'] = a.limit
            target.attrs['num_transitions'] = transitions
        validate(temp, a.limit)
        os.replace(temp, final)
        print(json.dumps(dict(shard=index, status='complete', transitions=transitions)),flush=True)

    with ThreadPoolExecutor(max_workers=a.num_shards) as pool:
        list(pool.map(worker,range(1,a.num_shards+1)))
    (dst/'COMPLETE.json').write_text(json.dumps(dict(episodes=a.limit*a.num_shards,seconds=time.monotonic()-started),indent=2)+'\n')
    print('ALL SHARDS COMPLETE AND VALIDATED',flush=True)


if __name__ == '__main__':
    main()

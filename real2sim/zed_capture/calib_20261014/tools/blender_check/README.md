# One-frame Blender (Cycles) vs MuJoCo vs real check (2026-10-15)

Blender 4.5.3 LTS at `~/opt/blender-4.5.3-linux-x64/` (4.2 has no Blackwell kernels); MuJoCo 3.3.7 + usd-core in
the separate venv `~/opt/blender_bridge` (never pip into the ibrl env, iron rule 35). `$B` = any scratch dir.

```bash
python dump_scene.py $B                       # ibrl env (MuJoCo 2.3.2): nominal cell at the real home q -> scene.xml + state.json
~/opt/blender_bridge/bin/python to_usd.py $B  # MuJoCo 3: rebuild, assert poses == dump, export USD + meta.json
blender -b --python render_blender.py -- $B $B 64   # Cycles, MuJoCo-visible geoms only, cameras from MuJoCo, room lighting
python real_and_seg.py $B                     # ibrl env: rectified real crops + MuJoCo class masks
~/opt/blender_bridge/bin/python -I compare.py $B blender   # one fitted number (exposure on the scene board)
```
Result: `../../figs/blender_vs_mujoco_vs_real_1015.jpg` (rows scene / wrist; columns real / MuJoCo / Blender), numbers in
`../../README.md` "Blender check".

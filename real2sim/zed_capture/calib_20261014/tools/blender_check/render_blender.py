"""Blender Cycles render of the dumped Threading Robotiq cell, MuJoCo-identical geometry/cameras.
Lighting = a plausible ROOM (not tuned to the photo except one global exposure applied afterwards):
ceiling LED panels (area lights) above the bench + white-ish room walls for bounce + weak grey world."""
import bpy, sys, json, math, re
from mathutils import Matrix, Vector
a = sys.argv[sys.argv.index("--") + 1:]
B, OUT, SPP = a[0], a[1], int(a[2]) if len(a) > 2 else 64
meta = json.load(open(f"{B}/meta.json"))
G = {int(k): v for k, v in meta["geoms"].items()}
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=f"{B}/usd/frames/frame_1.usdc", import_cameras=False, import_lights=False,
                      import_materials=True)
sc = bpy.context.scene

def srgb2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def mat(name, rgb, rough, spec=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = tuple(srgb2lin(float(c)) for c in rgb[:3]) + (1.0,)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Metallic"].default_value = 0.0
    return m

kept = 0
for o in list(sc.objects):
    if o.type != "MESH":
        continue
    mm = re.search(r"_id(\d+)_geom$", o.name)
    if mm is None:                       # sites
        o.hide_render = True; continue
    g = G[int(mm.group(1))]
    if g["group"] == 0 or g["rgba"][3] < 0.05:      # robosuite hides group 0 (collision)
        o.hide_render = True; continue
    n = g["name"] or ""
    if n.startswith("wall") or n == "floor":
        rgb, rough = (0.62, 0.62, 0.63), 0.9      # room surfaces: matte, for bounce
    elif n == "table_visual":
        rgb, rough = g["rgba"], 0.35              # white board: semi-gloss melamine
    elif n.startswith(("needle", "tripod")):
        rgb, rough = g["rgba"], 0.70              # matte red paint
    else:
        rgb, rough = g["rgba"], 0.5
    o.data.materials.clear(); o.data.materials.append(mat(f"M_{o.name}", rgb, rough, 0.2 if n.startswith(("needle", "tripod")) else 0.5))
    o.data.shade_smooth() if not n.startswith(("table", "needle", "tripod", "wall", "floor")) else None
    kept += 1
print("kept geoms", kept)

# ---- room: ceiling panels ~2.6 m above the floor (work surface ~0.94 m), grid over the bench
table_z = max(G_["z"] for G_ in [{"z": 0.94}])
for i, (x, y) in enumerate([(x, y) for x in (-0.9, 0.0, 0.9) for y in (-0.9, 0.0, 0.9)]):
    L = bpy.data.lights.new(f"panel{i}", "AREA"); L.shape = "SQUARE"; L.size = 0.6; L.energy = 120.0
    L.color = (1.0, 0.98, 0.95)
    o = bpy.data.objects.new(f"panel{i}", L); sc.collection.objects.link(o)
    o.location = (x, y, 2.7)                       # pointing down by default
w = bpy.data.worlds.new("room"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.4, 0.4, 0.42, 1.0)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.15
sc.world = w

# ---- cameras straight from MuJoCo (camera looks along -z, y up: same as Blender)
for name, c in meta["cams"].items():
    cd = bpy.data.cameras.new(name); cd.sensor_fit = "VERTICAL"; cd.sensor_height = 24.0
    cd.lens = 12.0 / math.tan(math.radians(c["fovy"]) / 2); cd.clip_start = 0.005
    co = bpy.data.objects.new(name, cd); sc.collection.objects.link(co)
    R = Matrix([c["mat"][0:3], c["mat"][3:6], c["mat"][6:9]])
    co.matrix_world = Matrix.Translation(Vector(c["pos"])) @ R.to_4x4()

sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"; prefs.get_devices()
for d in prefs.devices: d.use = d.type == "OPTIX"
sc.cycles.device = "GPU"; sc.cycles.samples = SPP; sc.cycles.use_denoising = True
sc.cycles.denoiser = "OPENIMAGEDENOISE"
sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 8, 4
sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"; sc.view_settings.exposure = 0.0
sc.render.resolution_x = sc.render.resolution_y = 224
sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
sc.render.image_settings.color_management = "OVERRIDE"
sc.render.image_settings.view_settings.view_transform = "Standard"
import time
for name in meta["cams"]:
    sc.camera = bpy.data.objects[name]
    sc.render.filepath = f"{OUT}/blender_{name}.exr"
    t = time.time(); bpy.ops.render.render(write_still=True)
    print(f"RENDERED {name} {time.time() - t:.2f}s")

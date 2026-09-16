"""Render the rebuilt Rocky (official kit parts as rigid bodies) with the procedural stone shader in Cycles.

  Blender -b --python tools/blender/render_poses.py -- --poses poses.json --frames statue,stand,walk1 [--out docs/images]
Poses come from tools/analysis/export_poses.mjs (simulator body transforms); 'statue' uses the registered sculpture pose.
"""
import bpy, bmesh, sys, os, json, math
import mathutils
argv = sys.argv[sys.argv.index('--') + 1:]
def arg(n, d=None, c=str): return c(argv[argv.index(n) + 1]) if n in argv else d
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # simulator/
import numpy as np
_src = open(os.path.join(ROOT, 'tools', 'blender', 'bake_parts.py')).read()
_ns = {'bpy': bpy, 'bmesh': bmesh, 'math': math, 'os': os}
exec(_src[_src.index("def stone_material("):_src.index("def new_image(name, size, noncolor):")], _ns)
stone_material = _ns['stone_material']

def world_points(ob, step):
    co = np.empty(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)[::step]
    M = np.array(ob.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]

POSES = json.load(open(arg('--poses')))
ROBOT = json.load(open(os.path.join(ROOT, 'assets', 'robot.json')))
OUT = os.path.abspath(arg('--out', os.path.join(ROOT, 'docs', 'images')))
FRAMES = arg('--frames', 'statue,stand').split(',')
W, H = [int(x) for x in arg('--res', '1400x1000').split('x')]
SAMPLES = arg('--samples', 48, int)
S = POSES['scale']
os.makedirs(OUT, exist_ok=True)

scene = bpy.context.scene
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
scene.render.engine = 'CYCLES'
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences; prefs.compute_device_type = 'METAL'; prefs.get_devices()
    for d in prefs.devices: d.use = True
    scene.cycles.device = 'GPU'
except Exception as e: print('CPU render', e)
scene.cycles.samples = SAMPLES; scene.cycles.use_denoising = True
try: scene.cycles.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
scene.render.resolution_x = W; scene.render.resolution_y = H
scene.view_settings.view_transform = 'AgX'; scene.view_settings.look = 'AgX - Medium High Contrast'; scene.view_settings.exposure = -0.4

mat = stone_material('RockyStone', unit=S)
objs = {}
def part(name):
    if name in objs: return objs[name]
    bpy.ops.wm.stl_import(filepath=os.path.join(ROOT, 'kit', 'stl', name + '.stl'))
    ob = bpy.context.selected_objects[0]; ob.name = name
    bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4); bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(ob.data); bm.free()
    for p in ob.data.polygons: p.use_smooth = True
    ob.data.materials.append(mat); objs[name] = ob
    return ob

def M3(R): return mathutils.Matrix(R)
def set_pose_sim(name, pose):
    ob = part(name)
    Rw = M3(pose['R']); pw = mathutils.Vector(pose['t'])
    P = M3(pose['partToBody']['R']); pp = mathutils.Vector(pose['partToBody']['t'])
    lin = Rw @ (P * S); tr = Rw @ (pp * S) + pw
    m = lin.to_4x4(); m.translation = tr; ob.matrix_world = m; ob.hide_render = False

def statue_up_rotation():
    up = mathutils.Vector(ROBOT['torso']['statueUp']).normalized()
    return up.rotation_difference(mathutils.Vector((0, 0, 1))).to_matrix()

def set_pose_statue():
    U = statue_up_rotation()
    lowest = 1e9
    for name, pose in ROBOT['statuePose'].items():
        if name == '1-B': continue
        ob = part(name)
        lin = U @ (M3(pose['R']) * S); tr = U @ (mathutils.Vector(pose['t']) * S)
        m = lin.to_4x4(); m.translation = tr; ob.matrix_world = m; ob.hide_render = False
    bpy.context.view_layer.update()
    for name in ROBOT['statuePose']:
        if name == '1-B': continue
        lowest = min(lowest, float(world_points(objs[name], 25)[:, 2].min()))
    for name in ROBOT['statuePose']:
        if name == '1-B': continue
        objs[name].matrix_world.translation.z -= lowest

# ground, lights, world
bpy.ops.mesh.primitive_plane_add(size=40); ground = bpy.context.object; ground.name = 'ground'
gm = bpy.data.materials.new('floor'); gm.use_nodes = True; gb = gm.node_tree.nodes['Principled BSDF']
nz = gm.node_tree.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 40; nz.inputs['Detail'].default_value = 8
cr = gm.node_tree.nodes.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (0.030, 0.032, 0.033, 1); cr.color_ramp.elements[1].color = (0.058, 0.060, 0.060, 1)
gm.node_tree.links.new(nz.outputs['Fac'], cr.inputs['Fac']); gm.node_tree.links.new(cr.outputs['Color'], gb.inputs['Base Color'])
gb.inputs['Roughness'].default_value = 0.78
ground.data.materials.append(gm)
w = bpy.data.worlds.new('w'); scene.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.11, 0.125, 0.13, 1); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.12

def area(name, loc, target, energy, size, color=(1, 1, 1)):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size; L.color = color
    o = bpy.data.objects.new(name, L); scene.collection.objects.link(o)
    o.location = loc; o.rotation_euler = (mathutils.Vector(target) - mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
cam.data.lens = 55

def frame_and_render(tag, view=(1.0, -1.35, 0.72), fill=1.0):
    bpy.context.view_layer.update()
    vis = [o for o in objs.values() if not o.hide_render]
    pts = np.concatenate([world_points(o, 40) for o in vis])
    lo = mathutils.Vector(pts.min(0).tolist()); hi = mathutils.Vector(pts.max(0).tolist())
    ctr = (lo + hi) / 2; size = max(hi - lo)
    for o in [o for o in scene.objects if o.type == 'LIGHT']: bpy.data.objects.remove(o, do_unlink=True)
    area('key', ctr + mathutils.Vector((-1.6, -1.3, 2.0)) * size * 1.6, ctr, 95 * (size / 0.6) ** 2, size * 1.2, (1.0, 0.93, 0.84))
    area('fill', ctr + mathutils.Vector((1.8, -1.2, 0.6)) * size * 1.8, ctr, 22 * (size / 0.6) ** 2, size * 2.0, (0.86, 0.93, 1.0))
    area('rim', ctr + mathutils.Vector((0.6, 1.8, 1.3)) * size * 1.6, ctr, 70 * (size / 0.6) ** 2, size * 0.8, (0.9, 0.96, 1.0))
    d = mathutils.Vector(view).normalized()
    dist = size / (2 * math.tan(cam.data.angle / 2)) * 1.25 * fill + size * 0.5
    cam.location = ctr + d * dist; cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = os.path.join(OUT, f'rocky_{tag}.png')
    bpy.ops.render.render(write_still=True)
    print('rendered', tag, flush=True)

for fr in FRAMES:
    for o in objs.values(): o.hide_render = True
    if fr == 'statue':
        set_pose_statue(); frame_and_render('statue')
    else:
        for name, pose in POSES['frames'][fr]['parts'].items(): set_pose_sim(name, pose)
        frame_and_render(fr, view=(0.95, -1.4, 0.8) if not fr.startswith('walk') else (0.2, -1.6, 0.75))

"""Blender (4.2, Cycles) look-development and bake pipeline for Rocky's stone surface.

Run headless:
  Blender -b --python tools/blender/bake_parts.py -- --parts 3-B --size 2048 [--out bakes] [--preview]

For each official print part:
 1. import the full-resolution STL (high), weld, recompute outward normals;
 2. make a real-time copy (low): collapse-decimate to a target triangle count, smooth-by-angle shading,
    Smart UV Project + island packing;
 3. shade the high mesh with a layered procedural stone material evaluated in object space (print mm):
      macro tone field (FBM noise) · soft plate-scale variation · mineral grains (Voronoi) ·
      cavity darkening (AO) · convex-edge wear (Pointiness) · micro-pitting/grain/microcrack bump;
 4. bake high → low (selected-to-active): tangent-space normal (geometry + micro bump), albedo (diffuse colour
    only, no lighting), roughness, ambient occlusion;
 5. write the low mesh (positions, normals, MikkTSpace tangents, UVs, indices) as a compact binary + JSON header.
"""
import bpy, bmesh, sys, os, json, math, struct, time
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def arg(name, default=None, cast=str):
    return cast(argv[argv.index(name) + 1]) if name in argv else default
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # simulator/
SRC = os.path.join(ROOT, 'kit', 'stl')
OUT = os.path.abspath(arg('--out', os.path.join(ROOT, 'bakes')))
PARTS = arg('--parts', 'torso,1-A,1-B,1-C,2-A,2-B,3-A,3-B,4-A,4-B,5-A,5-B').split(',')
SIZE = arg('--size', 2048, int)
TORSO_SIZE = arg('--torso-size', 4096, int)
TARGET_TRIS = {'torso': 160000}
DEFAULT_TRIS = arg('--tris', 60000, int)
PREVIEW = '--preview' in argv
SAMPLES_AO = arg('--ao-samples', 128, int)
os.makedirs(OUT, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for d in prefs.devices: d.use = True
    scene.cycles.device = 'GPU'
except Exception as e:
    print('GPU unavailable, using CPU:', e)
scene.cycles.samples = 16
scene.render.bake.margin = 24
scene.render.bake.margin_type = 'EXTEND'

def clear_scene():
    for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)
    for img in list(bpy.data.images): bpy.data.images.remove(img)
    for mat in list(bpy.data.materials): bpy.data.materials.remove(mat)

def import_part(name):
    bpy.ops.wm.stl_import(filepath=os.path.join(SRC, name + '.stl'))
    ob = bpy.context.selected_objects[0]
    ob.name = name + '_high'
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data); bm.free()
    for p in ob.data.polygons: p.use_smooth = True
    return ob

def make_low(high, target):
    low = high.copy(); low.data = high.data.copy(); low.name = high.name.replace('_high', '_low')
    scene.collection.objects.link(low)
    ratio = min(1.0, target / max(1, len(high.data.polygons)))
    bpy.ops.object.select_all(action='DESELECT'); low.select_set(True); bpy.context.view_layer.objects.active = low
    if ratio < 0.999:
        mod = low.modifiers.new('dec', 'DECIMATE'); mod.decimate_type = 'COLLAPSE'; mod.ratio = ratio; mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
    # UV: Smart UV Project with a wide angle limit (fewer, larger islands on organic stone), then a concave
    # island pack. Select-sync is required in background mode or pack_islands silently packs nothing
    # (measured on 3-B: 13.7 % → 51 % texture coverage, i.e. ≈1.9× linear texel density).
    scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(78), island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    unpacked = uv_array(low.data); q0 = uv_quality(low.data)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(udim_source='CLOSEST_UDIM', rotate=True, rotate_method='ANY', scale=True, margin_method='FRACTION', margin=0.004, shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')
    q1 = uv_quality(low.data)
    # pack_islands collapses the torso (2 300 non-manifold edges): every UV triangle ends with zero area and the bake
    # writes nothing. Keep the Smart UV Project layout whenever packing makes coverage worse or creates degenerates.
    if q1['degenerate'] > 0.001 or q1['coverage'] < q0['coverage']:
        low.data.uv_layers.active.data.foreach_set('uv', unpacked)
        print(f'  {low.name}: pack_islands rejected {q1}, keeping smart-project UVs {q0}', flush=True)
    else:
        print(f'  {low.name}: UV coverage {q0["coverage"]:.3f} -> {q1["coverage"]:.3f}', flush=True)
    return low

def uv_array(me):
    uv = np.empty(len(me.uv_layers.active.data) * 2, np.float32); me.uv_layers.active.data.foreach_get('uv', uv)
    return uv

def uv_quality(me):
    """Texture-space coverage (sum of UV triangle areas inside the unit square) and degenerate-triangle fraction."""
    me.calc_loop_triangles()
    uv = uv_array(me).reshape(-1, 2)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('loops', tri); tri = tri.reshape(-1, 3)
    a, b, c = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    inside = np.all((uv[tri] >= -1e-6) & (uv[tri] <= 1 + 1e-6), axis=(1, 2))
    return dict(coverage=round(float(area[inside].sum()), 4), degenerate=round(float(np.mean(~(area > 1e-10))), 4))

def stone_material(name='RockyStone', unit=1.0):
    """Layered procedural stone in object space. Units: print millimetres (film scale ≈ ×11).

    Albedo is kept in a physically plausible range for dark weathered rock (linear 0.03–0.16, i.e. sRGB ≈ 50–110)
    and mostly neutral-warm, following the book ("blackish-brown to brown") and the film puppet's painted stone.
    Per-cell randomness uses White Noise of the Voronoi cell position (uniform statistics), not the Voronoi colour
    luminance, whose averaged RGB distribution almost never reaches tail thresholds.
    """
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; N = nt.nodes; L = nt.links
    for n in list(N): N.remove(n)
    out = N.new('ShaderNodeOutputMaterial'); out.location = (2200, 0)
    bsdf = N.new('ShaderNodeBsdfPrincipled'); bsdf.location = (1900, 0)
    bsdf.inputs['IOR'].default_value = 1.5
    L.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    tc = N.new('ShaderNodeTexCoord'); tc.location = (-2000, 0)
    obj = tc.outputs['Object']
    X = [-1700]
    def node(kind, loc, **props):
        n = N.new(kind); n.location = loc
        for k, v in props.items(): setattr(n, k, v)
        return n
    def noise(scale, detail, rough, loc, dist=0.0, lac=2.0):
        n = node('ShaderNodeTexNoise', loc); n.noise_dimensions = '3D'
        n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough; n.inputs['Distortion'].default_value = dist; n.inputs['Lacunarity'].default_value = lac
        L.new(obj, n.inputs['Vector']); return n
    def voronoi(scale, feature, loc, rand=1.0, smooth=None):
        v = node('ShaderNodeTexVoronoi', loc); v.voronoi_dimensions = '3D'; v.feature = feature
        v.inputs['Scale'].default_value = scale; v.inputs['Randomness'].default_value = rand
        if smooth is not None: v.inputs['Smoothness'].default_value = smooth
        L.new(obj, v.inputs['Vector']); return v
    def ramp(inp, stops, loc, interp='LINEAR'):
        r = node('ShaderNodeValToRGB', loc); r.color_ramp.interpolation = interp
        els = r.color_ramp.elements
        while len(els) > len(stops): els.remove(els[-1])
        while len(els) < len(stops): els.new(0.5)
        for e, (pos, col) in zip(els, stops):
            e.position = pos; e.color = (*col, 1.0) if len(col) == 3 else col
        L.new(inp, r.inputs['Fac']); return r
    def mix(a, b, fac, blend, loc):
        m = node('ShaderNodeMix', loc); m.data_type = 'RGBA'; m.blend_type = blend; m.clamp_result = True
        L.new(a, m.inputs[6])
        if isinstance(b, tuple): m.inputs[7].default_value = (*b, 1.0)
        else: L.new(b, m.inputs[7])
        if isinstance(fac, (int, float)): m.inputs['Factor'].default_value = fac
        else: L.new(fac, m.inputs['Factor'])
        return m
    def mth(op, a, b, loc, clamp=False):
        m = node('ShaderNodeMath', loc); m.operation = op; m.use_clamp = clamp
        for i, val in ((0, a), (1, b)):
            if isinstance(val, (int, float)): m.inputs[i].default_value = val
            else: L.new(val, m.inputs[i])
        return m
    def cell_random(vor, loc):
        w = node('ShaderNodeTexWhiteNoise', loc); w.noise_dimensions = '3D'
        L.new(vor.outputs['Position'], w.inputs['Vector']); return w
    # ---------- ALBEDO ----------
    # 1. broad tone: two FBM octave bands → cool grey-brown ↔ warm brown
    m1 = noise(0.035, 5, 0.55, (-1700, 900), dist=0.4)
    m2 = noise(0.14, 6, 0.6, (-1700, 650), dist=0.2)
    tone = ramp(m1.outputs['Fac'], [(0.32, (0.040, 0.036, 0.032)), (0.50, (0.078, 0.066, 0.054)), (0.70, (0.135, 0.113, 0.090))], (-1400, 900))
    mott = ramp(m2.outputs['Fac'], [(0.35, (0.78, 0.80, 0.80)), (0.65, (1.18, 1.12, 1.05))], (-1400, 650))
    c = mix(tone.outputs['Color'], mott.outputs['Color'], 1.0, 'MULTIPLY', (-1100, 800))
    # 2. granular texture: every ~0.25 mm grain gets a uniform random value → ±15 % luminance, slight hue drift
    g = voronoi(4.0, 'F1', (-1700, 350))
    gr = cell_random(g, (-1400, 350))
    gval = ramp(gr.outputs['Value'], [(0.0, (0.80, 0.80, 0.82)), (1.0, (1.16, 1.12, 1.08))], (-1150, 350))
    c = mix(c.outputs[2], gval.outputs['Color'], 1.0, 'MULTIPLY', (-850, 700))
    # 3. accessory minerals: 4 % light specks (quartz/feldspar), 5 % dark specks (mafic), by uniform thresholds
    light_mask = ramp(gr.outputs['Value'], [(0.972, (0, 0, 0)), (0.980, (1, 1, 1))], (-1150, 150), 'CONSTANT')
    dark_mask = ramp(gr.outputs['Value'], [(0.05, (1, 1, 1)), (0.06, (0, 0, 0))], (-1150, -50), 'CONSTANT')
    core = ramp(g.outputs['Distance'], [(0.0, (1, 1, 1)), (0.55, (0, 0, 0))], (-1150, -250))
    lm = mth('MULTIPLY', light_mask.outputs['Color'], core.outputs['Color'], (-850, 150))
    dm = mth('MULTIPLY', dark_mask.outputs['Color'], core.outputs['Color'], (-850, -50))
    c = mix(c.outputs[2], (0.20, 0.185, 0.16), lm.outputs[0], 'MIX', (-600, 600))
    c = mix(c.outputs[2], (0.010, 0.009, 0.008), dm.outputs[0], 'MIX', (-400, 600))
    # 4. cavity: local AO (1.0 mm) darkens crevices, plate seams, carvings
    ao = node('ShaderNodeAmbientOcclusion', (-850, 1250)); ao.samples = 16; ao.only_local = True; ao.inputs['Distance'].default_value = 1.0 * unit
    cav = ramp(ao.outputs['AO'], [(0.20, (0.33, 0.32, 0.31)), (0.80, (1, 1, 1))], (-600, 1250))
    c = mix(c.outputs[2], cav.outputs['Color'], 1.0, 'MULTIPLY', (-150, 600))
    # 5. convex wear: Pointiness × breakup noise → paler, dustier grey-tan
    geo = node('ShaderNodeNewGeometry', (-850, 1550))
    edge = ramp(geo.outputs['Pointiness'], [(0.515, (0, 0, 0)), (0.60, (1, 1, 1))], (-600, 1550))
    br = noise(0.8, 5, 0.7, (-850, 1800))
    brk = ramp(br.outputs['Fac'], [(0.38, (0, 0, 0)), (0.60, (1, 1, 1))], (-600, 1800))
    wear = mth('MULTIPLY', edge.outputs['Color'], brk.outputs['Color'], (-350, 1650), clamp=True)
    wear_amt = mth('MULTIPLY', wear.outputs[0], 0.6, (-150, 1650))
    c = mix(c.outputs[2], (0.215, 0.190, 0.155), wear_amt.outputs[0], 'MIX', (100, 600))
    L.new(c.outputs[2], bsdf.inputs['Base Color'])
    # ---------- ROUGHNESS ----------
    r_base = ramp(m2.outputs['Fac'], [(0.3, (0.80, 0.80, 0.80)), (0.7, (0.92, 0.92, 0.92))], (400, -100))
    r_grain = ramp(gr.outputs['Value'], [(0.0, (-0.05, -0.05, -0.05)), (1.0, (0.05, 0.05, 0.05))], (400, -300))
    rr = mth('ADD', r_base.outputs['Color'], r_grain.outputs['Color'], (650, -150))
    rr = mth('SUBTRACT', rr.outputs[0], mth('MULTIPLY', wear.outputs[0], 0.18, (650, -350)).outputs[0], (850, -150))
    rr = mth('ADD', rr.outputs[0], 0.0, (1050, -150), clamp=True)
    L.new(rr.outputs[0], bsdf.inputs['Roughness'])
    # ---------- MICRO-RELIEF (bump stack, heights in mm) ----------
    undul = noise(0.30, 4, 0.55, (-1700, -700))
    pits = voronoi(1.3, 'SMOOTH_F1', (-1700, -950), smooth=0.35)
    pit_h = ramp(pits.outputs['Distance'], [(0.0, (0, 0, 0)), (0.32, (1, 1, 1))], (-1400, -950))
    grain_h = noise(7.0, 4, 0.65, (-1700, -1200))
    crk = voronoi(0.45, 'DISTANCE_TO_EDGE', (-1700, -1450))
    crk_line = ramp(crk.outputs['Distance'], [(0.0, (0, 0, 0)), (0.025, (1, 1, 1))], (-1400, -1450))
    crk_mask_n = noise(0.30, 3, 0.5, (-1700, -1700))
    crk_mask = ramp(crk_mask_n.outputs['Fac'], [(0.52, (1, 1, 1)), (0.58, (0, 0, 0))], (-1400, -1700))
    crk_h = mix(crk_line.outputs['Color'], crk_mask.outputs['Color'], 1.0, 'LIGHTEN', (-1100, -1550))
    def bump(height, strength, distance, loc, prev=None, invert=False):
        b = node('ShaderNodeBump', loc); b.invert = invert
        b.inputs['Strength'].default_value = strength; b.inputs['Distance'].default_value = distance * unit
        L.new(height, b.inputs['Height'])
        if prev is not None: L.new(prev.outputs['Normal'], b.inputs['Normal'])
        return b
    b = bump(undul.outputs['Fac'], 0.35, 0.40, (900, -800))
    b = bump(pit_h.outputs['Color'], 0.55, 0.22, (1150, -800), b)
    b = bump(grain_h.outputs['Fac'], 0.60, 0.07, (1400, -800), b)
    b = bump(crk_h.outputs[2], 0.70, 0.22, (1650, -800), b)
    L.new(b.outputs['Normal'], bsdf.inputs['Normal'])
    return mat

def new_image(name, size, noncolor):
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    if noncolor: img.colorspace_settings.name = 'Non-Color'
    return img

def bake_part(name):
    t0 = time.time()
    clear_scene()
    high = import_part(name)
    low = make_low(high, TARGET_TRIS.get(name, DEFAULT_TRIS))
    mat = stone_material()
    high.data.materials.clear(); high.data.materials.append(mat)
    # low mesh gets a bake-target material with an active image node
    bm = bpy.data.materials.new('bake_target'); bm.use_nodes = True
    low.data.materials.clear(); low.data.materials.append(bm)
    size = TORSO_SIZE if name == 'torso' else SIZE
    tex = bm.node_tree.nodes.new('ShaderNodeTexImage'); bm.node_tree.nodes.active = tex
    scene.world = scene.world or bpy.data.worlds.new('World')
    scene.world.light_settings.distance = 1.0
    bpy.ops.object.select_all(action='DESELECT'); high.select_set(True); low.select_set(True); bpy.context.view_layer.objects.active = low
    bake = scene.render.bake
    bake.use_selected_to_active = True; bake.use_cage = False; bake.cage_extrusion = 0.6; bake.max_ray_distance = 1.6
    results = {}
    for key, btype, noncolor, extra in [
        ('normal', 'NORMAL', True, {}),
        ('albedo', 'DIFFUSE', False, {'pass_filter': {'COLOR'}}),
        ('roughness', 'ROUGHNESS', True, {}),
        ('ao', 'AO', True, {}),
    ]:
        img = new_image(f'{name}_{key}', size, noncolor); tex.image = img
        scene.cycles.samples = SAMPLES_AO if key == 'ao' else (8 if key != 'normal' else 4)
        tb = time.time()
        bpy.ops.object.bake(type=btype, use_selected_to_active=True, cage_extrusion=0.6, max_ray_distance=1.6,
                            margin=24, margin_type='EXTEND', normal_space='TANGENT', use_clear=True, **extra)
        path = os.path.join(OUT, f'{name}_{key}.png')
        img.filepath_raw = path; img.file_format = 'PNG'; img.save()
        results[key] = dict(file=os.path.basename(path), seconds=round(time.time() - tb, 1))
        print(f'  baked {name}:{key} {size}px in {time.time() - tb:.1f}s', flush=True)
    export_low(low, name)
    meta = dict(part=name, high_tris=len(high.data.polygons), low_tris=len(low.data.polygons), texture_size=size, maps=results, seconds=round(time.time() - t0, 1))
    with open(os.path.join(OUT, f'{name}.json'), 'w') as fh: json.dump(meta, fh, indent=1)
    if PREVIEW: preview(low, name)
    return meta

def export_low(low, name):
    me = low.data
    me.calc_loop_triangles()
    uv = me.uv_layers.active.name
    me.calc_tangents(uvmap=uv)
    nl = len(me.loops)
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    lv = np.empty(nl, np.int32); me.loops.foreach_get('vertex_index', lv)
    ln = np.empty(nl * 3, np.float32); me.loops.foreach_get('normal', ln); ln = ln.reshape(-1, 3)
    lt = np.empty(nl * 3, np.float32); me.loops.foreach_get('tangent', lt); lt = lt.reshape(-1, 3)
    ls = np.empty(nl, np.float32); me.loops.foreach_get('bitangent_sign', ls)
    luv = np.empty(nl * 2, np.float32); me.uv_layers[uv].data.foreach_get('uv', luv); luv = luv.reshape(-1, 2)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('loops', tri); tri = tri.reshape(-1, 3)
    attr = np.concatenate([co[lv], ln, luv, lt, ls[:, None]], axis=1)
    key = np.round(attr, 5)
    uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = attr[first]
    idx = inverse.reshape(-1)[tri].astype(np.uint32)
    blob = bytearray(); layout = {}
    for fname, arr in [('position', verts[:, 0:3]), ('normal', verts[:, 3:6]), ('uv', verts[:, 6:8]), ('tangent', verts[:, 8:12]), ('index', idx.reshape(-1))]:
        while len(blob) % 4: blob.append(0)
        a = np.ascontiguousarray(arr, dtype=np.uint32 if fname == 'index' else np.float32)
        layout[fname] = dict(offset=len(blob), count=int(a.size), itemSize=int(a.shape[1]) if a.ndim == 2 else 1, type=str(a.dtype))
        blob.extend(a.tobytes())
    with open(os.path.join(OUT, f'{name}.bin'), 'wb') as fh: fh.write(blob)
    with open(os.path.join(OUT, f'{name}.layout.json'), 'w') as fh: json.dump(dict(vertices=int(len(verts)), triangles=int(len(tri)), units='print_mm', layout=layout), fh, indent=1)

def preview(low, name):
    """Cycles render of the baked low mesh under a studio HDR-like setup to judge realism."""
    mat = bpy.data.materials.new('baked_preview'); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']
    def img_node(key, noncolor):
        n = nt.nodes.new('ShaderNodeTexImage'); n.image = bpy.data.images.load(os.path.join(OUT, f'{name}_{key}.png'))
        if noncolor: n.image.colorspace_settings.name = 'Non-Color'
        return n
    alb, rou, nor, ao = img_node('albedo', False), img_node('roughness', True), img_node('normal', True), img_node('ao', True)
    mixao = nt.nodes.new('ShaderNodeMix'); mixao.data_type = 'RGBA'; mixao.blend_type = 'MULTIPLY'; mixao.inputs['Factor'].default_value = 0.6
    nt.links.new(alb.outputs['Color'], mixao.inputs[6]); nt.links.new(ao.outputs['Color'], mixao.inputs[7])
    nt.links.new(mixao.outputs[2], bsdf.inputs['Base Color'])
    nt.links.new(rou.outputs['Color'], bsdf.inputs['Roughness'])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nt.links.new(nor.outputs['Color'], nm.inputs['Color']); nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    low.data.materials.clear(); low.data.materials.append(mat)
    for o in list(scene.objects):
        if o is not low: o.hide_render = True
    import mathutils
    bb = [low.matrix_world @ mathutils.Vector(c) for c in low.bound_box]
    ctr = sum(bb, mathutils.Vector()) / 8; size = max((max(v[i] for v in bb) - min(v[i] for v in bb)) for i in range(3))
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); scene.collection.objects.link(cam); scene.camera = cam
    cam.data.lens = 85; d = mathutils.Vector((0.9, -1.4, 0.8)).normalized()
    cam.location = ctr + d * size * 3.2; cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    def sun(name, direction, strength, angle_deg, color=(1, 1, 1)):
        L = bpy.data.lights.new(name, 'SUN'); L.energy = strength; L.angle = math.radians(angle_deg); L.color = color
        o = bpy.data.objects.new(name, L); scene.collection.objects.link(o)
        o.rotation_euler = mathutils.Vector(direction).normalized().to_track_quat('Z', 'Y').to_euler()
    sun('key', (-1.2, -1.0, 1.6), 4.5, 6, (1.0, 0.95, 0.88))
    sun('rim', (1.4, 1.2, 0.9), 2.2, 12, (0.85, 0.92, 1.0))
    sun('fill', (1.0, -1.5, 0.2), 0.6, 30, (0.9, 0.95, 1.0))
    w = scene.world; w.use_nodes = True; w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.05, 0.055, 0.06, 1); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    scene.cycles.samples = 64; scene.render.resolution_x = 900; scene.render.resolution_y = 900
    scene.view_settings.view_transform = 'AgX'; scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.render.filepath = os.path.join(OUT, f'{name}_preview.png'); bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    report = [bake_part(p) for p in PARTS]
    print(json.dumps(report, indent=1))

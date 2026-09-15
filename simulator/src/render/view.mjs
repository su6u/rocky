// Real-time view of the rebuilt Rocky. The official kit parts are rigid bodies placed every frame from the physics
// engine's body transforms; surfaces use the Blender-baked stone maps; lighting mirrors the Cycles look-dev setup
// (warm key, cool fill, rim) with an image-based studio environment, AgX tone mapping and soft shadows.
//
// Frames: the simulator is Z-up (x forward, y left). Everything physical lives under `simRoot`, which rotates the
// sim frame into three.js's Y-up frame, so part matrices and overlays are written in simulator coordinates.
import * as THREE from 'three';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { PART_NAMES, loadPartGeometry, loadTexture, loadStlGeometry, pool } from './assets.mjs';

const JOINT_COLORS = { yaw: new THREE.Color('#7fb4ff'), pitch: new THREE.Color('#8fe0b4'), elbow: new THREE.Color('#f0b27a') };
const KEY_DIR = new THREE.Vector3(-1.6, 2.0, 1.3).normalize();   // three.js frame; = sim (−1.6, −1.3, 2.0) as in tools/blender/render_poses.py
const RIM_DIR = new THREE.Vector3(0.6, 1.3, -1.8).normalize();
const FILL_DIR = new THREE.Vector3(1.8, 0.6, 1.2).normalize();
const MAX_POLY = 12;

export class RockyView {
  constructor(container) {
    this.container = container;
    this.parts = new Map();
    this.mode = 'surface';
    this.follow = true;
    this.info = null; this.frame = null; this.statue = null;
    this.span = 0.182;                      // carapace across (m); every framing distance scales with it
    this.orbit = { theta: 0.64, phi: 1.17, radius: 1.0, offset: new THREE.Vector3() };
    this.cameraMode = 'orbit';                  // orbit (free, follows the body) | chase (behind the heading) | top
    this.chase = { theta: null, dragTheta: 0, dragPhi: 0 };
    this.perf = { frameMs: 16.7, fps: 60, lastAt: 0, checkAt: 0, fastChecks: 0, adaptive: true };
    this.anchor = null; this.smoothTarget = null;

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.maxPixelRatio = Math.min(window.devicePixelRatio || 1, 1.75);
    this.pixelRatio = this.maxPixelRatio;
    r.setPixelRatio(this.pixelRatio);
    r.toneMapping = THREE.AgXToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    container.append(r.domElement);
    this.anisotropy = Math.min(8, r.capabilities.getMaxAnisotropy());
    // GPU-compressed textures: Basis UASTC in KTX2, transcoded off-thread to the best format this GPU supports.
    this.compression = true;
    this.ktx2 = new KTX2Loader().setTranscoderPath('assets/vendor/three/examples/jsm/libs/basis/').setWorkerLimit(2).detectSupport(r);
    const w = this.ktx2.workerConfig;
    this.gpuFormat = w.astcSupported ? 'ASTC 4×4' : w.bptcSupported ? 'BC7' : w.etc2Supported ? 'ETC2' : w.dxtSupported ? 'BC1/BC3' : null;

    this.scene = new THREE.Scene();
    this.background = new THREE.Color('#1a1818');
    this.scene.background = this.background;
    this.scene.fog = new THREE.Fog(this.background, 2.2, 9);
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.005, 80);

    this.simRoot = new THREE.Group();
    this.simRoot.rotation.x = -Math.PI / 2;
    this.scene.add(this.simRoot);

    this.buildEnvironment();
    this.buildLights();
    this.buildFloor();
    this.buildContactShadows();
    this.buildOverlays();
    this.terrainGroup = new THREE.Group();
    this.simRoot.add(this.terrainGroup);
    this.materials = this.buildSharedMaterials();
    this.bindInput();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  // ---------------------------------------------------------------- scene construction
  buildEnvironment() {
    // Studio "light box": dark room, three soft boxes placed along the key/fill/rim directions, grey floor bounce.
    const env = new THREE.Scene();
    const room = new THREE.Mesh(new THREE.BoxGeometry(24, 14, 24), new THREE.MeshBasicMaterial({ color: 0x121010, side: THREE.BackSide }));
    room.position.y = 6; env.add(room);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshBasicMaterial({ color: 0x2e2a2a }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.02; env.add(floor);
    const softbox = (dir, dist, w, h, color, intensity) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(intensity), side: THREE.DoubleSide }));
      m.position.copy(dir).multiplyScalar(dist); m.lookAt(0, 0, 0); env.add(m);
    };
    softbox(KEY_DIR, 7, 4.2, 3.2, 0xfff0dc, 7.5);
    softbox(FILL_DIR, 7, 5.5, 2.6, 0xdce8ff, 1.6);
    softbox(RIM_DIR, 7, 3.2, 4.2, 0xe6f0ff, 4.2);
    softbox(new THREE.Vector3(0, 1, 0), 6.5, 10, 1.2, 0xffffff, 0.9);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envRT = pmrem.fromScene(env, 0.025);
    this.scene.environment = this.envRT.texture;
    this.scene.environmentIntensity = 0.55;
    pmrem.dispose();
    env.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
  }

  buildLights() {
    const key = this.key = new THREE.DirectionalLight(0xffeedd, 2.9);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.0015;
    this.scene.add(key, key.target);
    this.rim = new THREE.DirectionalLight(0xe6f0ff, 1.35);
    this.fill = new THREE.DirectionalLight(0xdce8ff, 0.35);
    this.scene.add(this.rim, this.rim.target, this.fill, this.fill.target);
  }

  buildFloor() {
    this.floorPivot = new THREE.Group();
    this.simRoot.add(this.floorPivot);
    const tex = new THREE.CanvasTexture(floorCanvas());
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.anisotropy = this.anisotropy;
    this.floorTex = tex;
    const floor = this.floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.84, metalness: 0, envMapIntensity: 0.6 }));
    floor.receiveShadow = true;
    this.floorPivot.add(floor);
    const grid = this.grid = new THREE.GridHelper(6, 120, 0x8f8985, 0x8f8985);
    grid.rotation.x = Math.PI / 2;               // GridHelper lies in XZ; the sim ground is XY
    grid.position.z = 0.0004;
    grid.material.transparent = true; grid.material.opacity = 0.085; grid.material.depthWrite = false;
    this.floorPivot.add(grid);
  }

  buildContactShadows() {
    // Soft occlusion decals: the carapace and each foot darken the floor more the closer they are to it.
    const tex = new THREE.CanvasTexture(blobCanvas());
    const mat = (opacity) => new THREE.MeshBasicMaterial({ map: tex, color: 0x000000, transparent: true, opacity, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    const plane = new THREE.PlaneGeometry(1, 1);
    this.bodyBlob = new THREE.Mesh(plane, mat(0.5));
    this.footBlobs = Array.from({ length: 5 }, () => new THREE.Mesh(plane, mat(0.6)));
    for (const m of [this.bodyBlob, ...this.footBlobs]) { m.renderOrder = 1; this.floorPivot.add(m); }
  }

  buildOverlays() {
    const o = this.overlay = new THREE.Group();
    o.visible = false;
    this.simRoot.add(o);
    const lines = (count, material) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
      const l = new THREE.LineSegments(g, material); l.frustumCulled = false; l.renderOrder = 10; o.add(l); return l;
    };
    this.axisLines = lines(64, new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true }));
    this.axisLines.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
    this.skeleton = lines(40, new THREE.LineBasicMaterial({ color: 0xcff5e3, depthTest: false, transparent: true, opacity: 0.9 }));
    this.pucks = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 20), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45, metalness: 0.2 }), 32);
    this.pucks.frustumCulled = false; o.add(this.pucks);
    this.shellSpheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 14, 10), new THREE.MeshBasicMaterial({ color: 0xc4d6cd, wireframe: true, transparent: true, opacity: 0.16, depthWrite: false }), 96);
    this.shellSpheres.frustumCulled = false; o.add(this.shellSpheres);
    this.footSpheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthTest: false }), 5);
    this.footSpheres.frustumCulled = false; this.footSpheres.renderOrder = 11; o.add(this.footSpheres);
    this.arrows = Array.from({ length: 12 }, () => { const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.1, 0x9ff0c8); a.visible = false; a.renderOrder = 12; a.traverse((c) => { if (c.material) { c.material.depthTest = false; c.material.transparent = true; } }); o.add(a); return a; });
    this.comMarker = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({ color: 0xf4f1e0, depthTest: false, transparent: true }));
    this.comMarker.renderOrder = 12; o.add(this.comMarker);
    this.comDrop = lines(2, new THREE.LineBasicMaterial({ color: 0xf4f1e0, depthTest: false, transparent: true, opacity: 0.6 }));
    const fillGeom = new THREE.BufferGeometry();
    fillGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array((MAX_POLY + 1) * 3), 3));
    const fillIndex = []; for (let i = 1; i < MAX_POLY; i++) fillIndex.push(0, i, i + 1);
    fillGeom.setIndex(fillIndex);
    this.polyFill = new THREE.Mesh(fillGeom, new THREE.MeshBasicMaterial({ color: 0x86d8b5, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
    this.polyFill.frustumCulled = false; this.polyFill.renderOrder = 2; o.add(this.polyFill);
    const loopGeom = new THREE.BufferGeometry();
    loopGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_POLY * 3), 3));
    this.polyLine = new THREE.LineLoop(loopGeom, new THREE.LineBasicMaterial({ color: 0x9fe6c6, transparent: true, opacity: 0.85, depthTest: false }));
    this.polyLine.frustumCulled = false; this.polyLine.renderOrder = 3; o.add(this.polyLine);
    this.zmpMarker = new THREE.Mesh(new THREE.OctahedronGeometry(1), new THREE.MeshBasicMaterial({ color: 0xffd28a, depthTest: false }));
    this.captureMarker = new THREE.Mesh(new THREE.RingGeometry(0.75, 1, 28), new THREE.MeshBasicMaterial({ color: 0xff9f8a, side: THREE.DoubleSide, depthTest: false }));
    for (const m of [this.zmpMarker, this.captureMarker]) { m.renderOrder = 12; o.add(m); }
  }

  buildSharedMaterials() {
    return {
      wire: new THREE.MeshBasicMaterial({ color: 0x8fd9b8, wireframe: true, transparent: true, opacity: 0.2, depthWrite: false }),
    };
  }

  // ---------------------------------------------------------------- parts
  async loadParts({ manifest, base, tier, compression = true, onProgress }) {
    this.compression = compression && !!this.gpuFormat;
    const total = PART_NAMES.length;
    let done = 0;
    await pool(PART_NAMES, 3, async (name) => {
      const record = manifest.parts?.[name];
      let entry;
      if (record) {
        const [geometry, maps] = await Promise.all([loadPartGeometry(base, record), this.loadMaps(base, record, tier)]);
        entry = { name, baked: true, record, geometry, maps };
      } else {
        const geometry = await loadStlGeometry(`kit/stl/${name}.stl`);
        entry = { name, baked: false, geometry, maps: null };
      }
      this.installPart(entry);
      onProgress?.(++done, total, name, entry.baked);
    });
    this.tier = tier;
    this.applyMode();
  }

  async loadMaps(base, record, tier) {
    const set = record.textures[tier] || record.textures.balanced;
    const formats = {};
    const load = async (key, srgb) => {
      if (this.compression && set.ktx2) {
        try {
          const tex = await this.ktx2.loadAsync(base + set.ktx2[key]);
          tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
          tex.anisotropy = this.anisotropy;
          formats[key] = 'ktx2';
          return tex;
        } catch (err) {
          console.warn(`KTX2 failed for ${set.ktx2[key]}, using WebP`, err);
        }
      }
      formats[key] = 'webp';
      return loadTexture(base + set[key], { srgb, anisotropy: this.anisotropy });
    };
    const [albedo, normal, orm] = await Promise.all([load('albedo', true), load('normal', false), load('orm', false)]);
    return { albedo, normal, orm, size: set.size, formats };
  }

  installPart(entry) {
    const surface = entry.baked
      ? new THREE.MeshStandardMaterial({ map: entry.maps.albedo, normalMap: entry.maps.normal, roughnessMap: entry.maps.orm, aoMap: entry.maps.orm, aoMapIntensity: 1, roughness: 1, metalness: 0 })
      : fallbackStoneMaterial();
    const xray = surface.clone();
    Object.assign(xray, { transparent: true, opacity: 0.32, depthWrite: false });
    const mesh = new THREE.Mesh(entry.geometry, surface);
    mesh.matrixAutoUpdate = false;
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.visible = false;
    mesh.name = entry.name;
    this.simRoot.add(mesh);
    const old = this.parts.get(entry.name);
    if (old) this.disposePart(old);
    this.parts.set(entry.name, { ...entry, mesh, surface, xray });
  }

  disposePart(p) {
    this.simRoot.remove(p.mesh);
    p.geometry.dispose(); p.surface.dispose(); p.xray.dispose();
    if (p.maps) for (const k of ['albedo', 'normal', 'orm']) p.maps[k].dispose();
  }

  async setTextures({ tier, compression }, manifest, base, onProgress) {
    compression = compression && !!this.gpuFormat;
    if (tier === this.tier && compression === this.compression) return;
    this.compression = compression;
    const baked = [...this.parts.values()].filter((p) => p.baked);
    let done = 0;
    await pool(baked, 2, async (p) => {
      const maps = await this.loadMaps(base, manifest.parts[p.name], tier);
      const old = p.maps;
      p.maps = maps;
      for (const mat of [p.surface, p.xray]) { mat.map = maps.albedo; mat.normalMap = maps.normal; mat.roughnessMap = maps.orm; mat.aoMap = maps.orm; mat.needsUpdate = true; }
      for (const k of ['albedo', 'normal', 'orm']) old[k].dispose();
      onProgress?.(++done, baked.length, p.name);
    });
    this.tier = tier;
  }

  partStats() {
    const list = [...this.parts.values()];
    // GPU texture memory: compressed 4×4 block formats cost 1 byte per texel (ETC2 RGB 0.5), RGBA8 costs 4; mips add 1/3.
    const perTexel = { ktx2: this.gpuFormat === 'ETC2' ? 0.5 : 1, webp: 4 };
    let bytes = 0, compressed = 0, maps = 0;
    for (const p of list) {
      if (!p.maps) continue;
      for (const key of ['albedo', 'normal', 'orm']) {
        const f = p.maps.formats?.[key] ?? 'webp';
        bytes += p.maps.size * p.maps.size * perTexel[f] * (4 / 3);
        maps++; if (f === 'ktx2') compressed++;
      }
    }
    return {
      baked: list.filter((p) => p.baked).map((p) => p.name),
      fallback: list.filter((p) => !p.baked).map((p) => p.name),
      triangles: list.reduce((s, p) => s + (p.geometry.index ? p.geometry.index.count / 3 : 0), 0),
      textureMB: bytes / 2 ** 20,
      compressedMaps: compressed, maps,
      gpuFormat: this.gpuFormat,
    };
  }

  // ---------------------------------------------------------------- state
  setInfo(info) {
    const rescale = !this.info || Math.abs(info.carapaceAcross - this.span) > 1e-6;
    this.info = info;
    this.span = info.carapaceAcross;
    this.partBodies = info.bodies.filter((b) => b.part).map((b) => ({ index: b.index, part: b.part, P: b.partToBody.R.flat(), t: b.partToBody.t }));
    const used = new Set(this.partBodies.map((b) => b.part));
    for (const [name, p] of this.parts) p.mesh.visible = !this.statue && used.has(name);
    this.floorPivot.rotation.y = -info.terrain.slope;
    const s = this.span / 0.182;
    this.floorTex.repeat.set(80 / (0.9 * s), 80 / (0.9 * s));
    this.grid.scale.setScalar(s);
    this.scene.fog.near = 2.4 * s; this.scene.fog.far = 10 * s;
    const sh = this.key.shadow.camera, half = 0.55 * s;
    Object.assign(sh, { left: -half, right: half, top: half, bottom: -half, near: 0.05 * s, far: 6 * s });
    sh.updateProjectionMatrix();
    this.key.shadow.normalBias = 0.0015 * s;
    this.camera.near = 0.02 * s; this.camera.far = 60 * s; this.camera.updateProjectionMatrix();
    if (rescale) this.resetCamera();
    this.layoutShellSpheres();
    this.setTerrain(info.terrain);
  }

  // Blocks and floor patches from the scenario (the solver collides with exactly these shapes).
  setTerrain(terrain) {
    for (const child of [...this.terrainGroup.children]) { this.terrainGroup.remove(child); child.traverse((o) => { o.geometry?.dispose(); if (o.material && !o.material.userData.shared) o.material.dispose(); }); }
    if (!terrain) return;
    this.terrainMaterials ??= {
      block: new THREE.MeshStandardMaterial({ color: 0x77726b, roughness: 0.93, metalness: 0, envMapIntensity: 0.7 }),
      edge: new THREE.LineBasicMaterial({ color: 0xb9c7c0, transparent: true, opacity: 0.35 }),
      tile: new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(tileCanvas()), roughness: 0.18, metalness: 0, envMapIntensity: 1.1, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
    };
    for (const m of Object.values(this.terrainMaterials)) m.userData.shared = true;
    this.terrainMaterials.tile.map.colorSpace = THREE.SRGBColorSpace;
    this.terrainMaterials.tile.map.wrapS = this.terrainMaterials.tile.map.wrapT = THREE.RepeatWrapping;
    for (const b of terrain.boxes ?? []) {
      const size = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
      const geo = new THREE.BoxGeometry(...size);
      const mesh = new THREE.Mesh(geo, this.terrainMaterials.block);
      mesh.position.set((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
      mesh.castShadow = true; mesh.receiveShadow = true; mesh.name = b.label || b.id;
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), this.terrainMaterials.edge));
      this.terrainGroup.add(mesh);
    }
    for (const p of terrain.patches ?? []) {
      const w = p.max[0] - p.min[0], h = p.max[1] - p.min[1];
      const geo = new THREE.PlaneGeometry(w, h);
      const uv = geo.attributes.uv; for (let k = 0; k < uv.count; k++) uv.setXY(k, uv.getX(k) * w / 0.06, uv.getY(k) * h / 0.06);
      const mesh = new THREE.Mesh(geo, this.terrainMaterials.tile);
      mesh.position.set((p.min[0] + p.max[0]) / 2, (p.min[1] + p.max[1]) / 2, 0.0003);
      mesh.receiveShadow = true; mesh.name = p.label || p.id;
      this.terrainGroup.add(mesh);
    }
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    this.chase = { theta: null, dragTheta: 0, dragPhi: 0 };
    if (mode !== 'orbit') this.orbit.offset.set(0, 0, 0);
  }

  setFrame(frame) { this.frame = frame; }

  setStatue(statue) {
    this.statue = statue;
    if (!statue) { if (this.info) this.setInfo(this.info); this.applyMode(); return; }
    // Registered sculpture pose (tools/geometry/registration.py): part → statue frame, rotated so the statue's up is +z.
    const up = new THREE.Vector3(...statue.up).normalize(), z = new THREE.Vector3(0, 0, 1);
    const axis = new THREE.Vector3().crossVectors(up, z), ang = Math.acos(THREE.MathUtils.clamp(up.dot(z), -1, 1));
    const U = axis.lengthSq() > 1e-12 ? rodrigues(axis.normalize(), ang) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const s = statue.scale, placed = [];
    for (const [name, p] of this.parts) {
      const pose = statue.pose[name];
      p.mesh.visible = !!pose && name !== '1-B';
      if (!p.mesh.visible) continue;
      const A = mul3(U, pose.R.flat()).map((x) => x * s), t = mulv(U, pose.t).map((x) => x * s);
      setMatrix(p.mesh.matrix, A, t);
      placed.push(p);
    }
    let lowest = Infinity;
    const v = new THREE.Vector3();
    for (const p of placed) {
      const pos = p.geometry.attributes.position;
      for (let i = 0; i < pos.count; i += 9) { v.fromBufferAttribute(pos, i).applyMatrix4(p.mesh.matrix); lowest = Math.min(lowest, v.z); }
    }
    for (const p of placed) p.mesh.matrix.elements[14] -= lowest;
    this.floorPivot.rotation.y = 0;
    this.applyMode();
  }

  setMode(mode) { this.mode = mode; this.applyMode(); }
  applyMode() {
    for (const p of this.parts.values()) p.mesh.material = this.mode === 'wire' ? this.materials.wire : this.mode === 'xray' ? p.xray : p.surface;
    this.overlay.visible = this.mode !== 'surface' && !this.statue;
  }
  setShadows(on) { this.key.castShadow = on; for (const m of [this.bodyBlob, ...this.footBlobs]) m.visible = on; }
  setGrid(on) { this.grid.visible = on; }
  setFollow(on) { this.follow = on; if (on) this.orbit.offset.set(0, 0, 0); }
  resetCamera() {
    this.chase = { theta: null, dragTheta: 0, dragPhi: 0 };
    Object.assign(this.orbit, { theta: 0.64, phi: 1.17, radius: 5.8 * this.span });
    this.orbit.offset.set(0, 0, 0);
    this.anchor = null; this.smoothTarget = null;
  }

  // ---------------------------------------------------------------- per-frame update
  render() {
    const f = this.frame, info = this.info;
    let focus;
    if (this.statue) {
      focus = new THREE.Vector3(0, 0.55 * this.span, 0);
    } else if (f && info) {
      this.placeParts(f);
      this.updateContactShadows(f);
      if (this.overlay.visible) this.updateOverlays(f);
      focus = simToThree(f.com);
      focus.y = Math.max(focus.y - 0.1 * this.span, 0.45 * this.span + groundZ(info, f.com[0]));
    } else {
      focus = new THREE.Vector3(0, 0.6 * this.span, 0);
    }
    // follow: aim at the body; free: keep the last aim point. Panning offsets either.
    if (this.follow || this.cameraMode !== 'orbit' || !this.anchor || this.statue) this.anchor = focus;
    const target = this.anchor.clone().add(this.orbit.offset);
    this.smoothTarget = this.smoothTarget ? this.smoothTarget.lerp(target, 0.12) : target.clone();
    const o = this.orbit, c = this.smoothTarget;
    let theta = o.theta, phi = o.phi;
    if (this.cameraMode !== 'orbit' && f && !this.statue) {
      // chase: behind the carapace heading (sim yaw ψ looks along three.js θ = ψ − π/2 from behind); top: from above
      const yaw = Math.atan2(f.xf[3], f.xf[0]), want = yaw - Math.PI / 2 + this.chase.dragTheta;
      if (this.chase.theta === null) this.chase.theta = want;
      const d = Math.atan2(Math.sin(want - this.chase.theta), Math.cos(want - this.chase.theta));
      this.chase.theta += d * (this.reducedMotion ? 1 : 0.06);
      theta = this.chase.theta;
      phi = THREE.MathUtils.clamp((this.cameraMode === 'top' ? 0.1 : 1.08) + this.chase.dragPhi, 0.06, Math.PI * 0.49);
    }
    this.camera.position.set(c.x + o.radius * Math.sin(phi) * Math.sin(theta), c.y + o.radius * Math.cos(phi), c.z + o.radius * Math.sin(phi) * Math.cos(theta));
    this.camera.lookAt(c);
    // lights ride with the subject so the shadow frustum stays tight
    const d = 3 * this.span / 0.182;
    this.key.target.position.copy(c); this.key.position.copy(c).addScaledVector(KEY_DIR, d);
    this.rim.target.position.copy(c); this.rim.position.copy(c).addScaledVector(RIM_DIR, d);
    this.fill.target.position.copy(c); this.fill.position.copy(c).addScaledVector(FILL_DIR, d);
    this.renderer.render(this.scene, this.camera);
    this.trackPerformance();
  }

  // Frame-time tracking and adaptive resolution: drop the pixel ratio by 0.25 while frames take > 24 ms, raise it again
  // after three consecutive seconds under 14 ms.
  trackPerformance() {
    const p = this.perf, now = performance.now(), dt = now - p.lastAt;
    p.lastAt = now;
    if (dt > 0 && dt < 250) p.frameMs += (dt - p.frameMs) * 0.05;
    if (now < p.checkAt) return;
    p.checkAt = now + 1000;
    p.fps = 1000 / p.frameMs;
    if (!p.adaptive) return;
    if (p.frameMs > 24 && this.pixelRatio > 1) { this.setPixelRatio(this.pixelRatio - 0.25); p.fastChecks = 0; }
    else if (p.frameMs < 14 && this.pixelRatio < this.maxPixelRatio) { if (++p.fastChecks >= 3) { this.setPixelRatio(this.pixelRatio + 0.25); p.fastChecks = 0; } }
    else p.fastChecks = 0;
  }
  setPixelRatio(value) {
    this.pixelRatio = Math.max(1, Math.min(this.maxPixelRatio, value));
    this.renderer.setPixelRatio(this.pixelRatio);
    this.resize();
  }

  placeParts(f) {
    const s = this.info.scale, xf = f.xf;
    for (const b of this.partBodies) {
      const p = this.parts.get(b.part);
      if (!p) continue;
      const o = b.index * 15, R = xf.subarray(o, o + 9);
      const A = mul3(R, b.P).map((x) => x * s);
      const Rt = mulv(R, b.t);
      setMatrix(p.mesh.matrix, A, [Rt[0] * s + xf[o + 9], Rt[1] * s + xf[o + 10], Rt[2] * s + xf[o + 11]]);
    }
  }

  updateContactShadows(f) {
    const info = this.info, th = info.terrain.slope, cs = Math.cos(th), sn = Math.sin(th);
    const toLocal = (p) => [p[0] * cs + p[2] * sn, p[1], -p[0] * sn + p[2] * cs];   // sim → tilted floor frame
    const b = toLocal(f.com), bodyH = Math.max(0, b[2]);
    const k = this.span;
    this.bodyBlob.position.set(b[0], b[1], 0.0006);
    this.bodyBlob.scale.setScalar(1.25 * k * (1 + bodyH / k * 0.6));
    this.bodyBlob.material.opacity = 0.55 * Math.exp(-bodyH / (1.2 * k));
    info.limbs.forEach((L, i) => {
      const p = toLocal(f.limbs[i].foot), h = Math.max(0, p[2] - L.footRadius);
      const m = this.footBlobs[i];
      m.position.set(p[0], p[1], 0.0007);
      m.scale.setScalar(L.footRadius * (4.5 + 40 * h / k));
      m.material.opacity = 0.62 * Math.exp(-h / (0.12 * k));
    });
  }

  layoutShellSpheres() {
    const info = this.info; if (!info) return;
    this.shellShapes = info.shapes.filter((c) => c.kind === 'shell');
    this.shellSpheres.count = Math.min(this.shellShapes.length, 96);
  }

  updateOverlays(f) {
    const info = this.info, xf = f.xf, k = this.span / 0.182;
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sc = new THREE.Vector3();
    const Y = new THREE.Vector3(0, 1, 0);
    // joint axes + servo pucks
    const pos = this.axisLines.geometry.attributes.position.array, col = this.axisLines.geometry.attributes.color.array;
    let n = 0;
    info.actuators.forEach((a, i) => {
      const o = a.body * 15, ax = [xf[o + 12], xf[o + 13], xf[o + 14]], p = [xf[o + 9], xf[o + 10], xf[o + 11]];
      const len = 0.045 * k, c = JOINT_COLORS[a.joint];
      pos.set([p[0] - ax[0] * len, p[1] - ax[1] * len, p[2] - ax[2] * len, p[0] + ax[0] * len, p[1] + ax[1] * len, p[2] + ax[2] * len], n * 3);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], n * 3); n += 2;
      const servo = f.servos[i];
      q.setFromUnitVectors(Y, v.set(ax[0], ax[1], ax[2]));
      m4.compose(v.set(p[0], p[1], p[2]), q, sc.set(0.0125, 0.026, 0.0125));
      this.pucks.setMatrixAt(i, m4);
      this.pucks.setColorAt(i, servo.failed ? tmpColor('#ff6b5e') : servo.protected ? tmpColor('#ffb347') : servo.hot ? tmpColor('#ff8a65') : c);
    });
    this.pucks.count = info.actuators.length;
    this.pucks.instanceMatrix.needsUpdate = true; if (this.pucks.instanceColor) this.pucks.instanceColor.needsUpdate = true;
    this.axisLines.geometry.setDrawRange(0, n); this.axisLines.geometry.attributes.position.needsUpdate = true; this.axisLines.geometry.attributes.color.needsUpdate = true;
    // skeleton: shoulder → elbow → foot centre
    const sk = this.skeleton.geometry.attributes.position.array;
    f.limbs.forEach((L, i) => { sk.set([...L.shoulder, ...L.elbow, ...L.elbow, ...L.foot], i * 12); });
    this.skeleton.geometry.setDrawRange(0, f.limbs.length * 4); this.skeleton.geometry.attributes.position.needsUpdate = true;
    // collision spheres (body-attached)
    this.shellShapes.forEach((c, i) => {
      if (i >= 96) return;
      const o = c.body * 15, R = xf.subarray(o, o + 9), w = mulv(R, c.local);
      m4.makeScale(c.radius, c.radius, c.radius).setPosition(w[0] + xf[o + 9], w[1] + xf[o + 10], w[2] + xf[o + 11]);
      this.shellSpheres.setMatrixAt(i, m4);
    });
    this.shellSpheres.instanceMatrix.needsUpdate = true;
    // feet: colour by measured load; arrows show the contact force
    const perFoot = info.weight / 5, loads = f.contacts.filter((c) => c.kind === 'foot');
    info.limbs.forEach((L, i) => {
      const p = f.limbs[i].foot, load = loads.filter((c) => c.name === `foot${L.limb}`).reduce((s, c) => s + c.fn, 0);
      m4.makeScale(L.footRadius, L.footRadius, L.footRadius).setPosition(p[0], p[1], p[2]);
      this.footSpheres.setMatrixAt(i, m4);
      this.footSpheres.setColorAt(i, load > 0.05 * perFoot ? tmpColor('#8ff0c2') : f.limbs[i].planned ? tmpColor('#c9d3cf') : tmpColor('#f2b27b'));
    });
    this.footSpheres.instanceMatrix.needsUpdate = true; if (this.footSpheres.instanceColor) this.footSpheres.instanceColor.needsUpdate = true;
    const arrowScale = (0.5 * this.span) / perFoot;
    let ai = 0;
    for (const c of f.contacts) {
      if (ai >= this.arrows.length) break;
      const mag = Math.hypot(...c.f);
      if (mag < 0.02 * perFoot) continue;
      const a = this.arrows[ai++];
      a.visible = true;
      a.position.set(c.p[0], c.p[1], c.p[2]);
      a.setDirection(v.set(c.f[0] / mag, c.f[1] / mag, c.f[2] / mag));
      const len = Math.min(mag * arrowScale, 2.5 * this.span);
      a.setLength(len, Math.min(0.3 * len, 0.03 * k), Math.min(0.15 * len, 0.015 * k));
      a.setColor(c.kind === 'foot' ? (c.use > 0.95 ? 0xffb347 : 0x9ff0c8) : 0xff6b5e);
    }
    for (; ai < this.arrows.length; ai++) this.arrows[ai].visible = false;
    // COM, its projection, support polygon, ZMP, capture point
    const com = f.com, gz = (x) => groundZ(info, x) + 0.0012 * k;
    this.comMarker.position.set(com[0], com[1], com[2]); this.comMarker.scale.setScalar(0.0075 * k);
    const proj = f.stab.comProjection || [com[0], com[1], gz(com[0])];
    this.comDrop.geometry.attributes.position.array.set([com[0], com[1], com[2], proj[0], proj[1], gz(proj[0])]);
    this.comDrop.geometry.attributes.position.needsUpdate = true;
    const poly = f.stab.polygon || [];
    const fillPos = this.polyFill.geometry.attributes.position.array, loopPos = this.polyLine.geometry.attributes.position.array;
    const count = Math.min(poly.length, MAX_POLY);
    for (let i = 0; i < count; i++) {
      const [x, y] = poly[i];
      fillPos.set([x, y, gz(x)], i * 3);
      loopPos.set([x, y, gz(x) + 0.0002], i * 3);
    }
    this.polyFill.geometry.setDrawRange(0, count >= 3 ? (count - 2) * 3 : 0);
    this.polyFill.geometry.attributes.position.needsUpdate = true;
    this.polyLine.geometry.setDrawRange(0, count >= 3 ? count : 0);
    this.polyLine.geometry.attributes.position.needsUpdate = true;
    const zmp = f.stab.zmp;
    this.zmpMarker.visible = !!zmp;
    if (zmp) { this.zmpMarker.position.set(zmp[0], zmp[1], gz(zmp[0]) + 0.004 * k); this.zmpMarker.scale.setScalar(0.006 * k); }
    const cp = f.stab.capture;
    this.captureMarker.visible = !!cp;
    if (cp) { this.captureMarker.position.set(cp[0], cp[1], gz(cp[0]) + 0.001 * k); this.captureMarker.scale.setScalar(0.009 * k); this.captureMarker.rotation.y = -info.terrain.slope; }
  }

  // ---------------------------------------------------------------- camera input
  bindInput() {
    const el = this.renderer.domElement, pointers = new Map();
    el.style.touchAction = 'none';
    let pinch = null;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    });
    el.addEventListener('pointermove', (e) => {
      const p = pointers.get(e.pointerId); if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) this.zoom(pinch / d); pinch = d; return;
      }
      if (p.pan) this.pan(dx, dy);
      else if (this.cameraMode !== 'orbit') { this.chase.dragTheta -= dx * 0.0065; this.chase.dragPhi -= dy * 0.005; }
      else { this.orbit.theta -= dx * 0.0065; this.orbit.phi = THREE.MathUtils.clamp(this.orbit.phi - dy * 0.005, 0.08, Math.PI * 0.49); }
    });
    const up = (e) => { pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null; };
    el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.0011)); }, { passive: false });
    el.addEventListener('dblclick', () => this.resetCamera());
  }
  zoom(factor) { this.orbit.radius = THREE.MathUtils.clamp(this.orbit.radius * factor, 1.2 * this.span, 25 * this.span); }
  pan(dx, dy) {
    const o = this.orbit, scale = o.radius * 0.0016;
    const right = new THREE.Vector3(Math.cos(o.theta), 0, -Math.sin(o.theta));
    const upv = new THREE.Vector3(0, 1, 0);
    o.offset.addScaledVector(right, -dx * scale).addScaledVector(upv, dy * scale);
  }
  resize() {
    const r = this.container.getBoundingClientRect();
    if (!r.width || !r.height) return;
    this.renderer.setSize(r.width, r.height);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
  }
}

// ---------------------------------------------------------------- helpers
function simToThree(p) { return new THREE.Vector3(p[0], p[2], -p[1]); }
function groundZ(info, x) { return Math.tan(info.terrain.slope) * x; }
function mul3(A, B) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return o;
}
function mulv(A, v) { return [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]]; }
function setMatrix(m, A, t) { m.set(A[0], A[1], A[2], t[0], A[3], A[4], A[5], t[1], A[6], A[7], A[8], t[2], 0, 0, 0, 1); }
function rodrigues(a, ang) {
  const [x, y, z] = [a.x, a.y, a.z], c = Math.cos(ang), s = Math.sin(ang), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}
const colorCache = new Map();
function tmpColor(hex) { if (!colorCache.has(hex)) colorCache.set(hex, new THREE.Color(hex)); return colorCache.get(hex); }

// Procedural concrete-like studio floor (tileable value-noise fBm + speckle), sRGB.
function floorCanvas(size = 512) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const ctx = c.getContext('2d'), img = ctx.createImageData(size, size);
  const rand = mulberry32(7);
  const lattice = (n) => { const g = new Float32Array(n * n); for (let i = 0; i < g.length; i++) g[i] = rand(); return g; };
  const octaves = [[8, 0.5], [16, 0.25], [32, 0.14], [64, 0.07], [128, 0.04]].map(([n, a]) => ({ n, a, g: lattice(n) }));
  const sample = (o, x, y) => {
    const fx = (x / size) * o.n, fy = (y / size) * o.n, ix = Math.floor(fx), iy = Math.floor(fy), tx = fx - ix, ty = fy - iy;
    const at = (i, j) => o.g[((j % o.n) + o.n) % o.n * o.n + ((i % o.n) + o.n) % o.n];
    const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    return (at(ix, iy) * (1 - sx) + at(ix + 1, iy) * sx) * (1 - sy) + (at(ix, iy + 1) * (1 - sx) + at(ix + 1, iy + 1) * sx) * sy;
  };
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let v = 0; for (const o of octaves) v += o.a * sample(o, x, y);
    v = v / 1.0 + (rand() - 0.5) * 0.05;
    const base = 44 + 22 * v, i = (y * size + x) * 4;
    img.data[i] = base * 0.98; img.data[i + 1] = base; img.data[i + 2] = base * 1.02; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
// Polished floor tiles: pale stone squares with thin grout lines (one tile per texture repeat).
function tileCanvas(size = 128) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#b9bcbc'; ctx.fillRect(0, 0, size, size);
  const g = ctx.createLinearGradient(0, 0, size, size); g.addColorStop(0, 'rgba(255,255,255,0.10)'); g.addColorStop(1, 'rgba(0,0,0,0.08)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#6f7474'; ctx.fillRect(0, 0, size, 3); ctx.fillRect(0, 0, 3, size);
  return c;
}
function blobCanvas(size = 128) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const ctx = c.getContext('2d'), g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (let i = 0; i <= 10; i++) { const u = i / 10, a = Math.pow(1 - u, 2.2) * (1 - u * u); g.addColorStop(u, `rgba(255,255,255,${a.toFixed(4)})`); }
  ctx.fillStyle = g; ctx.fillRect(0, 0, size, size);
  return c;
}
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// Unbaked parts: object-space procedural stone (print-mm frequencies matching the bake shader's macro tone and grain).
function fallbackStoneMaterial() {
  const mat = new THREE.MeshStandardMaterial({ color: 0x4d463f, roughness: 0.88, metalness: 0 });
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vStone;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvStone = position;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
varying vec3 vStone;
float h3(vec3 p){ return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float n3(vec3 p){ vec3 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
  return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x), mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x), mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z); }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
float broad = .55 * n3(vStone * .035) + .3 * n3(vStone * .14) + .15 * n3(vStone * .6);
float grain = n3(vStone * 4.0);
diffuseColor.rgb *= mix(.62, 1.28, broad) * (.86 + .26 * grain);`);
  };
  mat.customProgramCacheKey = () => 'rocky-fallback-stone';
  return mat;
}

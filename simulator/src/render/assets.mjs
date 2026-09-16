// Asset loading for the viewer: baked part meshes (tools/blender/bake_parts.py), texture sets
// (tools/textures/encode_web_textures.py) and a full-resolution STL fallback for parts that have not been baked yet.
import * as THREE from 'three';

export const PART_NAMES = ['torso', '1-A', '1-B', '1-C', '2-A', '2-B', '3-A', '3-B', '4-A', '4-B', '5-A', '5-B'];

async function fetchOk(url, as) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return as === 'json' ? res.json() : as === 'blob' ? res.blob() : res.arrayBuffer();
}

export async function loadManifest(url) {
  try { return await fetchOk(url, 'json'); } catch { return { parts: {} }; }
}

// Low-poly baked mesh: positions/normals/uv/tangents (MikkTSpace, from Blender) + uint32 indices, print mm.
export async function loadPartGeometry(base, record) {
  const [layout, buffer] = await Promise.all([fetchOk(base + record.layout, 'json'), fetchOk(base + record.mesh)]);
  const L = layout.layout, g = new THREE.BufferGeometry();
  const view = (name, Type) => new Type(buffer, L[name].offset, L[name].count);
  g.setAttribute('position', new THREE.BufferAttribute(view('position', Float32Array), 3));
  g.setAttribute('normal', new THREE.BufferAttribute(view('normal', Float32Array), 3));
  // Blender UVs put v = 0 at the bottom of the image. Textures are uploaded unflipped (ImageBitmap), so flip v
  // here; tangents stay as baked, which keeps the tangent-space normal map exact.
  const uv = view('uv', Float32Array).slice();
  for (let i = 1; i < uv.length; i += 2) uv[i] = 1 - uv[i];
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('tangent', new THREE.BufferAttribute(view('tangent', Float32Array), 4));
  g.setIndex(new THREE.BufferAttribute(view('index', Uint32Array), 1));
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

// Decode off the main thread, without colour management (normal/ORM data must reach the GPU untouched).
export async function loadTexture(url, { srgb, anisotropy = 8 }) {
  const blob = await fetchOk(url, 'blob');
  let image;
  try {
    image = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  } catch {
    image = await createImageBitmap(blob);
  }
  const tex = new THREE.Texture(image);
  tex.flipY = false;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.anisotropy = anisotropy;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.needsUpdate = true;
  return tex;
}

// Binary STL → welded, indexed geometry with smooth normals (print mm). Used until a part is baked.
export async function loadStlGeometry(url) {
  const buf = await fetchOk(url);
  const dv = new DataView(buf), n = dv.getUint32(80, true);
  const map = new Map(), pos = [], index = new Uint32Array(n * 3);
  let off = 84;
  for (let t = 0; t < n; t++, off += 50) {
    for (let v = 0; v < 3; v++) {
      const o = off + 12 + v * 12;
      const x = dv.getFloat32(o, true), y = dv.getFloat32(o + 4, true), z = dv.getFloat32(o + 8, true);
      const key = `${Math.round(x * 1e4)},${Math.round(y * 1e4)},${Math.round(z * 1e4)}`;
      let id = map.get(key);
      if (id === undefined) { id = pos.length / 3; map.set(key, id); pos.push(x, y, z); }
      index[t * 3 + v] = id;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  g.computeBoundingBox();
  return g;
}

// Run async jobs with bounded concurrency (keeps peak decode memory low on 8 GB machines).
export async function pool(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const i = next++; results[i] = await fn(items[i], i); }
  });
  await Promise.all(workers);
  return results;
}

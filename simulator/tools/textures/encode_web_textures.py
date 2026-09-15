"""Publish Blender bake outputs (bakes/) as runtime assets for the browser viewer.

  python3 tools/textures/encode_web_textures.py            # every completely baked part, skipping up-to-date outputs
  python3 tools/textures/encode_web_textures.py --force    # rebuild everything
  python3 tools/textures/encode_web_textures.py --no-ktx2  # WebP only (no basisu needed)

Copies each baked low-poly mesh (bakes/<part>.bin + .layout.json) to assets/meshes/ and writes
assets/textures/<tier>/<part>_{albedo,normal,orm}.{ktx2,webp} for two tiers:
  balanced  limbs 1024², torso 2048². Limb shells are 1000–1500 print-mm² and the torso 6380 mm², so this gives
            every part ≈20 texels per print-mm (UV coverage ≈45–51 %).
  high      bake resolution (limbs 2048², torso 4096²), ≈40 texels per print-mm.
Maps
  albedo    sRGB. Downsampled in linear light.
  normal    tangent space, OpenGL convention (+Y), as baked by Blender. Downsampled as vectors and renormalised.
  orm       R = ambient occlusion, G = roughness, B = 0: the channels three.js reads for aoMap and roughnessMap.
Formats
  .ktx2     GPU-compressed (default in the viewer). Basis Universal UASTC LDR 4×4 + Zstandard with a full mip chain,
            transcoded in the browser to ASTC 4×4 / BC7 / ETC2: 1 byte per texel in GPU memory instead of 4.
            Needs `basisu` (brew install basis_universal). Normal maps: -normal_map + renormalised mips, no RDO.
  .webp     fallback for browsers without a compressed format: albedo lossy q92; normal/orm lossless (lossy WebP
            subsamples chroma, which would blur the normal X/Y components). Decoded to RGBA8 on the GPU.
Writes assets/manifest.json (paths relative to assets/), which the viewer reads to decide between baked parts and the STL fallback.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # simulator/
PARTS_DIR = ROOT / "bakes"                          # Blender outputs: full-resolution PNG maps, bake reports, meshes
ASSETS = ROOT / "assets"
MESHES = ASSETS / "meshes"
WEB = ASSETS / "textures"
ALL_PARTS = ["torso", "1-A", "1-B", "1-C", "2-A", "2-B", "3-A", "3-B", "4-A", "4-B", "5-A", "5-B"]
MAPS = ("normal", "albedo", "roughness", "ao")
BASISU = shutil.which("basisu")
KTX2_ARGS = {
    "albedo": ["-uastc", "-uastc_level", "2", "-uastc_rdo_l", "1.0", "-srgb", "-mipmap", "-mip_srgb", "-mip_clamp"],
    "normal": ["-uastc", "-uastc_level", "2", "-normal_map", "-mipmap", "-mip_renorm", "-mip_clamp"],
    "orm": ["-uastc", "-uastc_level", "2", "-uastc_rdo_l", "1.0", "-linear", "-mipmap", "-mip_linear", "-mip_clamp"],
}


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def halve(a, times):
    """Exact 2×2 box average, applied `times` times (bake sizes are powers of two)."""
    for _ in range(times):
        h, w = a.shape[:2]
        a = a.reshape(h // 2, 2, w // 2, 2, *a.shape[2:]).mean(axis=(1, 3))
    return a


def load(path, channels=3):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return a if channels == 3 else a[..., 0]


def to_u8(a):
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def complete(part):
    meta = PARTS_DIR / f"{part}.json"
    if not meta.exists():
        return False
    if not all((PARTS_DIR / f"{part}_{m}.png").exists() for m in MAPS):
        return False
    if not all(mesh_file(part, ext).exists() for ext in (".bin", ".layout.json")):
        return False
    # A bake whose rays found nothing (e.g. collapsed UVs) leaves a uniformly black albedo; treat it as missing.
    _, hi = zip(*Image.open(PARTS_DIR / f"{part}_albedo.png").convert("RGB").getextrema())
    if max(hi) == 0:
        print(f"  {part}: bake is blank (black albedo), ignoring it", flush=True)
        return False
    return True


def mesh_file(part, ext):
    """A fresh bake leaves the mesh in bakes/; a published one lives in assets/meshes/."""
    baked = PARTS_DIR / f"{part}{ext}"
    return baked if baked.exists() else MESHES / f"{part}{ext}"


def publish_mesh(part):
    MESHES.mkdir(parents=True, exist_ok=True)
    for ext in (".bin", ".layout.json"):
        src, dst = PARTS_DIR / f"{part}{ext}", MESHES / f"{part}{ext}"
        if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
            shutil.copy2(src, dst)


def encode_ktx2(pixels, kind, out, tmp):
    png = Path(tmp) / f"{out.stem}.png"
    Image.fromarray(pixels).save(png, compress_level=1)
    cmd = [BASISU, "-ktx2", *KTX2_ARGS[kind], "-file", str(png), "-output_file", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    png.unlink(missing_ok=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"basisu failed for {out.name}: {res.stdout[-400:]} {res.stderr[-400:]}")


def build_part(part, force, ktx2):
    bake = json.loads((PARTS_DIR / f"{part}.json").read_text())
    publish_mesh(part)
    layout = json.loads((MESHES / f"{part}.layout.json").read_text())
    size = bake["texture_size"]
    tiers = {"high": size, "balanced": size // 2}
    newest_src = max((PARTS_DIR / f"{part}_{m}.png").stat().st_mtime for m in MAPS)
    record = {"mesh": f"meshes/{part}.bin", "layout": f"meshes/{part}.layout.json", "triangles": layout["triangles"], "vertices": layout["vertices"], "textures": {}}
    sources = None
    fresh = lambda p: p.exists() and p.stat().st_mtime >= newest_src  # noqa: E731
    for tier, px in tiers.items():
        webp = {k: WEB / tier / f"{part}_{k}.webp" for k in ("albedo", "normal", "orm")}
        kt = {k: WEB / tier / f"{part}_{k}.ktx2" for k in ("albedo", "normal", "orm")}
        entry = {"size": px, **{k: f"textures/{tier}/{v.name}" for k, v in webp.items()}}
        want = list(webp.values()) + (list(kt.values()) if ktx2 else [])
        if force or not all(fresh(p) for p in want):
            if sources is None:
                sources = {
                    "albedo": srgb_to_linear(load(PARTS_DIR / f"{part}_albedo.png")),
                    "normal": load(PARTS_DIR / f"{part}_normal.png") * 2.0 - 1.0,
                    "roughness": load(PARTS_DIR / f"{part}_roughness.png", 1),
                    "ao": load(PARTS_DIR / f"{part}_ao.png", 1),
                }
            k = int(round(np.log2(size / px)))
            t0 = time.time()
            (WEB / tier).mkdir(parents=True, exist_ok=True)
            n = halve(sources["normal"], k)
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            pixels = {
                "albedo": to_u8(linear_to_srgb(halve(sources["albedo"], k))),
                "normal": to_u8(n * 0.5 + 0.5),
                "orm": to_u8(np.stack([halve(sources["ao"], k), halve(sources["roughness"], k), np.zeros((px, px), np.float32)], axis=2)),
            }
            Image.fromarray(pixels["albedo"]).save(webp["albedo"], "WEBP", quality=92, method=6)
            Image.fromarray(pixels["normal"]).save(webp["normal"], "WEBP", lossless=True, quality=60, method=4)
            Image.fromarray(pixels["orm"]).save(webp["orm"], "WEBP", lossless=True, quality=60, method=4)
            if ktx2:
                with tempfile.TemporaryDirectory() as tmp:
                    for kind, out in kt.items():
                        encode_ktx2(pixels[kind], kind, out, tmp)
            files = want
            kb = {p.name.split("_", 1)[1]: round(p.stat().st_size / 1024) for p in files}
            print(f"  {part} {tier} {px}px in {time.time() - t0:.1f}s  {kb} KiB", flush=True)
        if all(p.exists() for p in kt.values()):
            entry["ktx2"] = {k: f"textures/{tier}/{v.name}" for k, v in kt.items()}
        record["textures"][tier] = entry
    return record


def main():
    force = "--force" in sys.argv
    ktx2 = "--no-ktx2" not in sys.argv
    if ktx2 and not BASISU:
        print("basisu not found (brew install basis_universal): writing WebP only", flush=True)
        ktx2 = False
    manifest = {"format": "rocky-web-assets/3", "units": "print_mm", "defaultTier": "balanced", "parts": {}}
    for part in ALL_PARTS:
        if not complete(part):
            print(f"  {part}: bake incomplete, viewer will use the STL fallback", flush=True)
            continue
        manifest["parts"][part] = build_part(part, force, ktx2)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    with_ktx2 = sum(1 for p in manifest["parts"].values() if all("ktx2" in t for t in p["textures"].values()))
    print(f"manifest: {len(manifest['parts'])}/{len(ALL_PARTS)} parts baked, {with_ktx2} with KTX2")


if __name__ == "__main__":
    main()

// Ground model shared by the contact solver and the controller's foothold planner.
//
// The floor is a plane through the origin tilted about the world y axis (uphill toward +x). On top of it sit
// axis-aligned boxes (steps, blocks, planks), each with its own friction, and friction patches on the floor itself
// (a polished tile, a rug). The controller only queries the terrain as a physical robot could perceive it
// (height and friction under a point); the solver collides against the same geometry.
export class Terrain {
  constructor({ slopeDeg = 0, boxes = [], patches = [], mu = 0.8, restitutionAlpha = 0.6 } = {}) {
    this.slope = (slopeDeg * Math.PI) / 180;
    this.boxes = boxes.map((b, i) => ({ id: b.id ?? `box${i}`, mu: b.mu ?? mu, label: b.label ?? '', min: [...b.min], max: [...b.max] }));
    // patches: {min:[x, y], max:[x, y], mu, label} rectangles on the floor plane (projected along z)
    this.patches = patches.map((p, i) => ({ id: p.id ?? `patch${i}`, mu: p.mu, label: p.label ?? '', min: [...p.min], max: [...p.max] }));
    this.mu = mu; this.alpha = restitutionAlpha;
  }
  planeNormal() { return [-Math.sin(this.slope), 0, Math.cos(this.slope)]; }
  height(x) { return Math.tan(this.slope) * x; }
  // Floor friction under (x, y): the last matching patch wins.
  muAt(x, y) {
    let mu = this.mu;
    for (const p of this.patches) if (x >= p.min[0] && x <= p.max[0] && y >= p.min[1] && y <= p.max[1]) mu = p.mu;
    return mu;
  }
  // Height of the walkable surface under (x, y): the plane, or the top of the highest box containing the point.
  // `margin` widens boxes so a foot of that radius near an edge counts as standing on the block.
  heightAt(x, y, margin = 0) {
    let z = this.height(x);
    for (const b of this.boxes) {
      if (x >= b.min[0] - margin && x <= b.max[0] + margin && y >= b.min[1] - margin && y <= b.max[1] + margin) z = Math.max(z, b.max[2]);
    }
    return z;
  }
  // Horizontal distance from (x, y) to the nearest vertical edge of any block (Infinity without blocks): a round foot
  // closer than its radius to an edge rests on the corner and rolls off.
  edgeDistance(x, y) {
    let d = Infinity;
    for (const b of this.boxes) {
      const inside = x >= b.min[0] && x <= b.max[0] && y >= b.min[1] && y <= b.max[1];
      const dx = Math.max(b.min[0] - x, 0, x - b.max[0]), dy = Math.max(b.min[1] - y, 0, y - b.max[1]);
      d = Math.min(d, inside ? Math.min(x - b.min[0], b.max[0] - x, y - b.min[1], b.max[1] - y) : Math.hypot(dx, dy));
    }
    return d;
  }
  // Highest surface under a straight segment a→b, sampled every ≈5 mm, for swing-foot clearance.
  maxHeightAlong(a, b, margin = 0) {
    const n = Math.max(2, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.005));
    let z = -Infinity;
    for (let i = 0; i <= n; i++) { const u = i / n; z = Math.max(z, this.heightAt(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, margin)); }
    return z;
  }
  describe() {
    return { slope: this.slope, mu: this.mu, boxes: this.boxes.map((b) => ({ ...b })), patches: this.patches.map((p) => ({ ...p })) };
  }
}

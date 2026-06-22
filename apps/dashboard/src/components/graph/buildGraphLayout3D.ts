import type { Entity, GraphEdge } from '@/types';

import { buildGraphSkeleton } from './graphClusterCore';
import {
  CLUSTER_R_BASE,
  CLUSTER_R_K,
  COLOR,
  DEFAULT_SEED,
  SPHERE_R,
  SUNFLOWER_ANGLE,
  TYPES,
} from './graphConstants';
import type { GraphCluster3D, GraphLayout3D, GraphNode3D } from './graphTypes';

// mulberry32 — same deterministic RNG as the 2D builder, so the 3D layout is stable across reloads.
function makeRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 3D sibling of buildGraphLayout: same topology (from buildGraphSkeleton) laid out in space the way the
 * design export does — cluster centers on a Fibonacci sphere (the first cluster at the origin), nodes
 * packed in a solid ball around each center with the hub at the center. Adds the glow-shader `phase` and
 * screen-space `size` per node. Node `idx` matches the 2D builder so a selection survives a mode toggle.
 */
export function buildGraphLayout3D(
  entities: Entity[],
  edges: GraphEdge[],
  opts?: { seed?: number },
): GraphLayout3D {
  const rnd = makeRng(opts?.seed ?? DEFAULT_SEED);
  const skeleton = buildGraphSkeleton(entities, edges);

  const clusters: GraphCluster3D[] = [];
  const nodes: GraphNode3D[] = [];

  const m = skeleton.clusters.length - 1; // number of outer (non-origin) clusters
  // With ≤2 clusters the export's `(k/(m-1))` term divides by zero; clamp the denominator and pin the
  // single outer cluster to the equator so a 1–2 type graph still lays out cleanly.
  const denom = m > 1 ? m - 1 : 1;

  skeleton.clusters.forEach((cl, ci) => {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    if (ci !== 0) {
      const k = ci - 1;
      const y = m > 1 ? 1 - (k / denom) * 2 : 0;
      const rr = Math.sqrt(Math.max(0, 1 - y * y));
      const th = k * SUNFLOWER_ANGLE;
      const R = SPHERE_R * (0.82 + rnd() * 0.5);
      cx = Math.cos(th) * rr * R;
      cy = y * R * 0.8;
      cz = Math.sin(th) * rr * R;
    }

    const count = cl.count;
    const clR = CLUSTER_R_BASE + Math.sqrt(count) * CLUSTER_R_K;

    for (let i = 0; i < count; i++) {
      const base = skeleton.nodes[cl.first + i];
      let dx: number;
      let dy: number;
      let dz: number;
      do {
        dx = rnd() * 2 - 1;
        dy = rnd() * 2 - 1;
        dz = rnd() * 2 - 1;
      } while (dx * dx + dy * dy + dz * dz > 1 || dx === 0);
      const rad = clR * Math.cbrt(rnd()) * (base.isHub ? 0 : 1);
      const d = base.deg;
      nodes.push({
        ...base,
        pos: [cx + dx * rad, cy + dy * rad, cz + dz * rad],
        phase: rnd() * 6.28,
        size: base.isHub ? 52 + Math.min(d, 12) * 4 : 22 + Math.min(d, 8) * 3,
      });
    }

    clusters.push({ ...cl, center: [cx, cy, cz], clR });
  });

  return {
    TYPES,
    COLOR,
    clusters,
    nodes,
    edges: skeleton.edges,
    adj: skeleton.adj,
    typeCount: skeleton.typeCount,
  };
}

import type { Entity, GraphEdge } from '@/types';

import { buildGraphSkeleton } from './graphClusterCore';
import {
  COLOR,
  DEFAULT_SEED,
  JITTER,
  RING_PHASE,
  RING_RADII,
  SUNFLOWER_ANGLE,
  TYPES,
} from './graphConstants';
import type { GraphCluster, GraphLayout, GraphNode } from './graphTypes';

// mulberry32 — the deterministic RNG from the design, so layouts are stable across reloads.
function makeRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Clusters fill rings outward: the first cluster sits at the center, the next four on ring 1, etc.
function ringForCluster(i: number): number {
  if (i === 0) return 0;
  if (i <= 4) return 1;
  if (i <= 7) return 2;
  return 3;
}

/**
 * Map server entities + edges into the {clusters, nodes, edges, adj, ...} shape the canvas renderer
 * consumes. Topology (clustering, hubs, adjacency, edges) comes from buildGraphSkeleton; this builder
 * only lays the clusters out on concentric rings and packs each cluster's nodes by sunflower spiral.
 */
export function buildGraphLayout(
  entities: Entity[],
  edges: GraphEdge[],
  opts?: { seed?: number },
): GraphLayout {
  const rnd = makeRng(opts?.seed ?? DEFAULT_SEED);
  const skeleton = buildGraphSkeleton(entities, edges);

  if (skeleton.nodes.length === 0) {
    return {
      TYPES,
      COLOR,
      clusters: [],
      nodes: [],
      edges: [],
      adj: [],
      typeCount: skeleton.typeCount,
      bounds: { minX: -100, minY: -100, maxX: 100, maxY: 100 },
    };
  }

  // How many clusters land on each ring (drives the angular slot spacing within a ring).
  const ringList = skeleton.clusterTypes.map((_t, ci) => ringForCluster(ci));
  const ringSize: Record<number, number> = {};
  ringList.forEach((r) => {
    ringSize[r] = (ringSize[r] ?? 0) + 1;
  });

  const clusters: GraphCluster[] = [];
  const nodes: GraphNode[] = [];

  skeleton.clusters.forEach((cl, ci) => {
    const ring = ringList[ci];
    let cx = 0;
    let cy = 0;
    if (ring !== 0) {
      const k = ringList.slice(0, ci).filter((r) => r === ring).length;
      const ang = (k / ringSize[ring]) * Math.PI * 2 + (RING_PHASE[ring] ?? 0);
      const radius = RING_RADII[ring] * (0.86 + rnd() * 0.28);
      cx = Math.cos(ang) * radius;
      cy = Math.sin(ang) * radius;
    }

    const count = cl.count;
    const R = 90 + Math.sqrt(count) * 42;

    for (let i = 0; i < count; i++) {
      const base = skeleton.nodes[cl.first + i];
      const t = i / Math.max(1, count - 1);
      const rad = R * Math.sqrt(t) * 0.96;
      const a = i * SUNFLOWER_ANGLE;
      const x = cx + Math.cos(a) * rad + (rnd() - 0.5) * JITTER;
      const y = cy + Math.sin(a) * rad + (rnd() - 0.5) * JITTER;
      const d = base.deg;
      nodes.push({
        ...base,
        x,
        y,
        r0: base.isHub ? 16 + Math.min(d, 10) : 5 + Math.min(d, 7) * 1.1,
      });
    }

    clusters.push({ ...cl, cx, cy, R });
  });

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of nodes) {
    if (n.x < minX) minX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.x > maxX) maxX = n.x;
    if (n.y > maxY) maxY = n.y;
  }
  const pad = 300;

  return {
    TYPES,
    COLOR,
    clusters,
    nodes,
    edges: skeleton.edges,
    adj: skeleton.adj,
    typeCount: skeleton.typeCount,
    bounds: { minX: minX - pad, minY: minY - pad, maxX: maxX + pad, maxY: maxY + pad },
  };
}

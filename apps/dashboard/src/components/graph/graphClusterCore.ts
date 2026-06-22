import type { Entity, EntityType, GraphEdge } from '@/types';

import { COLOR, TYPES } from './graphConstants';
import type { GraphClusterBase, GraphEdgeLayout, GraphNodeBase, GraphSkeleton } from './graphTypes';

function emptyTypeCount(): Record<EntityType, number> {
  return {
    Person: 0,
    Place: 0,
    Concept: 0,
    Project: 0,
    Event: 0,
    Organization: 0,
    Document: 0,
    Task: 0,
  };
}

/**
 * Dimension-independent core shared by the 2D and 3D layout builders. Clusters entities by type (one
 * cluster per present type, in TYPES order), sorts each cluster's members by degree so the highest-
 * degree node becomes the hub, and builds the deduped undirected edge list with bridge classification.
 * The 2D and 3D builders consume this and differ only in how they place clusters/nodes in space, which
 * keeps node `idx` identical across both so a selection survives a 2D↔3D toggle.
 */
export function buildGraphSkeleton(entities: Entity[], edges: GraphEdge[]): GraphSkeleton {
  const typeCount = emptyTypeCount();

  if (entities.length === 0) {
    return { clusterTypes: [], nodes: [], clusters: [], adj: [], edges: [], typeCount };
  }

  // id → original index, and an undirected adjacency over original indices (dangling/self edges dropped).
  const idToOrig = new Map<string, number>();
  entities.forEach((e, i) => idToOrig.set(e.id, i));
  const origAdj: Set<number>[] = entities.map(() => new Set<number>());
  for (const edge of edges) {
    const a = idToOrig.get(edge.source_entity_id);
    const b = idToOrig.get(edge.target_entity_id);
    if (a === undefined || b === undefined || a === b) continue;
    origAdj[a].add(b);
    origAdj[b].add(a);
  }
  const deg = origAdj.map((s) => s.size);

  const clusterTypes = TYPES.filter((t) => entities.some((e) => e.type === t));

  const nodes: GraphNodeBase[] = [];
  const clusters: GraphClusterBase[] = [];
  const origToFinal = new Array<number>(entities.length).fill(-1);

  clusterTypes.forEach((type, ci) => {
    const members = entities
      .map((e, i) => ({ i, e }))
      .filter((x) => x.e.type === type)
      .sort((p, q) => deg[q.i] - deg[p.i] || p.i - q.i);

    const count = members.length;
    const first = nodes.length;
    let hub = first;

    members.forEach((m, i) => {
      const isHub = i === 0;
      const finalIdx = nodes.length;
      origToFinal[m.i] = finalIdx;
      if (isHub) hub = finalIdx;
      typeCount[type] += 1;
      nodes.push({
        idx: finalIdx,
        id: m.e.id,
        cluster: ci,
        type,
        name: m.e.name,
        isHub,
        deg: deg[m.i],
        confidence: m.e.confidence,
        sourceCount: m.e.source_note_ids.length,
        aliases: m.e.aliases,
        props: m.e.properties,
      });
    });

    clusters.push({ name: type, ci, accent: COLOR[type], hub, first, count });
  });

  // Adjacency + deduped edges in final-index space; a bridge spans two different clusters.
  const adj: Set<number>[] = nodes.map(() => new Set<number>());
  const layoutEdges: GraphEdgeLayout[] = [];
  origAdj.forEach((neighbors, orig) => {
    const a = origToFinal[orig];
    neighbors.forEach((n) => {
      const b = origToFinal[n];
      adj[a].add(b);
      if (a < b) {
        layoutEdges.push({ a, b, bridge: nodes[a].cluster !== nodes[b].cluster });
      }
    });
  });

  return { clusterTypes, nodes, clusters, adj, edges: layoutEdges, typeCount };
}

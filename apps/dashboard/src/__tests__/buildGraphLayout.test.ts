import { describe, expect, it } from 'vitest';

import { buildGraphLayout } from '@/components/graph/buildGraphLayout';
import { COLOR } from '@/components/graph/graphConstants';
import type { Entity, EntityType, GraphEdge } from '@/types';

function ent(
  id: string,
  name: string,
  type: EntityType,
  opts?: { aliases?: string[]; properties?: Record<string, unknown>; sources?: string[] },
): Entity {
  return {
    id,
    name,
    type,
    aliases: opts?.aliases ?? [],
    properties: opts?.properties ?? {},
    confidence: 0.5,
    source_note_ids: opts?.sources ?? [],
  };
}

function edge(source: string, target: string): GraphEdge {
  return {
    source_entity_id: source,
    target_entity_id: target,
    type: 'related_to',
    confidence: 0.9,
  };
}

// p1 is a Person hub linked to p2, p3 (Person) and c1 (Concept); d1 is an isolated Document.
function fixture() {
  const entities: Entity[] = [
    ent('p1', 'Alice', 'Person', { sources: ['n1', 'n2'], properties: { Role: 'Engineer' } }),
    ent('p2', 'Bob', 'Person'),
    ent('p3', 'Carol', 'Person'),
    ent('c1', 'Emergence', 'Concept'),
    ent('d1', 'RFC', 'Document'),
  ];
  const edges: GraphEdge[] = [edge('p1', 'p2'), edge('p1', 'p3'), edge('p1', 'c1')];
  return { entities, edges };
}

describe('buildGraphLayout', () => {
  it('produces one node per entity and dedupes valid edges', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    expect(layout.nodes).toHaveLength(5);
    expect(layout.edges).toHaveLength(3);
  });

  it('maps entity id → node and carries sourceCount + props', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    const p1 = layout.nodes.find((n) => n.id === 'p1');
    expect(p1).toBeDefined();
    expect(p1?.name).toBe('Alice');
    expect(p1?.sourceCount).toBe(2);
    expect(p1?.props).toEqual({ Role: 'Engineer' });
  });

  it('drops dangling edges whose endpoint is missing', () => {
    const { entities } = fixture();
    const edges: GraphEdge[] = [edge('p1', 'p2'), edge('p1', 'ghost')];
    const layout = buildGraphLayout(entities, edges);
    expect(layout.edges).toHaveLength(1);
  });

  it('builds symmetric adjacency with deg === adj size', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    layout.nodes.forEach((n) => {
      expect(n.deg).toBe(layout.adj[n.idx].size);
    });
    const p1 = layout.nodes.find((n) => n.id === 'p1')!;
    const p2 = layout.nodes.find((n) => n.id === 'p2')!;
    expect(p1.deg).toBe(3);
    expect(layout.adj[p1.idx].has(p2.idx)).toBe(true);
    expect(layout.adj[p2.idx].has(p1.idx)).toBe(true);
  });

  it('clusters by type with one cluster per present type, colored by type', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    expect(layout.clusters).toHaveLength(3); // Person, Concept, Document
    layout.nodes.forEach((n) => {
      const cluster = layout.clusters[n.cluster];
      expect(cluster.accent).toBe(COLOR[n.type]);
      expect(cluster.name).toBe(n.type);
    });
    expect(layout.typeCount.Person).toBe(3);
    expect(layout.typeCount.Concept).toBe(1);
    expect(layout.typeCount.Document).toBe(1);
  });

  it('picks the highest-degree node of a type as the cluster hub', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    const personCluster = layout.clusters.find((c) => c.name === 'Person')!;
    expect(layout.nodes[personCluster.hub].id).toBe('p1');
    expect(layout.nodes[personCluster.hub].isHub).toBe(true);
  });

  it('marks inter-cluster edges as bridges and intra-cluster edges as non-bridges', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    const p1 = layout.nodes.find((n) => n.id === 'p1')!;
    const c1 = layout.nodes.find((n) => n.id === 'c1')!;
    const p2 = layout.nodes.find((n) => n.id === 'p2')!;
    const bridge = layout.edges.find(
      (e) => (e.a === p1.idx && e.b === c1.idx) || (e.a === c1.idx && e.b === p1.idx),
    )!;
    const intra = layout.edges.find(
      (e) => (e.a === p1.idx && e.b === p2.idx) || (e.a === p2.idx && e.b === p1.idx),
    )!;
    expect(bridge.bridge).toBe(true);
    expect(intra.bridge).toBe(false);
  });

  it('keeps isolated nodes (degree 0) in their type cluster', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout(entities, edges);
    const d1 = layout.nodes.find((n) => n.id === 'd1')!;
    expect(d1.deg).toBe(0);
    expect(layout.adj[d1.idx].size).toBe(0);
    expect(layout.clusters[d1.cluster].name).toBe('Document');
  });

  it('returns empty arrays and finite bounds for empty input', () => {
    const layout = buildGraphLayout([], []);
    expect(layout.nodes).toHaveLength(0);
    expect(layout.edges).toHaveLength(0);
    expect(layout.clusters).toHaveLength(0);
    for (const v of Object.values(layout.bounds)) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('is deterministic for a given seed', () => {
    const { entities, edges } = fixture();
    const a = buildGraphLayout(entities, edges, { seed: 42 });
    const b = buildGraphLayout(entities, edges, { seed: 42 });
    expect(a.nodes.map((n) => [n.x, n.y])).toEqual(b.nodes.map((n) => [n.x, n.y]));
  });
});

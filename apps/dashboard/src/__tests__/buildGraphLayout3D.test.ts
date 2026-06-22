import { describe, expect, it } from 'vitest';

import { buildGraphLayout } from '@/components/graph/buildGraphLayout';
import { buildGraphLayout3D } from '@/components/graph/buildGraphLayout3D';
import { COLOR } from '@/components/graph/graphConstants';
import type { Entity, EntityType, GraphEdge } from '@/types';

function ent(id: string, name: string, type: EntityType): Entity {
  return { id, name, type, aliases: [], properties: {}, confidence: 0.5, source_note_ids: [] };
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
    ent('p1', 'Alice', 'Person'),
    ent('p2', 'Bob', 'Person'),
    ent('p3', 'Carol', 'Person'),
    ent('c1', 'Emergence', 'Concept'),
    ent('d1', 'RFC', 'Document'),
  ];
  const edges: GraphEdge[] = [edge('p1', 'p2'), edge('p1', 'p3'), edge('p1', 'c1')];
  return { entities, edges };
}

describe('buildGraphLayout3D', () => {
  it('produces one node per entity with finite 3D positions', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout3D(entities, edges);
    expect(layout.nodes).toHaveLength(5);
    expect(layout.edges).toHaveLength(3);
    layout.nodes.forEach((n) => {
      expect(n.pos).toHaveLength(3);
      n.pos.forEach((v) => expect(Number.isFinite(v)).toBe(true));
      expect(n.size).toBeGreaterThan(0);
    });
  });

  it('shares node ordering and idx with the 2D builder so a selection survives a mode toggle', () => {
    const { entities, edges } = fixture();
    const l2 = buildGraphLayout(entities, edges);
    const l3 = buildGraphLayout3D(entities, edges);
    expect(l3.nodes.map((n) => n.id)).toEqual(l2.nodes.map((n) => n.id));
    expect(l3.nodes.map((n) => n.idx)).toEqual(l2.nodes.map((n) => n.idx));
    expect(l3.nodes.map((n) => n.cluster)).toEqual(l2.nodes.map((n) => n.cluster));
  });

  it('clusters by type with one cluster per present type, colored by type', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout3D(entities, edges);
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

  it('places the cluster hub at the cluster center', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout3D(entities, edges);
    const personCluster = layout.clusters.find((c) => c.name === 'Person')!;
    expect(layout.nodes[personCluster.hub].id).toBe('p1');
    expect(layout.nodes[personCluster.hub].pos).toEqual(personCluster.center);
  });

  it('puts the first cluster at the origin', () => {
    const { entities, edges } = fixture();
    const layout = buildGraphLayout3D(entities, edges);
    expect(layout.clusters[0].center).toEqual([0, 0, 0]);
  });

  it('is deterministic for a given seed', () => {
    const { entities, edges } = fixture();
    const a = buildGraphLayout3D(entities, edges, { seed: 42 });
    const b = buildGraphLayout3D(entities, edges, { seed: 42 });
    expect(a.nodes.map((n) => n.pos)).toEqual(b.nodes.map((n) => n.pos));
  });

  it('lays out a single-type graph without NaN (the Fibonacci small-count guard)', () => {
    const entities: Entity[] = [ent('p1', 'Alice', 'Person'), ent('p2', 'Bob', 'Person')];
    const layout = buildGraphLayout3D(entities, []);
    expect(layout.clusters).toHaveLength(1);
    layout.nodes.forEach((n) => n.pos.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
  });

  it('lays out a two-cluster graph without NaN', () => {
    const entities: Entity[] = [ent('p1', 'Alice', 'Person'), ent('c1', 'Emergence', 'Concept')];
    const layout = buildGraphLayout3D(entities, []);
    expect(layout.clusters).toHaveLength(2);
    layout.clusters.forEach((c) => c.center.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
    layout.nodes.forEach((n) => n.pos.forEach((v) => expect(Number.isFinite(v)).toBe(true)));
  });

  it('returns empty arrays for empty input', () => {
    const layout = buildGraphLayout3D([], []);
    expect(layout.nodes).toHaveLength(0);
    expect(layout.edges).toHaveLength(0);
    expect(layout.clusters).toHaveLength(0);
  });
});

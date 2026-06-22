import type { EntityType } from '@/types';

// Dimension-independent node fields shared by the 2D and 3D layouts (positions added per dimension).
export interface GraphNodeBase {
  idx: number;
  id: string;
  cluster: number;
  type: EntityType;
  name: string;
  isHub: boolean;
  deg: number;
  confidence: number;
  sourceCount: number;
  aliases: string[];
  props: Record<string, unknown>;
}

export interface GraphNode extends GraphNodeBase {
  x: number;
  y: number;
  r0: number;
}

// 3D node: world position, glow-shader animation phase, and screen-space point size.
export interface GraphNode3D extends GraphNodeBase {
  pos: [number, number, number];
  phase: number;
  size: number;
}

export interface GraphClusterBase {
  name: string;
  ci: number;
  accent: string;
  hub: number;
  first: number;
  count: number;
}

export interface GraphCluster extends GraphClusterBase {
  cx: number;
  cy: number;
  R: number;
}

export interface GraphCluster3D extends GraphClusterBase {
  center: [number, number, number];
  clR: number;
}

export interface GraphEdgeLayout {
  a: number;
  b: number;
  bridge: boolean;
}

// Output of buildGraphSkeleton — the type-clustering, adjacency and edge topology both layouts reuse.
export interface GraphSkeleton {
  clusterTypes: EntityType[];
  nodes: GraphNodeBase[];
  clusters: GraphClusterBase[];
  adj: Set<number>[];
  edges: GraphEdgeLayout[];
  typeCount: Record<EntityType, number>;
}

export interface GraphBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface GraphLayout {
  TYPES: readonly EntityType[];
  COLOR: Record<EntityType, string>;
  clusters: GraphCluster[];
  nodes: GraphNode[];
  edges: GraphEdgeLayout[];
  adj: Set<number>[];
  typeCount: Record<EntityType, number>;
  bounds: GraphBounds;
}

export interface GraphLayout3D {
  TYPES: readonly EntityType[];
  COLOR: Record<EntityType, string>;
  clusters: GraphCluster3D[];
  nodes: GraphNode3D[];
  edges: GraphEdgeLayout[];
  adj: Set<number>[];
  typeCount: Record<EntityType, number>;
}

// Shared imperative surface so the React HUD can drive either the 2D canvas or the 3D Three.js engine.
export interface GraphEngineHandle {
  start(): void;
  destroy(): void;
  zoomIn(): void;
  zoomOut(): void;
  fit(): void;
  deselect(): void;
  focusEntity(idx: number): void;
}

export interface SelectedRelation {
  idx: number;
  name: string;
  type: EntityType;
  color: string;
}

export interface SelectedEntity {
  idx: number;
  name: string;
  type: EntityType;
  color: string;
  cluster: string;
  degree: number;
  sourceCount: number;
  confPct: string;
  aliases: string[];
  hasAliases: boolean;
  propList: { k: string; v: string }[];
  hasProps: boolean;
  relations: SelectedRelation[];
}

export type GraphLod = 'CLUSTERS' | 'ENTITIES' | 'DETAIL';

export interface GraphHud {
  zoomPct: number;
  lod: GraphLod;
  count: string;
}

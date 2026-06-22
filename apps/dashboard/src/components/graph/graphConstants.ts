import type { EntityType } from '@/types';

export const TYPES: readonly EntityType[] = [
  'Person',
  'Place',
  'Concept',
  'Project',
  'Event',
  'Organization',
  'Document',
  'Task',
];

export const COLOR: Record<EntityType, string> = {
  Person: '#e3a85f',
  Place: '#5fc98a',
  Concept: '#5fc9d4',
  Project: '#b488e6',
  Event: '#e87f9a',
  Organization: '#6f9be0',
  Document: '#9aa6b8',
  Task: '#dcc46a',
};

// HUD chrome accent (amber) — distinct from the per-type node colors above. Matches the design export.
export const ACCENT = '#d99a5b';

export const RING_RADII = [0, 1050, 1980, 2760];
export const RING_PHASE: Record<number, number> = { 0: 0, 1: 0.3, 2: 0.0, 3: 0.55 };

// Golden-angle sunflower packing within a cluster, plus position jitter (world units). From the design.
export const SUNFLOWER_ANGLE = 2.399963;
export const JITTER = 14;

// ── 3D view (faithful to the export's Three.js scene; world units unless noted) ──
export const CAM_FOV = 52;
export const SPHERE_R = 1500; // base radius of the Fibonacci sphere the clusters sit on
export const CLUSTER_R_BASE = 120; // node-cloud radius = CLUSTER_R_BASE + sqrt(count) * CLUSTER_R_K
export const CLUSTER_R_K = 22;
export const RADIUS_MIN = 280; // camera orbit radius clamp
export const RADIUS_MAX = 8200;
export const RADIUS_DEFAULT = 4600;
export const FADE_NEAR = 3200; // node distance-fade smoothstep range (camera-space depth)
export const FADE_FAR = 7000;
export const LOD_NEAR = 720; // cluster expand: fully expanded when camera is within LOD_NEAR
export const LOD_FAR = 2200; // …and collapsed to an orb past LOD_FAR
export const LOD_R_CLUSTERS = 2500; // camera radius → CLUSTERS / ENTITIES / DETAIL label thresholds
export const LOD_R_ENTITIES = 1150;
export const ORB_SCALE = 5.2; // cluster orb sprite scale = clR * ORB_SCALE

// Zoom level-of-detail thresholds: below Z0 clusters render as orbs; between Z0..Z1 they resolve into
// nodes; at ZLBL node labels appear.
export const Z0 = 0.3;
export const Z1 = 0.52;
export const ZLBL = 0.82;

export const DEFAULT_SEED = 20260621;

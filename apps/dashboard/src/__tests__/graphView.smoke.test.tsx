import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildGraphLayout } from '@/components/graph/buildGraphLayout';
import { buildGraphLayout3D } from '@/components/graph/buildGraphLayout3D';
import { KnowledgeGraphView } from '@/components/graph/KnowledgeGraphView';
import { HudToast } from '@/components/hud/HudToast';
import type { Entity, GraphEdge } from '@/types';

vi.mock('@/lib/api', () => ({ graphApi: { full: vi.fn() } }));

import GraphPage from '@/app/(app)/graph/page';
import { graphApi } from '@/lib/api';

const entities: Entity[] = [
  {
    id: 'a',
    name: 'Alice',
    type: 'Person',
    aliases: [],
    properties: {},
    confidence: 0.9,
    source_note_ids: [],
  },
  {
    id: 'b',
    name: 'Bob',
    type: 'Person',
    aliases: [],
    properties: {},
    confidence: 0.8,
    source_note_ids: [],
  },
];
const edges: GraphEdge[] = [
  { source_entity_id: 'a', target_entity_id: 'b', type: 'related_to', confidence: 0.9 },
];

const mockFull = graphApi.full as ReturnType<typeof vi.fn>;

function view() {
  return (
    <KnowledgeGraphView
      layout2d={buildGraphLayout(entities, edges)}
      layout3d={buildGraphLayout3D(entities, edges)}
    />
  );
}

describe('KnowledgeGraphView', () => {
  // jsdom has no WebGL/2D canvas context; both engines guard on it, so the view must still mount.
  it('renders stage, legend and zoom controls (defaults to 3D)', () => {
    render(view());
    expect(screen.getByTestId('graph-stage')).toBeInTheDocument();
    expect(screen.getByText('Person')).toBeInTheDocument();
    expect(screen.getByLabelText('Zoom in')).toBeInTheDocument();
    expect(screen.getByText(/entities ·/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '3D' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('toggles between 3D and 2D without crashing', () => {
    render(view());
    const btn2d = screen.getByRole('button', { name: '2D' });
    fireEvent.click(btn2d);
    expect(btn2d).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: '3D' })).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(screen.getByRole('button', { name: '3D' }));
    expect(screen.getByRole('button', { name: '3D' })).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('GraphPage', () => {
  beforeEach(() => vi.clearAllMocks());

  it('renders the graph after a successful load', async () => {
    mockFull.mockResolvedValue({ nodes: entities, edges });
    render(
      <HudToast>
        <GraphPage />
      </HudToast>,
    );
    await waitFor(() => expect(screen.getByTestId('graph-stage')).toBeInTheDocument());
  });

  it('shows an error with retry when the load fails', async () => {
    mockFull.mockRejectedValue(new Error('boom'));
    render(
      <HudToast>
        <GraphPage />
      </HudToast>,
    );
    await waitFor(() => expect(screen.getByText('Retry')).toBeInTheDocument());
  });
});

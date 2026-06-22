'use client';

import { type ReactNode, useEffect, useState } from 'react';

import { buildGraphLayout } from '@/components/graph/buildGraphLayout';
import { buildGraphLayout3D } from '@/components/graph/buildGraphLayout3D';
import type { GraphLayout, GraphLayout3D } from '@/components/graph/graphTypes';
import { KnowledgeGraphView } from '@/components/graph/KnowledgeGraphView';
import { HudButton } from '@/components/hud/HudButton';
import { HudSpinner } from '@/components/hud/HudSpinner';
import { useToast } from '@/components/hud/HudToast';
import { graphApi } from '@/lib/api';

function Centered({ children }: { children: ReactNode }) {
  return <div className="flex flex-1 items-center justify-center p-6">{children}</div>;
}

export default function GraphPage() {
  const [layout, setLayout] = useState<{ layout2d: GraphLayout; layout3d: GraphLayout3D } | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const { addToast } = useToast();

  const retry = () => {
    setLoading(true);
    setError(null);
    setReloadKey((k) => k + 1);
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await graphApi.full();
        if (cancelled) return;
        setLayout({
          layout2d: buildGraphLayout(data.nodes, data.edges),
          layout3d: buildGraphLayout3D(data.nodes, data.edges),
        });
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : 'Failed to load graph';
        setError(msg);
        addToast(msg, 'error');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [reloadKey, addToast]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {loading ? (
        <Centered>
          <HudSpinner size={28} />
        </Centered>
      ) : error ? (
        <Centered>
          <div className="flex flex-col items-center gap-3">
            <p className="font-mono text-xs text-aiki-danger">{error}</p>
            <HudButton onClick={retry}>Retry</HudButton>
          </div>
        </Centered>
      ) : !layout || layout.layout2d.nodes.length === 0 ? (
        <Centered>
          <p className="font-mono text-[13px] text-aiki-text-muted">
            No entities yet. Capture some notes and KIO will build your graph.
          </p>
        </Centered>
      ) : (
        <KnowledgeGraphView layout2d={layout.layout2d} layout3d={layout.layout3d} />
      )}
    </div>
  );
}

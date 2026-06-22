'use client';

import { type CSSProperties, useEffect, useRef, useState } from 'react';

import type { EntityType } from '@/types';

import { GraphEngine } from './GraphEngine';
import { GraphEngine3D } from './GraphEngine3D';
import { ACCENT, COLOR, TYPES } from './graphConstants';
import type {
  GraphEngineHandle,
  GraphHud,
  GraphLayout,
  GraphLayout3D,
  SelectedEntity,
} from './graphTypes';

type GraphMode = '2D' | '3D';

const PANEL_BG = 'rgba(13,17,26,.72)';
const HUD_BG = 'rgba(10,13,20,.78)';
const BORDER = '1px solid rgba(255,255,255,.07)';

const mono = "'JetBrains Mono', monospace";

export function KnowledgeGraphView({
  layout2d,
  layout3d,
}: {
  layout2d: GraphLayout;
  layout3d: GraphLayout3D;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvas2dRef = useRef<HTMLCanvasElement>(null);
  const miniRef = useRef<HTMLCanvasElement>(null);
  const canvas3dRef = useRef<HTMLCanvasElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<GraphEngineHandle | null>(null);

  const [mode, setMode] = useState<GraphMode>('3D');
  const [selected, setSelected] = useState<SelectedEntity | null>(null);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [hud, setHud] = useState<GraphHud>({ zoomPct: 0, lod: 'CLUSTERS', count: '—' });

  // The engine reads the live filter each frame without being re-instantiated when it changes.
  const hiddenRef = useRef(hidden);
  useEffect(() => {
    hiddenRef.current = hidden;
  }, [hidden]);

  // Read at construction time so a mode switch re-selects the same node (shared idx) without making the
  // mount effect depend on `selected`.
  const selectedRef = useRef(selected);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const shared = {
      container,
      onSelect: setSelected,
      onHud: setHud,
      isHidden: (t: EntityType) => !!hiddenRef.current[t],
      initialSelected: selectedRef.current?.idx,
    };
    let engine: GraphEngineHandle;
    if (mode === '3D') {
      const canvas = canvas3dRef.current;
      const label = labelRef.current;
      if (!canvas || !label) return;
      engine = new GraphEngine3D({ ...shared, canvas, label, layout: layout3d });
    } else {
      const canvas = canvas2dRef.current;
      const mini = miniRef.current;
      if (!canvas || !mini) return;
      engine = new GraphEngine({
        ...shared,
        canvas,
        mini,
        layout: layout2d,
        glowIntensity: 1,
        showRelations: true,
        ringGuides: true,
      });
    }
    engine.start();
    engineRef.current = engine;
    return () => {
      engine.destroy();
      engineRef.current = null;
    };
  }, [layout2d, layout3d, mode]);

  const toggleType = (t: EntityType) => setHidden((s) => ({ ...s, [t]: !s[t] }));

  return (
    <div
      ref={containerRef}
      data-testid="graph-stage"
      style={{
        position: 'relative',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
        cursor: 'grab',
        fontFamily: "'Spectral', Georgia, serif",
        color: '#c4ccd8',
        background: 'radial-gradient(125% 120% at 50% 32%,#0e1422 0%,#090d16 52%,#06080d 100%)',
      }}
    >
      {mode === '3D' ? (
        <>
          <canvas
            ref={canvas3dRef}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              display: 'block',
            }}
          />
          <div
            ref={labelRef}
            style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}
          />
        </>
      ) : (
        <canvas
          ref={canvas2dRef}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            display: 'block',
          }}
        />
      )}

      {/* header */}
      <div style={{ position: 'absolute', top: 26, left: 30, pointerEvents: 'none' }}>
        <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: '.32em', color: ACCENT }}>
          KNOWLEDGE&nbsp;GRAPH&nbsp;<span style={{ color: '#5fc9d4' }}>/&nbsp;{mode}</span>
        </div>
        <div style={{ fontSize: 13, color: '#8b95a6', marginTop: 7 }}>
          {layout2d.nodes.length} entities · {layout2d.edges.length} relations ·{' '}
          {layout2d.clusters.length} clusters
        </div>
        <div
          style={{
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: '.06em',
            color: '#525c6c',
            marginTop: 9,
            maxWidth: 240,
            lineHeight: 1.6,
          }}
        >
          {mode === '3D'
            ? 'DRAG TO ORBIT · SCROLL TO DIVE IN · CLICK A CLUSTER TO OPEN IT'
            : 'SCROLL TO ZOOM · CLUSTERS RESOLVE INTO ENTITIES AS YOU APPROACH'}
        </div>
      </div>

      {/* type legend / filter */}
      <div
        style={{
          position: 'absolute',
          top: 26,
          right: 30,
          width: 212,
          background: PANEL_BG,
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: BORDER,
          borderRadius: 14,
          padding: '14px 14px 10px',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontFamily: mono,
            fontSize: 10,
            letterSpacing: '.2em',
            color: '#737c8c',
            marginBottom: 11,
          }}
        >
          <span>ENTITY&nbsp;TYPES</span>
          <span style={{ color: '#4b5462', cursor: 'pointer' }} onClick={() => setHidden({})}>
            RESET
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {TYPES.map((t) => {
            const off = !!hidden[t];
            return (
              <div
                key={t}
                onClick={() => toggleType(t)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '6px 7px',
                  borderRadius: 8,
                  cursor: 'pointer',
                  opacity: off ? 0.38 : 1,
                  color: '#c4ccd8',
                }}
              >
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: '50%',
                    flex: 'none',
                    background: COLOR[t],
                    boxShadow: `0 0 8px ${off ? 'transparent' : COLOR[t]}`,
                    filter: off ? 'grayscale(1)' : undefined,
                  }}
                />
                <span style={{ flex: 1, fontSize: 13 }}>{t}</span>
                <span style={{ fontFamily: mono, fontSize: 10.5, color: '#5b6473' }}>
                  {layout2d.typeCount[t]}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* minimap (2D only) */}
      {mode === '2D' && (
        <div
          style={{
            position: 'absolute',
            left: 30,
            bottom: 30,
            width: 184,
            height: 130,
            background: 'rgba(10,13,20,.78)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            border: BORDER,
            borderRadius: 12,
            padding: 8,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 8,
              left: 11,
              fontFamily: mono,
              fontSize: 8.5,
              letterSpacing: '.2em',
              color: '#5b6473',
              zIndex: 2,
              pointerEvents: 'none',
            }}
          >
            MAP
          </div>
          <canvas
            ref={miniRef}
            style={{ width: '100%', height: '100%', display: 'block', cursor: 'pointer' }}
          />
        </div>
      )}

      {/* zoom + LOD HUD */}
      <div
        style={{
          position: 'absolute',
          right: 30,
          bottom: 30,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'flex-end',
          gap: 12,
        }}
      >
        <div
          style={{
            display: 'flex',
            background: 'rgba(10,13,20,.8)',
            border: '1px solid rgba(255,255,255,.08)',
            borderRadius: 11,
            overflow: 'hidden',
          }}
        >
          {(['2D', '3D'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              style={{
                background: mode === m ? 'rgba(217,154,91,.16)' : 'none',
                color: mode === m ? ACCENT : '#7c8696',
                border: 'none',
                padding: '8px 16px',
                cursor: 'pointer',
                fontFamily: mono,
                fontSize: 11,
                letterSpacing: '.16em',
              }}
            >
              {m}
            </button>
          ))}
        </div>
        <div
          style={{
            background: HUD_BG,
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            border: BORDER,
            borderRadius: 11,
            padding: '11px 14px',
            textAlign: 'right',
            minWidth: 150,
          }}
        >
          <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: '.2em', color: '#5b6473' }}>
            DETAIL LEVEL
          </div>
          <div
            style={{
              fontFamily: mono,
              fontSize: 13,
              letterSpacing: '.14em',
              color: ACCENT,
              marginTop: 4,
            }}
          >
            {hud.lod}
          </div>
          <div style={{ height: 1, background: 'rgba(255,255,255,.06)', margin: '9px 0' }} />
          <div style={{ fontFamily: mono, fontSize: 10, color: '#7c8696', letterSpacing: '.04em' }}>
            {hud.count}
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            background: 'rgba(10,13,20,.8)',
            border: '1px solid rgba(255,255,255,.08)',
            borderRadius: 11,
            overflow: 'hidden',
          }}
        >
          <button
            type="button"
            onClick={() => engineRef.current?.zoomIn()}
            style={zoomBtnStyle}
            aria-label="Zoom in"
          >
            +
          </button>
          <div
            style={{
              fontFamily: mono,
              fontSize: 10,
              color: '#7c8696',
              textAlign: 'center',
              padding: '5px 0',
              borderTop: '1px solid rgba(255,255,255,.06)',
              borderBottom: '1px solid rgba(255,255,255,.06)',
            }}
          >
            {hud.zoomPct}%
          </div>
          <button
            type="button"
            onClick={() => engineRef.current?.zoomOut()}
            style={zoomBtnStyle}
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            onClick={() => engineRef.current?.fit()}
            title="Fit"
            style={{
              fontFamily: mono,
              fontSize: 9,
              letterSpacing: '.12em',
              color: '#9aa3b2',
              background: 'none',
              border: 'none',
              borderTop: '1px solid rgba(255,255,255,.06)',
              padding: '8px 0',
              cursor: 'pointer',
            }}
          >
            FIT
          </button>
        </div>
      </div>

      {selected && (
        <DetailPanel
          selected={selected}
          onClose={() => engineRef.current?.deselect()}
          onFocus={(idx) => engineRef.current?.focusEntity(idx)}
        />
      )}
    </div>
  );
}

const zoomBtnStyle: CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#cdd4df',
  fontSize: 17,
  width: 40,
  height: 36,
  cursor: 'pointer',
  lineHeight: 1,
};

function DetailPanel({
  selected,
  onClose,
  onFocus,
}: {
  selected: SelectedEntity;
  onClose: () => void;
  onFocus: (idx: number) => void;
}) {
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        right: 0,
        height: '100%',
        width: 344,
        background: 'linear-gradient(180deg,rgba(14,18,28,.96),rgba(10,13,20,.96))',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderLeft: '1px solid rgba(255,255,255,.08)',
        boxShadow: '-24px 0 60px rgba(0,0,0,.5)',
        animation: 'kgPanelIn .26s cubic-bezier(.2,.7,.2,1)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          padding: '24px 24px 0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <span
            style={{
              width: 11,
              height: 11,
              borderRadius: '50%',
              background: selected.color,
              boxShadow: `0 0 12px ${selected.color}`,
            }}
          />
          <span
            style={{
              fontFamily: mono,
              fontSize: 10.5,
              letterSpacing: '.22em',
              color: selected.color,
              textTransform: 'uppercase',
            }}
          >
            {selected.type}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close details"
          style={{
            background: 'none',
            border: 'none',
            color: '#6d7686',
            fontSize: 22,
            lineHeight: 1,
            cursor: 'pointer',
            padding: '0 2px',
          }}
        >
          ×
        </button>
      </div>

      <div style={{ padding: '14px 24px 0' }}>
        <div style={{ fontSize: 26, fontWeight: 500, color: '#edf0f5', lineHeight: 1.18 }}>
          {selected.name}
        </div>
        <div
          style={{
            fontSize: 12,
            color: '#737c8c',
            marginTop: 8,
            fontFamily: mono,
            letterSpacing: '.05em',
          }}
        >
          in {selected.cluster}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10, padding: '18px 24px 0' }}>
        <StatCard label="LINKS" value={selected.degree} />
        <StatCard label="SOURCES" value={selected.sourceCount} />
      </div>

      <div style={{ padding: '18px 24px 0' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontFamily: mono,
            fontSize: 9,
            letterSpacing: '.16em',
            color: '#5b6473',
            marginBottom: 6,
          }}
        >
          <span>CONFIDENCE</span>
          <span style={{ color: '#9aa3b2' }}>{selected.confPct}</span>
        </div>
        <div
          style={{
            height: 5,
            borderRadius: 3,
            background: 'rgba(255,255,255,.06)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: selected.confPct,
              background: selected.color,
              borderRadius: 3,
            }}
          />
        </div>
      </div>

      {selected.hasAliases && (
        <div style={{ padding: '18px 24px 0' }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 9,
              letterSpacing: '.16em',
              color: '#5b6473',
              marginBottom: 8,
            }}
          >
            ALSO KNOWN AS
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {selected.aliases.map((a) => (
              <span
                key={a}
                style={{
                  fontSize: 12,
                  color: '#aeb6c4',
                  background: 'rgba(255,255,255,.04)',
                  border: '1px solid rgba(255,255,255,.06)',
                  borderRadius: 20,
                  padding: '3px 11px',
                }}
              >
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      {selected.hasProps && (
        <div style={{ padding: '18px 24px 0' }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 9,
              letterSpacing: '.16em',
              color: '#5b6473',
              marginBottom: 8,
            }}
          >
            PROPERTIES
          </div>
          {selected.propList.map((p) => (
            <div
              key={p.k}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 12,
                padding: '6px 0',
                borderBottom: '1px solid rgba(255,255,255,.045)',
                fontSize: 13,
              }}
            >
              <span style={{ color: '#737c8c' }}>{p.k}</span>
              <span style={{ color: '#cfd5df', textAlign: 'right' }}>{p.v}</span>
            </div>
          ))}
        </div>
      )}

      <div
        style={{
          padding: '20px 24px 8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span style={{ fontFamily: mono, fontSize: 9, letterSpacing: '.16em', color: '#5b6473' }}>
          CONNECTIONS
        </span>
        <span style={{ fontFamily: mono, fontSize: 9, color: '#4b5462' }}>{selected.degree}</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 16px 22px' }}>
        {selected.relations.map((r) => (
          <div
            key={r.idx}
            onClick={() => onFocus(r.idx)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 11,
              padding: '9px 8px',
              borderRadius: 9,
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                flex: 'none',
                background: r.color,
                boxShadow: `0 0 7px ${r.color}`,
              }}
            />
            <span
              style={{
                flex: 1,
                fontSize: 13.5,
                color: '#c8ceda',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {r.name}
            </span>
            <span
              style={{
                fontFamily: mono,
                fontSize: 9,
                letterSpacing: '.1em',
                color: '#5b6473',
                textTransform: 'uppercase',
              }}
            >
              {r.type}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div
      style={{
        flex: 1,
        background: 'rgba(255,255,255,.03)',
        border: '1px solid rgba(255,255,255,.05)',
        borderRadius: 10,
        padding: '11px 12px',
      }}
    >
      <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: '.16em', color: '#5b6473' }}>
        {label}
      </div>
      <div style={{ fontSize: 21, color: '#dfe4ec', marginTop: 3 }}>{value}</div>
    </div>
  );
}

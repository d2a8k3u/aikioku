import type { EntityType } from '@/types';

import { RING_RADII, Z0, Z1, ZLBL } from './graphConstants';
import type {
  GraphEngineHandle,
  GraphHud,
  GraphLayout,
  GraphLod,
  SelectedEntity,
} from './graphTypes';

interface Camera {
  x: number;
  y: number;
  zoom: number;
}

interface VisNode {
  i: number;
  sx: number;
  sy: number;
  r: number;
}

interface FrameCluster {
  ci: number;
  sx: number;
  sy: number;
  sr: number;
  expand: number;
}

export interface GraphEngineOptions {
  container: HTMLElement;
  canvas: HTMLCanvasElement;
  mini: HTMLCanvasElement;
  layout: GraphLayout;
  glowIntensity: number;
  showRelations: boolean;
  ringGuides: boolean;
  onSelect: (sel: SelectedEntity | null) => void;
  onHud: (hud: GraphHud) => void;
  isHidden: (type: EntityType) => boolean;
  initialSelected?: number;
}

/**
 * Imperative Canvas-2D renderer for the knowledge graph, ported from the design export. Holds all
 * per-frame mutable state (camera, hover, hit-test caches) outside React; pushes only the selected
 * entity and HUD readouts back to React through callbacks. Construct inside a client effect and pair
 * start()/destroy().
 */
export class GraphEngine implements GraphEngineHandle {
  private readonly o: GraphEngineOptions;
  private readonly layout: GraphLayout;

  private cam: Camera = { x: 0, y: 0, zoom: 0.12 };
  private camGoal: Camera | null = null;
  private mouse = { x: -9999, y: -9999, inside: false };
  private hover = -1;
  private selIdx = -1;
  private dragging = false;
  private moved = false;
  private dragStart = { x: 0, y: 0 };
  private camStart = { x: 0, y: 0 };
  private didFit = false;

  private w = 0;
  private h = 0;
  private dpr = 0;
  private mw = 0;
  private mh = 0;

  private visNodes: VisNode[] = [];
  private frameClusters: FrameCluster[] = [];

  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;
  private lastHud: GraphHud | null = null;

  constructor(options: GraphEngineOptions) {
    this.o = options;
    this.layout = options.layout;
  }

  // ── lifecycle ──────────────────────────────────────────
  start(): void {
    this.ensureSize();
    this.o.canvas.addEventListener('wheel', this.onWheel, { passive: false });
    this.o.canvas.addEventListener('pointerdown', this.onDown);
    window.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    this.o.mini.addEventListener('pointerdown', this.onMiniDown);
    if (typeof ResizeObserver !== 'undefined') {
      this.ro = new ResizeObserver(() => this.ensureSize());
      this.ro.observe(this.o.container);
    }
    if (this.o.initialSelected != null && this.layout.nodes[this.o.initialSelected]) {
      this.selectNode(this.o.initialSelected);
    }
    if (typeof requestAnimationFrame === 'function') {
      this.rafId = requestAnimationFrame(this.loop);
    }
  }

  destroy(): void {
    if (this.rafId != null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(this.rafId);
    }
    this.rafId = null;
    this.ro?.disconnect();
    this.ro = null;
    this.o.canvas.removeEventListener('wheel', this.onWheel);
    this.o.canvas.removeEventListener('pointerdown', this.onDown);
    window.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    this.o.mini.removeEventListener('pointerdown', this.onMiniDown);
  }

  // ── public controls (driven by the React HUD) ──────────
  zoomIn(): void {
    this.zoomBy(1.5);
  }

  zoomOut(): void {
    this.zoomBy(1 / 1.5);
  }

  fit(): void {
    this.fitView(true);
  }

  deselect(): void {
    this.selIdx = -1;
    this.o.onSelect(null);
  }

  focusEntity(idx: number): void {
    const n = this.layout.nodes[idx];
    if (!n) return;
    this.camGoal = { x: n.x, y: n.y, zoom: Math.max(this.cam.zoom, 1.0) };
    this.selectNode(idx);
  }

  // ── camera transforms ──────────────────────────────────
  private w2s(x: number, y: number): { x: number; y: number } {
    return {
      x: (x - this.cam.x) * this.cam.zoom + this.w / 2,
      y: (y - this.cam.y) * this.cam.zoom + this.h / 2,
    };
  }

  private s2w(x: number, y: number): { x: number; y: number } {
    return {
      x: this.cam.x + (x - this.w / 2) / this.cam.zoom,
      y: this.cam.y + (y - this.h / 2) / this.cam.zoom,
    };
  }

  private clampZoom(z: number): number {
    return Math.max(0.06, Math.min(2.6, z));
  }

  private zoomBy(f: number): void {
    this.camGoal = { x: this.cam.x, y: this.cam.y, zoom: this.clampZoom(this.cam.zoom * f) };
  }

  private fitView(animate: boolean): void {
    const b = this.layout.bounds;
    const ww = b.maxX - b.minX;
    const wh = b.maxY - b.minY;
    if (ww <= 0 || wh <= 0 || this.w <= 0 || this.h <= 0) return;
    const z = Math.min(this.w / ww, this.h / wh) * 0.92;
    const goal: Camera = { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2, zoom: z };
    if (animate) {
      this.camGoal = goal;
    } else {
      this.cam = { ...goal };
      this.camGoal = null;
    }
  }

  // ── input ──────────────────────────────────────────────
  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const rect = this.o.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const before = this.s2w(mx, my);
    const f = Math.exp(-e.deltaY * 0.0013);
    this.cam.zoom = this.clampZoom(this.cam.zoom * f);
    const after = this.s2w(mx, my);
    this.cam.x += before.x - after.x;
    this.cam.y += before.y - after.y;
    this.camGoal = null;
  };

  private onDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.moved = false;
    this.dragStart = { x: e.clientX, y: e.clientY };
    this.camStart = { x: this.cam.x, y: this.cam.y };
    this.camGoal = null;
    this.o.container.style.cursor = 'grabbing';
  };

  private onMove = (e: PointerEvent): void => {
    const rect = this.o.canvas.getBoundingClientRect();
    this.mouse = { x: e.clientX - rect.left, y: e.clientY - rect.top, inside: true };
    if (this.dragging) {
      const dx = e.clientX - this.dragStart.x;
      const dy = e.clientY - this.dragStart.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) this.moved = true;
      this.cam.x = this.camStart.x - dx / this.cam.zoom;
      this.cam.y = this.camStart.y - dy / this.cam.zoom;
    } else {
      this.hover = this.hitTest(this.mouse.x, this.mouse.y);
      this.o.container.style.cursor = this.hover >= 0 ? 'pointer' : 'grab';
    }
  };

  private onUp = (): void => {
    if (this.dragging && !this.moved) {
      const hit = this.hitTest(this.mouse.x, this.mouse.y);
      if (hit >= 0) {
        this.selectNode(hit);
      } else {
        const cl = this.hitCluster(this.mouse.x, this.mouse.y);
        if (cl >= 0) {
          const c = this.layout.clusters[cl];
          this.camGoal = { x: c.cx, y: c.cy, zoom: Math.max(0.62, this.cam.zoom) };
        } else if (this.selIdx >= 0) {
          this.deselect();
        }
      }
    }
    this.dragging = false;
    this.o.container.style.cursor = this.hover >= 0 ? 'pointer' : 'grab';
  };

  private onMiniDown = (e: PointerEvent): void => {
    const rect = this.o.mini.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const p = this.miniToWorld(mx, my);
    this.camGoal = { x: p.x, y: p.y, zoom: Math.max(this.cam.zoom, 0.45) };
  };

  // ── hit testing ────────────────────────────────────────
  private hitTest(mx: number, my: number): number {
    let best = -1;
    let bd = 18 * 18;
    for (const v of this.visNodes) {
      const dx = v.sx - mx;
      const dy = v.sy - my;
      const d = dx * dx + dy * dy;
      const rr = (v.r + 6) * (v.r + 6);
      if (d < rr && d < bd) {
        bd = d;
        best = v.i;
      }
    }
    return best;
  }

  private hitCluster(mx: number, my: number): number {
    let best = -1;
    let bd = Infinity;
    for (const c of this.frameClusters) {
      if (c.expand > 0.55) continue;
      const dx = c.sx - mx;
      const dy = c.sy - my;
      const d = Math.hypot(dx, dy);
      if (d < c.sr && d < bd) {
        bd = d;
        best = c.ci;
      }
    }
    return best;
  }

  private selectNode(idx: number): void {
    this.selIdx = idx;
    const { nodes, clusters, adj, COLOR } = this.layout;
    const n = nodes[idx];
    const relations = [...adj[idx]]
      .map((j) => ({
        idx: j,
        name: nodes[j].name,
        type: nodes[j].type,
        color: COLOR[nodes[j].type],
      }))
      .sort((a, b) => nodes[b.idx].deg - nodes[a.idx].deg);
    const propList = Object.entries(n.props).map(([k, v]) => ({ k, v: String(v) }));
    this.o.onSelect({
      idx,
      name: n.name,
      type: n.type,
      color: COLOR[n.type],
      cluster: clusters[n.cluster].name,
      degree: n.deg,
      sourceCount: n.sourceCount,
      confPct: `${Math.round(n.confidence * 100)}%`,
      aliases: n.aliases,
      hasAliases: n.aliases.length > 0,
      propList,
      hasProps: propList.length > 0,
      relations,
    });
  }

  // ── sizing ─────────────────────────────────────────────
  private ensureSize(): boolean {
    const cont = this.o.container;
    const c = this.o.canvas;
    const r = cont.getBoundingClientRect();
    const dpr = Math.min((typeof window !== 'undefined' && window.devicePixelRatio) || 1, 2);
    if (this.w !== r.width || this.h !== r.height || this.dpr !== dpr) {
      this.w = r.width;
      this.h = r.height;
      this.dpr = dpr;
      c.width = Math.round(r.width * dpr);
      c.height = Math.round(r.height * dpr);
      const m = this.o.mini;
      const mr = m.getBoundingClientRect();
      this.mw = mr.width;
      this.mh = mr.height;
      m.width = Math.round(mr.width * dpr);
      m.height = Math.round(mr.height * dpr);
      if (!this.didFit && r.width > 0) {
        this.fitView(false);
        this.didFit = true;
      }
    }
    return this.w > 0;
  }

  // ── draw loop ──────────────────────────────────────────
  private loop = (): void => {
    if (this.camGoal) {
      const g = this.camGoal;
      const s = 0.16;
      this.cam.x += (g.x - this.cam.x) * s;
      this.cam.y += (g.y - this.cam.y) * s;
      this.cam.zoom += (g.zoom - this.cam.zoom) * s;
      if (
        Math.abs(g.x - this.cam.x) < 1 &&
        Math.abs(g.y - this.cam.y) < 1 &&
        Math.abs(g.zoom - this.cam.zoom) < 0.0008
      ) {
        this.cam = { ...g };
        this.camGoal = null;
      }
    }
    try {
      this.draw();
    } catch {
      // Keep the loop alive across a single bad frame.
    }
    if (typeof requestAnimationFrame === 'function') {
      this.rafId = requestAnimationFrame(this.loop);
    }
  };

  private hexA(hex: string, a: number): string {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  private rr(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  private draw(): void {
    const c = this.o.canvas;
    if (!this.ensureSize()) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    const { nodes, edges, clusters, adj, COLOR } = this.layout;
    const glow = this.o.glowIntensity;
    const showRel = this.o.showRelations;
    const rings = this.o.ringGuides;
    const zoom = this.cam.zoom;

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);

    // ring guides
    if (rings && zoom < 0.5) {
      const c0 = this.w2s(0, 0);
      ctx.save();
      ctx.globalAlpha = Math.min(0.5, (0.5 - zoom) * 1.6);
      for (const rr of RING_RADII.slice(1)) {
        ctx.beginPath();
        ctx.arc(c0.x, c0.y, rr * zoom, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(120,140,170,.10)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();
    }

    const focus = this.hover >= 0 ? this.hover : this.selIdx;
    const fset = focus >= 0 ? adj[focus] : null;

    // per-cluster expansion
    this.frameClusters = [];
    const margin = 260;
    const cstate = clusters.map((cl) => {
      const s = this.w2s(cl.cx, cl.cy);
      const sr = cl.R * zoom;
      const onScreen =
        s.x > -sr - margin &&
        s.x < this.w + sr + margin &&
        s.y > -sr - margin &&
        s.y < this.h + sr + margin;
      let expand = (zoom - Z0) / (Z1 - Z0);
      expand = Math.max(0, Math.min(1, expand));
      if (!onScreen) expand = 0;
      this.frameClusters.push({ ci: cl.ci, sx: s.x, sy: s.y, sr: Math.max(sr, 16), expand });
      return { onScreen, expand, sx: s.x, sy: s.y, sr };
    });

    // bridge edges (faint, between clusters)
    if (showRel) {
      ctx.save();
      ctx.lineWidth = 1;
      for (const e of edges) {
        if (!e.bridge) continue;
        const a = nodes[e.a];
        const b = nodes[e.b];
        const sa = this.w2s(a.x, a.y);
        const sb = this.w2s(b.x, b.y);
        const hot = fset && (e.a === focus || e.b === focus);
        ctx.strokeStyle = hot ? this.hexA(COLOR[nodes[focus].type], 0.5) : 'rgba(130,150,180,.09)';
        ctx.beginPath();
        ctx.moveTo(sa.x, sa.y);
        ctx.lineTo(sb.x, sb.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    let drawn = 0;
    // intra-cluster edges (only expanded, on-screen)
    if (showRel && zoom > Z0) {
      ctx.save();
      ctx.lineWidth = 1;
      for (const e of edges) {
        if (e.bridge) continue;
        const a = nodes[e.a];
        const ex = cstate[a.cluster].expand;
        if (ex <= 0.02) continue;
        const b = nodes[e.b];
        if (this.o.isHidden(a.type) && this.o.isHidden(b.type)) continue;
        const sa = this.w2s(a.x, a.y);
        const sb = this.w2s(b.x, b.y);
        const hot = fset && (e.a === focus || e.b === focus);
        if (fset && !hot) ctx.strokeStyle = this.hexA('#8aa0bd', 0.05 * ex);
        else if (hot) ctx.strokeStyle = this.hexA(COLOR[nodes[focus].type], 0.55);
        else ctx.strokeStyle = this.hexA('#7d93b3', 0.13 * ex);
        ctx.beginPath();
        ctx.moveTo(sa.x, sa.y);
        ctx.lineTo(sb.x, sb.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // clusters: orbs (collapsed) + nodes (expanded)
    this.visNodes = [];
    const showLabels = zoom >= ZLBL;
    clusters.forEach((cl, ci) => {
      const st = cstate[ci];
      // orb
      if (st.expand < 1) {
        const a = 1 - st.expand;
        const R = Math.max(st.sr, 18);
        const g = ctx.createRadialGradient(st.sx, st.sy, 0, st.sx, st.sy, R);
        g.addColorStop(0, this.hexA(cl.accent, 0.42 * a * glow));
        g.addColorStop(0.5, this.hexA(cl.accent, 0.14 * a * glow));
        g.addColorStop(1, this.hexA(cl.accent, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(st.sx, st.sy, R, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = this.hexA(cl.accent, 0.85 * a);
        ctx.beginPath();
        ctx.arc(st.sx, st.sy, Math.max(4, R * 0.12), 0, Math.PI * 2);
        ctx.fill();
        if (a > 0.3 && R > 24) {
          ctx.globalAlpha = a;
          ctx.fillStyle = '#e7ebf2';
          ctx.font = '500 13px Spectral, serif';
          ctx.textAlign = 'center';
          ctx.fillText(cl.name, st.sx, st.sy - R - 8);
          ctx.fillStyle = '#6b7686';
          ctx.font = "10px 'JetBrains Mono', monospace";
          ctx.fillText(`${cl.count} entities`, st.sx, st.sy - R + 6);
          ctx.globalAlpha = 1;
          ctx.textAlign = 'left';
        }
      }
      // nodes
      if (st.expand > 0.02 && st.onScreen) {
        const start = cl.first;
        const end = cl.first + cl.count;
        for (let i = start; i < end; i++) {
          const n = nodes[i];
          if (this.o.isHidden(n.type)) continue;
          const s = this.w2s(n.x, n.y);
          if (s.x < -40 || s.x > this.w + 40 || s.y < -40 || s.y > this.h + 40) continue;
          const r = Math.max(1.6, n.r0 * zoom * (0.5 + 0.5 * st.expand));
          const col = COLOR[n.type];
          let a = st.expand;
          if (fset) {
            const inF = i === focus || fset.has(i);
            a *= inF ? 1 : 0.16;
          }
          this.visNodes.push({ i, sx: s.x, sy: s.y, r });
          if (r > 2.4) {
            const gr = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, r * 3.2);
            gr.addColorStop(0, this.hexA(col, 0.5 * a * glow));
            gr.addColorStop(1, this.hexA(col, 0));
            ctx.fillStyle = gr;
            ctx.beginPath();
            ctx.arc(s.x, s.y, r * 3.2, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.fillStyle = this.hexA(col, Math.min(1, a));
          ctx.beginPath();
          ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
          ctx.fill();
          if (i === focus) {
            ctx.strokeStyle = this.hexA(col, 0.9);
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(s.x, s.y, r + 4, 0, Math.PI * 2);
            ctx.stroke();
          }
          drawn++;
          if (a > 0.5 && (n.isHub ? showLabels : zoom >= 1.35)) {
            ctx.fillStyle = this.hexA('#cdd4df', Math.min(1, a));
            ctx.font = `${n.isHub ? '500 12px' : '11px'} Spectral, serif`;
            ctx.textAlign = 'left';
            ctx.fillText(n.name, s.x + r + 5, s.y + 4);
          }
        }
      }
    });

    // hovered tooltip
    if (this.hover >= 0 && this.mouse.inside) {
      const n = nodes[this.hover];
      const s = this.w2s(n.x, n.y);
      const txt = n.name;
      ctx.font = '500 13px Spectral, serif';
      const tw = ctx.measureText(txt).width;
      const sub = n.type.toUpperCase();
      ctx.font = "9px 'JetBrains Mono', monospace";
      const sw = ctx.measureText(sub).width;
      const bw = Math.max(tw, sw) + 22;
      const bh = 42;
      let bx = s.x + 14;
      let by = s.y - bh - 8;
      if (bx + bw > this.w - 8) bx = s.x - bw - 14;
      if (by < 8) by = s.y + 12;
      ctx.fillStyle = 'rgba(12,16,24,.94)';
      ctx.strokeStyle = 'rgba(255,255,255,.1)';
      ctx.lineWidth = 1;
      this.rr(ctx, bx, by, bw, bh, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = COLOR[n.type];
      ctx.beginPath();
      ctx.arc(bx + 12, by + 15, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#e7ebf2';
      ctx.font = '500 13px Spectral, serif';
      ctx.textAlign = 'left';
      ctx.fillText(txt, bx + 22, by + 19);
      ctx.fillStyle = this.hexA(COLOR[n.type], 0.9);
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText(sub, bx + 12, by + 33);
    }

    this.drawMini();
    this.emitHud(zoom, drawn);
  }

  private emitHud(zoom: number, drawn: number): void {
    const lod: GraphLod = zoom < Z0 ? 'CLUSTERS' : zoom < ZLBL ? 'ENTITIES' : 'DETAIL';
    const total = this.layout.nodes.length;
    const count =
      zoom < Z0
        ? `${this.layout.clusters.length} clusters · ${total} hidden`
        : `rendering ${drawn} / ${total} nodes`;
    const zoomPct = Math.round(zoom * 100);
    if (
      this.lastHud &&
      this.lastHud.zoomPct === zoomPct &&
      this.lastHud.lod === lod &&
      this.lastHud.count === count
    ) {
      return;
    }
    this.lastHud = { zoomPct, lod, count };
    this.o.onHud(this.lastHud);
  }

  // ── minimap ────────────────────────────────────────────
  private miniMap(): { s: number; ox: number; oy: number } {
    const b = this.layout.bounds;
    const ww = b.maxX - b.minX;
    const wh = b.maxY - b.minY;
    const pad = 8;
    const s = Math.min((this.mw - pad * 2) / ww, (this.mh - pad * 2) / wh);
    const ox = (this.mw - ww * s) / 2 - b.minX * s;
    const oy = (this.mh - wh * s) / 2 - b.minY * s;
    return { s, ox, oy };
  }

  private miniToWorld(mx: number, my: number): { x: number; y: number } {
    const m = this.miniMap();
    return { x: (mx - m.ox) / m.s, y: (my - m.oy) / m.s };
  }

  private drawMini(): void {
    const mc = this.o.mini;
    if (!this.mw) return;
    const ctx = mc.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.mw, this.mh);
    const m = this.miniMap();
    for (const cl of this.layout.clusters) {
      const x = cl.cx * m.s + m.ox;
      const y = cl.cy * m.s + m.oy;
      const r = Math.max(2, cl.R * m.s * 0.5);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
      g.addColorStop(0, this.hexA(cl.accent, 0.5));
      g.addColorStop(1, this.hexA(cl.accent, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r * 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = this.hexA(cl.accent, 0.9);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1.4, r * 0.5), 0, Math.PI * 2);
      ctx.fill();
    }
    const tl = this.s2w(0, 0);
    const br = this.s2w(this.w, this.h);
    const rx = tl.x * m.s + m.ox;
    const ry = tl.y * m.s + m.oy;
    const rw = (br.x - tl.x) * m.s;
    const rh = (br.y - tl.y) * m.s;
    ctx.strokeStyle = 'rgba(217,154,91,.85)';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = 'rgba(217,154,91,.08)';
    ctx.fillRect(rx, ry, rw, rh);
  }
}

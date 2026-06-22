import * as THREE from 'three';

import type { EntityType } from '@/types';

import {
  CAM_FOV,
  FADE_FAR,
  FADE_NEAR,
  LOD_FAR,
  LOD_NEAR,
  LOD_R_CLUSTERS,
  LOD_R_ENTITIES,
  ORB_SCALE,
  RADIUS_DEFAULT,
  RADIUS_MAX,
  RADIUS_MIN,
} from './graphConstants';
import type {
  GraphEngineHandle,
  GraphHud,
  GraphLayout3D,
  GraphLod,
  SelectedEntity,
} from './graphTypes';

export interface GraphEngine3DOptions {
  container: HTMLElement;
  canvas: HTMLCanvasElement;
  label: HTMLElement;
  layout: GraphLayout3D;
  onSelect: (sel: SelectedEntity | null) => void;
  onHud: (hud: GraphHud) => void;
  isHidden: (type: EntityType) => boolean;
  initialSelected?: number;
}

const FOV_RAD = (CAM_FOV * Math.PI) / 180;

/**
 * Imperative Three.js renderer for the knowledge graph, a faithful port of the design export's 3D scene:
 * additive-blend glow-particle nodes (custom shader), shader line edges, cluster-orb sprites, and a
 * spherical orbital camera. Mirrors GraphEngine's public surface (GraphEngineHandle) so the React HUD can
 * drive either engine, and pushes only the selected entity + HUD readouts back through callbacks.
 * Construct inside a client effect and pair start()/destroy().
 */
export class GraphEngine3D implements GraphEngineHandle {
  private readonly o: GraphEngine3DOptions;
  private readonly layout: GraphLayout3D;

  private ready = false;
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;

  private ctrl = { theta: 0.7, phi: 1.15, radius: RADIUS_DEFAULT, target: new THREE.Vector3() };
  private radiusGoal = RADIUS_DEFAULT;
  private targetGoal = new THREE.Vector3();

  private uScale = { value: 1 };
  private uTime = { value: 0 };

  private nodeAlpha!: Float32Array;
  private edgeAlpha!: Float32Array;
  private nodeGeo!: THREE.BufferGeometry;
  private edgeGeo!: THREE.BufferGeometry;
  private points!: THREE.Points;
  private lines!: THREE.LineSegments;
  private orbs: THREE.Sprite[] = [];
  private glowTex!: THREE.Texture;

  private clusterLabels: HTMLDivElement[] = [];
  private nodeLabel!: HTMLDivElement;

  private hover = -1;
  private selIdx = -1;
  private dragging = false;
  private moved = false;
  private px = 0;
  private py = 0;
  private mx = 0;
  private my = 0;

  private w = 0;
  private h = 0;

  private t0 = 0;
  private lastInteract = 0;
  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;
  private readonly proj = new THREE.Vector3();
  private lastHud: GraphHud | null = null;

  constructor(options: GraphEngine3DOptions) {
    this.o = options;
    this.layout = options.layout;
  }

  // ── lifecycle ──────────────────────────────────────────
  start(): void {
    const canvas = this.o.canvas;
    // jsdom / unsupported GPUs return null here; bail gracefully so the component still mounts.
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return;
    try {
      this.initThree();
    } catch {
      return;
    }
    this.ready = true;

    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('pointerdown', this.onDown);
    window.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    if (typeof ResizeObserver !== 'undefined') {
      this.ro = new ResizeObserver(() => this.resize());
      this.ro.observe(this.o.container);
    }
    if (this.o.initialSelected != null && this.layout.nodes[this.o.initialSelected]) {
      this.selectNode(this.o.initialSelected, false);
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
    if (!this.ready) return;
    for (const lbl of this.clusterLabels) lbl.remove();
    this.nodeLabel?.remove();
    this.clusterLabels = [];
    this.nodeGeo.dispose();
    this.edgeGeo.dispose();
    (this.points.material as THREE.Material).dispose();
    (this.lines.material as THREE.Material).dispose();
    for (const orb of this.orbs) orb.material.dispose();
    this.glowTex.dispose();
    this.renderer.dispose();
    this.ready = false;
  }

  // ── public controls (driven by the React HUD) ──────────
  zoomIn(): void {
    this.radiusGoal = Math.max(RADIUS_MIN, this.radiusGoal * 0.6);
    this.markInteract();
  }

  zoomOut(): void {
    this.radiusGoal = Math.min(RADIUS_MAX, this.radiusGoal * 1.6);
    this.markInteract();
  }

  fit(): void {
    this.radiusGoal = RADIUS_DEFAULT;
    this.targetGoal.set(0, 0, 0);
    this.markInteract();
  }

  deselect(): void {
    this.selIdx = -1;
    this.o.onSelect(null);
  }

  focusEntity(idx: number): void {
    this.selectNode(idx, true);
  }

  // ── init ───────────────────────────────────────────────
  private glowTexture(): THREE.Texture {
    const s = 128;
    const cv = document.createElement('canvas');
    cv.width = s;
    cv.height = s;
    const g = cv.getContext('2d');
    if (g) {
      const rg = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
      rg.addColorStop(0, 'rgba(255,255,255,1)');
      rg.addColorStop(0.25, 'rgba(255,255,255,0.85)');
      rg.addColorStop(0.55, 'rgba(255,255,255,0.25)');
      rg.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = rg;
      g.fillRect(0, 0, s, s);
    }
    return new THREE.CanvasTexture(cv);
  }

  private initThree(): void {
    const el = this.o.container;
    const w = el.clientWidth || 1;
    const h = el.clientHeight || 1;
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.o.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(
      Math.min((typeof window !== 'undefined' && window.devicePixelRatio) || 1, 2),
    );
    this.renderer.setSize(w, h, false);
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(CAM_FOV, w / h, 1, 20000);
    this.glowTex = this.glowTexture();

    const { nodes, edges, clusters, COLOR } = this.layout;

    // ── edges ──
    const E = edges.length;
    const ePos = new Float32Array(E * 2 * 3);
    const eCol = new Float32Array(E * 2 * 3);
    this.edgeAlpha = new Float32Array(E * 2);
    edges.forEach((e, i) => {
      const a = nodes[e.a].pos;
      const b = nodes[e.b].pos;
      ePos.set([a[0], a[1], a[2], b[0], b[1], b[2]], i * 6);
      const c = e.bridge ? [0.42, 0.5, 0.62] : [0.45, 0.55, 0.7];
      eCol.set([c[0], c[1], c[2], c[0], c[1], c[2]], i * 6);
    });
    this.edgeGeo = new THREE.BufferGeometry();
    this.edgeGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
    this.edgeGeo.setAttribute('lcolor', new THREE.BufferAttribute(eCol, 3));
    this.edgeGeo.setAttribute('lalpha', new THREE.BufferAttribute(this.edgeAlpha, 1));
    const lineMat = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: `attribute vec3 lcolor; attribute float lalpha; varying vec3 vC; varying float vA;
        void main(){ vC=lcolor; vA=lalpha; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
      fragmentShader: `varying vec3 vC; varying float vA;
        void main(){ if(vA<=0.001) discard; gl_FragColor=vec4(vC,vA); }`,
    });
    this.lines = new THREE.LineSegments(this.edgeGeo, lineMat);
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);

    // ── node points ──
    const n = nodes.length;
    const pPos = new Float32Array(n * 3);
    const pCol = new Float32Array(n * 3);
    const pSize = new Float32Array(n);
    const pPh = new Float32Array(n);
    this.nodeAlpha = new Float32Array(n);
    const tmp = new THREE.Color();
    nodes.forEach((nd, i) => {
      pPos.set(nd.pos, i * 3);
      tmp.set(COLOR[nd.type]);
      pCol.set([tmp.r, tmp.g, tmp.b], i * 3);
      pSize[i] = nd.size;
      pPh[i] = nd.phase;
    });
    this.nodeGeo = new THREE.BufferGeometry();
    this.nodeGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    this.nodeGeo.setAttribute('acolor', new THREE.BufferAttribute(pCol, 3));
    this.nodeGeo.setAttribute('asize', new THREE.BufferAttribute(pSize, 1));
    this.nodeGeo.setAttribute('aalpha', new THREE.BufferAttribute(this.nodeAlpha, 1));
    this.nodeGeo.setAttribute('aphase', new THREE.BufferAttribute(pPh, 1));
    this.uScale = { value: h / (2 * Math.tan(0.5 * FOV_RAD)) };
    this.uTime = { value: 0 };
    const pMat = new THREE.ShaderMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: { uScale: this.uScale, uTime: this.uTime, uTex: { value: this.glowTex } },
      vertexShader: `attribute vec3 acolor; attribute float asize; attribute float aalpha; attribute float aphase;
        uniform float uScale; uniform float uTime; varying vec3 vC; varying float vA;
        void main(){ vC=acolor; vec3 p=position;
          p.x+=sin(uTime*0.55+aphase)*7.0; p.y+=cos(uTime*0.47+aphase*1.3)*7.0; p.z+=sin(uTime*0.4+aphase*0.7)*7.0;
          vec4 mv=modelViewMatrix*vec4(p,1.0); float d=-mv.z;
          float fade=1.0-smoothstep(${FADE_NEAR.toFixed(1)},${FADE_FAR.toFixed(1)},d);
          vA=aalpha*fade;
          gl_PointSize=clamp(asize*uScale/max(d,1.0),1.0,150.0);
          gl_Position=projectionMatrix*mv; }`,
      fragmentShader: `uniform sampler2D uTex; varying vec3 vC; varying float vA;
        void main(){ if(vA<=0.003) discard; vec4 t=texture2D(uTex,gl_PointCoord); float a=t.a*vA; if(a<0.003) discard;
          vec3 col=mix(vC, vec3(1.0), pow(t.a,3.0)*0.5); gl_FragColor=vec4(col, a); }`,
    });
    this.points = new THREE.Points(this.nodeGeo, pMat);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // ── cluster orbs ──
    this.orbs = clusters.map((cl) => {
      const mat = new THREE.SpriteMaterial({
        map: this.glowTex,
        color: new THREE.Color(cl.accent),
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthTest: false,
        depthWrite: false,
      });
      const sp = new THREE.Sprite(mat);
      sp.position.set(cl.center[0], cl.center[1], cl.center[2]);
      const sc = cl.clR * ORB_SCALE;
      sp.scale.set(sc, sc, 1);
      this.scene.add(sp);
      return sp;
    });

    // ── html labels ──
    const layer = this.o.label;
    this.clusterLabels = clusters.map((cl) => {
      const d = this.makeLabel();
      (d.children[0] as HTMLElement).textContent = cl.name;
      (d.children[1] as HTMLElement).textContent = `${cl.count} entities`;
      layer.appendChild(d);
      return d;
    });
    this.nodeLabel = this.makeLabel();
    layer.appendChild(this.nodeLabel);

    this.t0 = typeof performance !== 'undefined' ? performance.now() : 0;
    this.lastInteract = this.t0;
    this.resize();
    this.renderFrame();
  }

  private makeLabel(): HTMLDivElement {
    const d = document.createElement('div');
    d.style.cssText =
      'position:absolute;transform:translate(-50%,-140%);pointer-events:none;text-align:center;will-change:transform,left,top;opacity:0;';
    const name = document.createElement('div');
    name.style.cssText =
      "font-family:'Spectral',serif;font-size:14px;font-weight:500;color:#eef1f6;text-shadow:0 1px 10px rgba(0,0,0,.8);white-space:nowrap;";
    const cap = document.createElement('div');
    cap.style.cssText =
      "font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:.12em;color:#8a93a4;margin-top:2px;";
    d.appendChild(name);
    d.appendChild(cap);
    return d;
  }

  private resize(): void {
    if (!this.renderer) return;
    const el = this.o.container;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (!w || !h) return;
    this.w = w;
    this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.uScale.value = h / (2 * Math.tan(0.5 * FOV_RAD));
  }

  // ── interaction ────────────────────────────────────────
  private markInteract(): void {
    this.lastInteract = typeof performance !== 'undefined' ? performance.now() : 0;
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.markInteract();
    const f = Math.exp(e.deltaY * 0.0011);
    this.radiusGoal = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, this.radiusGoal * f));
  };

  private onDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.moved = false;
    this.px = e.clientX;
    this.py = e.clientY;
    this.markInteract();
    this.o.container.style.cursor = 'grabbing';
  };

  private onMove = (e: PointerEvent): void => {
    const r = this.o.canvas.getBoundingClientRect();
    this.mx = e.clientX - r.left;
    this.my = e.clientY - r.top;
    if (this.dragging) {
      const dx = e.clientX - this.px;
      const dy = e.clientY - this.py;
      if (Math.abs(dx) + Math.abs(dy) > 4) this.moved = true;
      this.px = e.clientX;
      this.py = e.clientY;
      this.ctrl.theta -= dx * 0.005;
      this.ctrl.phi = Math.max(0.18, Math.min(Math.PI - 0.18, this.ctrl.phi - dy * 0.005));
      this.markInteract();
    } else {
      this.hover = this.pick(this.mx, this.my);
      this.o.container.style.cursor = this.hover >= 0 ? 'pointer' : 'grab';
    }
  };

  private onUp = (): void => {
    if (this.dragging && !this.moved) {
      const hit = this.pick(this.mx, this.my);
      if (hit >= 0) {
        this.selectNode(hit, true);
      } else {
        const cl = this.pickCluster(this.mx, this.my);
        if (cl >= 0) {
          this.flyToCluster(cl);
        } else if (this.selIdx >= 0) {
          this.deselect();
        }
      }
    }
    this.dragging = false;
    this.o.container.style.cursor = this.hover >= 0 ? 'pointer' : 'grab';
    this.markInteract();
  };

  // ── projection + hit testing ───────────────────────────
  private toScreen(p: readonly [number, number, number]): { x: number; y: number } | null {
    this.proj.set(p[0], p[1], p[2]).project(this.camera);
    if (this.proj.z > 1) return null;
    return { x: (this.proj.x * 0.5 + 0.5) * this.w, y: (-this.proj.y * 0.5 + 0.5) * this.h };
  }

  private pick(mx: number, my: number): number {
    if (!this.camera) return -1;
    let best = -1;
    let bd = 18 * 18;
    const nodes = this.layout.nodes;
    for (let i = 0; i < nodes.length; i++) {
      if (this.nodeAlpha[i] < 0.25) continue;
      if (this.o.isHidden(nodes[i].type)) continue;
      const s = this.toScreen(nodes[i].pos);
      if (!s) continue;
      const dx = s.x - mx;
      const dy = s.y - my;
      const dd = dx * dx + dy * dy;
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    return best;
  }

  private pickCluster(mx: number, my: number): number {
    let best = -1;
    let bd = 46 * 46;
    this.layout.clusters.forEach((cl, ci) => {
      if (this.orbs[ci].material.opacity < 0.2) return;
      const s = this.toScreen(cl.center);
      if (!s) return;
      const dx = s.x - mx;
      const dy = s.y - my;
      const dd = dx * dx + dy * dy;
      if (dd < bd) {
        bd = dd;
        best = ci;
      }
    });
    return best;
  }

  private selectNode(idx: number, fly: boolean): void {
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
    if (fly) {
      this.targetGoal.set(n.pos[0], n.pos[1], n.pos[2]);
      this.radiusGoal = Math.max(360, Math.min(this.radiusGoal, 520));
      this.lastInteract = (typeof performance !== 'undefined' ? performance.now() : 0) + 3000;
    }
  }

  private flyToCluster(ci: number): void {
    const c = this.layout.clusters[ci].center;
    this.targetGoal.set(c[0], c[1], c[2]);
    this.radiusGoal = 620;
    this.lastInteract = (typeof performance !== 'undefined' ? performance.now() : 0) + 3000;
  }

  // ── draw loop ──────────────────────────────────────────
  private loop = (): void => {
    try {
      this.renderFrame();
    } catch {
      // Keep the loop alive across a single bad frame.
    }
    if (typeof requestAnimationFrame === 'function') {
      this.rafId = requestAnimationFrame(this.loop);
    }
  };

  private renderFrame(): void {
    if (!this.ready) return;
    if (!this.w) {
      this.resize();
      if (!this.w) return;
    }
    const now = typeof performance !== 'undefined' ? performance.now() : 0;
    this.uTime.value = (now - this.t0) / 1000;

    this.ctrl.target.lerp(this.targetGoal, 0.1);
    this.ctrl.radius += (this.radiusGoal - this.ctrl.radius) * 0.1;
    if (now - this.lastInteract > 2500 && !this.dragging) this.ctrl.theta += 0.001;
    const t = this.ctrl.theta;
    const p = this.ctrl.phi;
    const R = this.ctrl.radius;
    const tg = this.ctrl.target;
    this.camera.position.set(
      tg.x + R * Math.sin(p) * Math.cos(t),
      tg.y + R * Math.cos(p),
      tg.z + R * Math.sin(p) * Math.sin(t),
    );
    this.camera.lookAt(tg);

    const { nodes, edges, clusters } = this.layout;
    const focus = this.hover >= 0 ? this.hover : this.selIdx;
    const fset = focus >= 0 ? this.layout.adj[focus] : null;
    const camp = this.camera.position;
    let drawn = 0;

    // cluster expansion by camera distance → drives node/edge alpha + orb opacity
    const expand = clusters.map((cl, ci) => {
      const dx = camp.x - cl.center[0];
      const dy = camp.y - cl.center[1];
      const dz = camp.z - cl.center[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      let ex = (LOD_FAR - dist) / (LOD_FAR - LOD_NEAR);
      ex = Math.max(0, Math.min(1, ex));
      this.orbs[ci].material.opacity = Math.pow(1 - ex, 0.8) * 0.95;
      return ex;
    });

    for (let i = 0; i < nodes.length; i++) {
      const nd = nodes[i];
      let a = expand[nd.cluster];
      if (this.o.isHidden(nd.type)) a = 0;
      else if (fset) a *= i === focus || fset.has(i) ? 1 : 0.1;
      this.nodeAlpha[i] = a;
      if (a > 0.06) drawn++;
    }
    this.nodeGeo.attributes.aalpha.needsUpdate = true;

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      let a = e.bridge ? 0.1 : expand[nodes[e.a].cluster] * 0.32;
      if (this.o.isHidden(nodes[e.a].type) && this.o.isHidden(nodes[e.b].type)) a = 0;
      if (fset) a = e.a === focus || e.b === focus ? Math.max(a, 0.6) : a * 0.18;
      this.edgeAlpha[i * 2] = a;
      this.edgeAlpha[i * 2 + 1] = a;
    }
    this.edgeGeo.attributes.lalpha.needsUpdate = true;

    this.updateLabels();
    this.renderer.render(this.scene, this.camera);
    this.emitHud(drawn);
  }

  private updateLabels(): void {
    const { clusters, nodes, COLOR } = this.layout;
    clusters.forEach((cl, ci) => {
      const lbl = this.clusterLabels[ci];
      const op = this.orbs[ci].material.opacity;
      const s = op > 0.12 ? this.toScreen(cl.center) : null;
      if (s) {
        lbl.style.left = `${s.x}px`;
        lbl.style.top = `${s.y - cl.clR * 0.06}px`;
        lbl.style.opacity = String(Math.min(1, op * 1.4));
      } else {
        lbl.style.opacity = '0';
      }
    });

    const hl =
      this.hover >= 0 ? this.hover : this.selIdx >= 0 && this.ctrl.radius < 900 ? this.selIdx : -1;
    if (hl >= 0 && this.nodeAlpha[hl] > 0.2) {
      const nd = nodes[hl];
      const s = this.toScreen(nd.pos);
      if (s) {
        const name = this.nodeLabel.children[0] as HTMLElement;
        const cap = this.nodeLabel.children[1] as HTMLElement;
        name.textContent = nd.name;
        cap.textContent = nd.type.toUpperCase();
        cap.style.color = COLOR[nd.type];
        this.nodeLabel.style.left = `${s.x}px`;
        this.nodeLabel.style.top = `${s.y}px`;
        this.nodeLabel.style.opacity = '1';
        return;
      }
    }
    this.nodeLabel.style.opacity = '0';
  }

  private emitHud(drawn: number): void {
    const R = this.ctrl.radius;
    const lod: GraphLod =
      R > LOD_R_CLUSTERS ? 'CLUSTERS' : R > LOD_R_ENTITIES ? 'ENTITIES' : 'DETAIL';
    const total = this.layout.nodes.length;
    const count =
      drawn > 0
        ? `rendering ${drawn} / ${total} nodes`
        : `${this.layout.clusters.length} clusters · ${total} hidden`;
    const span = RADIUS_MAX - RADIUS_MIN;
    const zoomPct = Math.max(0, Math.min(100, Math.round((1 - (R - RADIUS_MIN) / span) * 100)));
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
}

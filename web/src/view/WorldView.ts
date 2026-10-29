// Top-down three.js view of one EcoArena world: grass texture, refuges, instanced animals, selected animal's vision.
import * as THREE from "three";
import type { Frame, Selected } from "../protocol";

// The world is drawn in its own fixed palette (a field map), the same in light and dark page themes.
export const WORLD_COLORS = {
  soil: [52, 42, 29],
  sparse: [96, 92, 46],
  visible: [78, 122, 48],
  lush: [128, 178, 70],
  stone: [70, 86, 92],
  prey: "#f3e7c4",
  predator: "#ff6a3d",
  predatorBusy: "#9c5340",
  refugeRing: "#a9bfc3",
  select: "#ffffff",
  see: { grass: "#a6d46a", prey: "#f3e7c4", predator: "#ff6a3d", refuge: "#a9bfc3", none: "#ffffff" },
};

const PREY_LEN = 2.4;
const PRED_LEN = 3.4;

function triangle(len: number): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const w = len * 0.42;
  g.setAttribute("position", new THREE.Float32BufferAttribute([len * 0.6, 0, 0, -len * 0.4, w, 0, -len * 0.4, -w, 0], 3));
  return g;
}

function lerp3(a: number[], b: number[], t: number): number[] {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export class WorldView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.OrthographicCamera;
  L = 200;
  G = 100;
  grassTex: THREE.DataTexture | null = null;
  grassData: Uint8Array | null = null;
  refuge: Uint8Array | null = null;
  ground: THREE.Mesh | null = null;
  rings: THREE.LineSegments | null = null;
  prey: THREE.InstancedMesh;
  pred: THREE.InstancedMesh;
  fan: THREE.Mesh;
  ring: THREE.LineLoop;
  private colBusy = new THREE.Color(WORLD_COLORS.predatorBusy);
  private colPred = new THREE.Color(WORLD_COLORS.predator);
  private disposed = false;

  constructor(private host: HTMLElement, onPick: (x: number, y: number) => void) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.OrthographicCamera(0, this.L, this.L, 0, -10, 10);
    this.scene.background = new THREE.Color(`rgb(${WORLD_COLORS.soil.join(",")})`);

    this.prey = new THREE.InstancedMesh(triangle(PREY_LEN), new THREE.MeshBasicMaterial({ color: WORLD_COLORS.prey }), 4800);
    this.pred = new THREE.InstancedMesh(triangle(PRED_LEN), new THREE.MeshBasicMaterial({ color: 0xffffff }), 1200);
    this.pred.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(1200 * 3), 3);
    for (const m of [this.prey, this.pred]) {
      m.count = 0;
      m.frustumCulled = false;
      m.position.z = 1;
      this.scene.add(m);
    }
    this.pred.position.z = 1.1;

    this.fan = new THREE.Mesh(new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, depthTest: false }));
    this.fan.position.z = 2;
    this.fan.visible = false;
    this.scene.add(this.fan);
    const circle: number[] = [];
    for (let k = 0; k < 32; k++) circle.push(Math.cos((k / 32) * Math.PI * 2) * 3.2, Math.sin((k / 32) * Math.PI * 2) * 3.2, 0);
    const rg = new THREE.BufferGeometry();
    rg.setAttribute("position", new THREE.Float32BufferAttribute(circle, 3));
    this.ring = new THREE.LineLoop(rg, new THREE.LineBasicMaterial({ color: WORLD_COLORS.select }));
    this.ring.position.z = 3;
    this.ring.visible = false;
    this.scene.add(this.ring);

    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      const r = this.renderer.domElement.getBoundingClientRect();
      onPick(((e.clientX - r.left) / r.width) * this.L, (1 - (e.clientY - r.top) / r.height) * this.L);
    });
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
  }

  resize() {
    if (this.disposed) return;
    const s = Math.max(50, Math.floor(Math.min(this.host.clientWidth, this.host.clientHeight || this.host.clientWidth)));
    this.renderer.setSize(s, s);
    this.render();
  }

  private setWorld(L: number, G: number) {
    if (L === this.L && G === this.G && this.ground) return;
    this.L = L;
    this.G = G;
    this.camera.left = 0;
    this.camera.right = L;
    this.camera.top = L;
    this.camera.bottom = 0;
    this.camera.updateProjectionMatrix();
    if (this.ground) {
      this.scene.remove(this.ground);
      this.ground.geometry.dispose();
    }
    this.grassData = new Uint8Array(G * G * 4);
    this.grassTex = new THREE.DataTexture(this.grassData, G, G, THREE.RGBAFormat);
    this.grassTex.magFilter = THREE.LinearFilter;
    this.grassTex.minFilter = THREE.LinearFilter;
    this.grassTex.colorSpace = THREE.SRGBColorSpace;
    const geo = new THREE.PlaneGeometry(L, L);
    this.ground = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: this.grassTex }));
    this.ground.position.set(L / 2, L / 2, 0);
    this.scene.add(this.ground);
  }

  private setRefuges(centres: [number, number][], radius: number) {
    if (this.rings) {
      this.scene.remove(this.rings);
      this.rings.geometry.dispose();
    }
    const pts: number[] = [];
    const seg = 40;
    for (const [cx, cy] of centres) {
      // draw the ring at every wrapped copy that touches the map
      for (const ox of [-this.L, 0, this.L]) {
        for (const oy of [-this.L, 0, this.L]) {
          const x = cx + ox, y = cy + oy;
          if (x < -radius || x > this.L + radius || y < -radius || y > this.L + radius) continue;
          for (let k = 0; k < seg; k++) {
            const a0 = (k / seg) * Math.PI * 2, a1 = ((k + 1) / seg) * Math.PI * 2;
            pts.push(x + Math.cos(a0) * radius, y + Math.sin(a0) * radius, 0, x + Math.cos(a1) * radius, y + Math.sin(a1) * radius, 0);
          }
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.rings = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: WORLD_COLORS.refugeRing, transparent: true, opacity: 0.45 }));
    this.rings.position.z = 0.5;
    this.scene.add(this.rings);
  }

  private setGrass(grass: Float32Array) {
    const G = this.G;
    const d = this.grassData!;
    const C = WORLD_COLORS;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const g = grass[i * G + j];
        let c = g < 0.25 ? lerp3(C.soil, C.sparse, g / 0.25) : g < 0.5 ? lerp3(C.sparse, C.visible, (g - 0.25) / 0.25)
          : lerp3(C.visible, C.lush, (g - 0.5) / 0.5);
        if (this.refuge && this.refuge[i * G + j]) c = lerp3(c, C.stone, 0.55);
        const k = (j * G + i) * 4; // texture row = y index, column = x index
        d[k] = c[0];
        d[k + 1] = c[1];
        d[k + 2] = c[2];
        d[k + 3] = 255;
      }
    }
    this.grassTex!.needsUpdate = true;
  }

  update(f: Frame) {
    this.setWorld(f.world, f.grid);
    if (f.refuge) {
      this.refuge = f.refuge;
      this.setRefuges(f.refugeCentres ?? [], f.refugeRadius);
    }
    if (f.grass) this.setGrass(f.grass);
    const m = new THREE.Matrix4();
    const fill = (mesh: THREE.InstancedMesh, x: Float32Array, y: Float32Array, h: Float32Array) => {
      const n = Math.min(x.length, mesh.instanceMatrix.count);
      const arr = mesh.instanceMatrix.array as Float32Array;
      for (let i = 0; i < n; i++) {
        const c = Math.cos(h[i]), s = Math.sin(h[i]);
        m.set(c, -s, 0, x[i], s, c, 0, y[i], 0, 0, 1, 0, 0, 0, 0, 1);
        m.toArray(arr, i * 16);
      }
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
    };
    fill(this.prey, f.prey.x, f.prey.y, f.prey.h);
    fill(this.pred, f.pred.x, f.pred.y, f.pred.h);
    for (let i = 0; i < this.pred.count; i++) this.pred.setColorAt(i, f.pred.busy[i] ? this.colBusy : this.colPred);
    if (this.pred.instanceColor) this.pred.instanceColor.needsUpdate = true;
    this.setSelected(f.selected);
    this.render();
  }

  private setSelected(sel: Selected | null) {
    if (!sel || !sel.alive) {
      this.fan.visible = false;
      this.ring.visible = sel !== null && !sel.alive;
      if (sel) this.ring.position.set(sel.x, sel.y, 3);
      return;
    }
    this.ring.visible = true;
    this.ring.position.set(sel.x, sel.y, 3);
    // vision fan: one wedge per sector, as long as what that sector reports, coloured by what it sees
    const R = sel.obs.length > 6 ? (sel.obs.length - 6) / 6 : 16;
    const cone = (sel.coneDeg * Math.PI) / 180;
    const pos: number[] = [];
    const col: number[] = [];
    const cols = WORLD_COLORS.see;
    const kinds: (keyof typeof cols)[] = ["grass", "prey", "predator", "refuge", "none"];
    for (let r = 0; r < R; r++) {
      const o = r * 6;
      const k = kinds[[1, 2, 3, 4, 5].findIndex((c) => sel.obs[o + c] > 0.5)] ?? "none";
      const len = Math.max(0.6, sel.obs[o] * sel.range);
      const a0 = sel.heading - cone / 2 + (r / R) * cone;
      const a1 = sel.heading - cone / 2 + ((r + 1) / R) * cone;
      const c = new THREE.Color(cols[k]);
      const alpha = k === "none" ? 0.25 : 1;
      pos.push(sel.x, sel.y, 0, sel.x + Math.cos(a0) * len, sel.y + Math.sin(a0) * len, 0, sel.x + Math.cos(a1) * len, sel.y + Math.sin(a1) * len, 0);
      for (let v = 0; v < 3; v++) col.push(c.r * alpha, c.g * alpha, c.b * alpha);
    }
    const g = this.fan.geometry;
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.computeBoundingSphere();
    this.fan.visible = true;
  }

  render() {
    if (!this.disposed) this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

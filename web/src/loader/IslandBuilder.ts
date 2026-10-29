// The loading screen's 3D piece: a tiny low-poly island that assembles as loading progresses. Rock strata
// rise from below, the meadow cap settles on top, grass sprouts, rocks and trees pop in, then a hare and a
// wolf start circling. pushThrough() dollies the camera into the island for the hand-over.
import * as THREE from "three";
import { animateMaterial, predatorGeometry, preyGeometry } from "../view/creatures";

type Piece = { obj: THREE.Object3D; at: number; kind: "rise" | "pop" | "sprout"; base: THREE.Vector3; scale: THREE.Vector3 };

const easeOutBack = (t: number) => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2);
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));

function rnd(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class IslandBuilder {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  private group = new THREE.Group();
  private pieces: Piece[] = [];
  private target = 0;
  private shown = 0;
  private clock = new THREE.Clock();
  private raf = 0;
  private time = { uTime: { value: 0 } };
  private hare: THREE.InstancedMesh;
  private wolf: THREE.InstancedMesh;
  private hareAnim: THREE.InstancedBufferAttribute;
  private wolfAnim: THREE.InstancedBufferAttribute;
  private motes: THREE.Points;
  private push: { t0: number; done: () => void } | null = null;
  private disposed = false;
  private readonly home = new THREE.Vector3(0, 4.6, 11.5);

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(this.renderer.domElement);
    this.camera.position.copy(this.home);
    this.camera.lookAt(0, 0.2, 0);

    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x3a2a1c, 1.5));
    const sun = new THREE.DirectionalLight(0xffd9a8, 2.6);
    sun.position.set(5, 8, 4);
    this.scene.add(sun);
    const rim = new THREE.DirectionalLight(0x8fb6ff, 1.2);
    rim.position.set(-6, 3, -5);
    this.scene.add(rim);
    this.scene.add(this.group);

    const r = rnd(11);
    const flat = (color: string, rough = 0.95) => new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: rough });
    const add = (obj: THREE.Object3D, at: number, kind: Piece["kind"]) => {
      this.group.add(obj);
      this.pieces.push({ obj, at, kind, base: obj.position.clone(), scale: obj.scale.clone() });
    };

    // strata: four hexagonal slices, the deepest rises first
    const strata: [number, number, number, string][] = [
      [1.2, -1.75, 0.5, "#3f3935"], [2.0, -1.25, 0.5, "#5a524c"], [2.6, -0.75, 0.5, "#8a5a36"], [2.95, -0.28, 0.45, "#5b4129"],
    ];
    strata.forEach(([rad, y, h, c], i) => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(rad + 0.25, rad, h, 7, 1), flat(c));
      m.position.y = y;
      m.rotation.y = i * 0.35;
      add(m, 0.02 + i * 0.05, "rise");
    });
    // a pointed rocky underside
    const under = new THREE.Mesh(new THREE.ConeGeometry(1.25, 1.6, 7), flat("#2f2b28"));
    under.rotation.x = Math.PI;
    under.position.y = -2.75;
    add(under, 0.0, "rise");

    // meadow cap
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(3.05, 3.05, 0.22, 7, 1), flat("#4f8a35", 0.9));
    cap.position.y = -0.02;
    cap.rotation.y = 3 * 0.35;
    add(cap, 0.24, "rise");

    // grass tufts
    const grassMat = flat("#5a9a3a", 0.9);
    const grassMat2 = flat("#78b04a", 0.9);
    for (let i = 0; i < 70; i++) {
      const a = r() * Math.PI * 2, d = Math.sqrt(r()) * 2.6;
      const m = new THREE.Mesh(new THREE.ConeGeometry(0.06 + r() * 0.05, 0.25 + r() * 0.3, 4), r() > 0.5 ? grassMat : grassMat2);
      m.position.set(Math.cos(a) * d, 0.15, Math.sin(a) * d);
      m.rotation.set((r() - 0.5) * 0.4, r() * 3, (r() - 0.5) * 0.4);
      add(m, 0.32 + r() * 0.16, "sprout");
    }
    // a refuge: rock ring with a bush
    const rockMat = flat("#9aa3a8", 0.95);
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2;
      const m = new THREE.Mesh(new THREE.DodecahedronGeometry(0.22 + r() * 0.16, 0), rockMat);
      m.position.set(1.15 + Math.cos(a) * 0.62, 0.18, -0.6 + Math.sin(a) * 0.62);
      m.rotation.set(r() * 3, r() * 3, r() * 3);
      add(m, 0.52 + i * 0.015, "pop");
    }
    const bush = new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 1), flat("#3e7a33", 0.9));
    bush.position.set(1.15, 0.28, -0.6);
    add(bush, 0.62, "pop");

    // trees
    const trunk = flat("#5b3d27"), crownA = flat("#2f6b34", 0.9), crownB = flat("#3c7d3b", 0.9);
    ([[-1.6, -1.0, 1.0], [-1.9, 0.6, 0.8], [-0.4, -2.1, 0.9], [2.1, 1.0, 0.75]] as const).forEach(([x, z, s], i) => {
      const t = new THREE.Group();
      const tr = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.5, 6), trunk);
      tr.position.y = 0.25;
      t.add(tr);
      for (let k = 0; k < 3; k++) {
        const c = new THREE.Mesh(new THREE.ConeGeometry(0.42 - k * 0.1, 0.5, 7), k % 2 ? crownB : crownA);
        c.position.y = 0.6 + k * 0.28;
        t.add(c);
      }
      t.position.set(x, 0.08, z);
      t.scale.setScalar(s);
      add(t, 0.66 + i * 0.05, "pop");
    });

    // the animals, with the same shader animation as the real world
    const mk = (geo: THREE.BufferGeometry, kind: "prey" | "predator") => {
      const mat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.85 });
      animateMaterial(mat, kind, this.time);
      const anim = new THREE.InstancedBufferAttribute(new Float32Array(4), 4);
      geo.setAttribute("aAnim", anim);
      const m = new THREE.InstancedMesh(geo, mat, 1);
      m.frustumCulled = false;
      this.group.add(m);
      return { m, anim };
    };
    const h = mk(preyGeometry(), "prey"), w = mk(predatorGeometry(), "predator");
    this.hare = h.m;
    this.hareAnim = h.anim;
    this.wolf = w.m;
    this.wolfAnim = w.anim;

    // drifting motes of light around the island
    const n = 90, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const a = r() * Math.PI * 2, d = 2.5 + r() * 3;
      pos.set([Math.cos(a) * d, -2 + r() * 5, Math.sin(a) * d], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.motes = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffe2a0, size: 0.06, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.scene.add(this.motes);

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.loop();
  }

  /** Overall progress 0..1; the island catches up smoothly. */
  setProgress(p: number) {
    this.target = clamp01(p);
  }

  /** Dolly the camera into the island, then call done. */
  pushThrough(done: () => void) {
    this.push = { t0: performance.now(), done };
  }

  private resize() {
    if (this.disposed) return;
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private placeAnimal(m: THREE.InstancedMesh, anim: THREE.InstancedBufferAttribute, ang: number, rad: number, scale: number, gait: number, spd: number) {
    const x = Math.cos(ang) * rad, z = Math.sin(ang) * rad;
    // facing along the circle (counter-clockwise when seen from above)
    const heading = ang + Math.PI / 2;
    const mm = new THREE.Matrix4().compose(new THREE.Vector3(x, 0.1, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -heading, 0)), new THREE.Vector3(scale, scale, scale));
    m.setMatrixAt(0, mm);
    m.instanceMatrix.needsUpdate = true;
    anim.setXYZW(0, gait, spd, 0, 0.3);
    anim.needsUpdate = true;
  }

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(0.05, this.clock.getDelta());
    const t = (this.time.uTime.value += dt);
    this.shown += (this.target - this.shown) * Math.min(1, dt * 4.5);
    const p = this.shown;

    for (const pc of this.pieces) {
      const u = clamp01((p - pc.at) / 0.14);
      if (pc.kind === "rise") {
        pc.obj.position.y = pc.base.y - (1 - easeOut(u)) * 4;
        pc.obj.visible = u > 0;
        (pc.obj as THREE.Mesh).scale.copy(pc.scale);
      } else {
        const s = u <= 0 ? 0 : easeOutBack(u);
        pc.obj.visible = u > 0;
        pc.obj.scale.copy(pc.scale).multiplyScalar(Math.max(0.0001, s));
      }
    }
    // the animals arrive last and run laps: the hare a little ahead of the wolf
    const life = clamp01((p - 0.82) / 0.12);
    const k = 0.18 * easeOutBack(life);
    const lap = t * 0.55;
    this.placeAnimal(this.hare, this.hareAnim, lap, 1.75, Math.max(0.0001, k), t * 9, 0.9);
    this.placeAnimal(this.wolf, this.wolfAnim, lap - 0.9, 1.75, Math.max(0.0001, k * 0.85), t * 7.5, 0.9);

    // gentle turn and float
    this.group.rotation.y = t * 0.25;
    this.group.position.y = Math.sin(t * 1.2) * 0.08;
    this.motes.rotation.y = -t * 0.05;

    if (this.push) {
      const pu = Math.min(1, (performance.now() - this.push.t0) / 950);
      const e = pu * pu * pu;
      this.camera.position.lerpVectors(this.home, new THREE.Vector3(0, 0.9, 1.2), e);
      this.camera.fov = 32 + 40 * e;
      this.camera.updateProjectionMatrix();
      if (pu >= 1) {
        const d = this.push.done;
        this.push = null;
        d();
      }
    }
    this.camera.lookAt(0, 0.2, 0);
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

// EcoArena as a floating 3D diorama: a meadow island with swaying grass, rocky refuges, animated low-poly
// animals, a day/night sky, fireflies, clouds and event effects. Purely visual: positions, headings, grass and
// events all come from the simulation frames, and nothing here changes the rules.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { Frame, Herd, Selected } from "../protocol";
import { animateMaterial, predatorGeometry, preyGeometry } from "./creatures";

export type CameraMode = "orbit" | "top" | "follow" | "tour";
export type Overlay = "none" | "trails" | "catches";
export type SkyMode = "cycle" | "day" | "golden" | "night";
export type Quality = "high" | "low";

export const DAY_STEPS = 1200; // one visual day per 1,200 simulation steps (about one animal lifetime)

// ------------------------------------------------------------------------------------------ terrain
/** Gentle rolling hills, periodic over the world so the torus wraps without a seam. */
export function makeHeight(L: number) {
  const a = (2 * Math.PI) / L;
  return (x: number, y: number) =>
    1.25 * Math.sin(a * 2 * x + 0.7) * Math.cos(a * y + 0.3) +
    0.85 * Math.sin(a * 3 * y + 1.3) * Math.cos(a * 2 * x + 0.4) +
    0.35 * Math.sin(a * 5 * (x + y) + 2.1) +
    0.2 * Math.cos(a * 7 * x - a * 4 * y);
}

function mulberry(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const C = (hex: string) => new THREE.Color(hex);

// sky keyframes by sun elevation (-1 midnight .. 1 noon)
const SKY = [
  { e: -1.0, top: C("#030712"), hor: C("#0e1a33"), sun: C("#000000"), sunI: 0, hemi: 0.18, moon: 0.45, exp: 1.05 },
  { e: -0.12, top: C("#0b1631"), hor: C("#2a3352"), sun: C("#ff8a4c"), sunI: 0, hemi: 0.25, moon: 0.35, exp: 1.0 },
  { e: 0.06, top: C("#30487e"), hor: C("#f29a62"), sun: C("#ff9a52"), sunI: 1.6, hemi: 0.45, moon: 0.0, exp: 0.95 },
  { e: 0.3, top: C("#3f7ccc"), hor: C("#cfe3ef"), sun: C("#ffe2b8"), sunI: 2.6, hemi: 0.8, moon: 0.0, exp: 0.9 },
  { e: 1.0, top: C("#3a7bd5"), hor: C("#d7ecf7"), sun: C("#fff6e6"), sunI: 3.0, hemi: 0.95, moon: 0.0, exp: 0.85 },
];

function skyAt(e: number) {
  let i = 0;
  while (i < SKY.length - 2 && e > SKY[i + 1].e) i++;
  const a = SKY[i], b = SKY[i + 1];
  const t = THREE.MathUtils.clamp((e - a.e) / (b.e - a.e), 0, 1);
  return {
    top: a.top.clone().lerp(b.top, t), hor: a.hor.clone().lerp(b.hor, t), sun: a.sun.clone().lerp(b.sun, t),
    sunI: THREE.MathUtils.lerp(a.sunI, b.sunI, t), hemi: THREE.MathUtils.lerp(a.hemi, b.hemi, t),
    moon: THREE.MathUtils.lerp(a.moon, b.moon, t), exp: THREE.MathUtils.lerp(a.exp, b.exp, t),
  };
}

// ------------------------------------------------------------------------------------------ per-animal state
interface Track {
  cap: number;
  px: Float32Array; py: Float32Array; ph: Float32Array; // previous frame
  cx: Float32Array; cy: Float32Array; ch: Float32Array; // current frame
  spd: Float32Array; spdS: Float32Array; // speed, smoothed
  gait: Float32Array;
  pose: Float32Array; poseT: Float32Array; // smoothed pose and its target
  energy: Float32Array;
  alive: Uint8Array;
  born: Float32Array; // time of birth (for the pop-in)
  dying: Float32Array; // time of death, 0 = not dying
  seed: Float32Array;
  mesh: THREE.InstancedMesh;
  anim: THREE.InstancedBufferAttribute;
  base: number; // model scale
}

function makeTrack(cap: number, geo: THREE.BufferGeometry, mat: THREE.Material, base: number): Track {
  const mesh = new THREE.InstancedMesh(geo, mat, cap);
  mesh.castShadow = true;
  mesh.frustumCulled = false;
  const anim = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
  anim.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("aAnim", anim);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const seed = new Float32Array(cap);
  for (let i = 0; i < cap; i++) seed[i] = ((Math.imul(i + 1, 2654435761) >>> 0) % 10000) / 10000;
  const colors = new Float32Array(cap * 3).fill(1);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  const f = () => new Float32Array(cap);
  return {
    cap, px: f(), py: f(), ph: f(), cx: f(), cy: f(), ch: f(), spd: f(), spdS: f(), gait: f(), pose: f(), poseT: f(),
    energy: f(), alive: new Uint8Array(cap), born: f(), dying: f(), seed, mesh, anim, base,
  };
}

// ------------------------------------------------------------------------------------------ particles
const MAX_P = 5000;

class Particles {
  points: THREE.Points;
  pos = new Float32Array(MAX_P * 3);
  vel = new Float32Array(MAX_P * 3);
  col = new Float32Array(MAX_P * 3);
  life = new Float32Array(MAX_P); // remaining seconds
  maxLife = new Float32Array(MAX_P);
  size = new Float32Array(MAX_P);
  alpha = new Float32Array(MAX_P);
  grav = new Float32Array(MAX_P);
  next = 0;

  constructor(scene: THREE.Scene) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("color", new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1).setUsage(THREE.DynamicDrawUsage));
    const m = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      uniforms: { uScale: { value: 400 } },
      vertexShader: `attribute float aSize; attribute float aAlpha; varying vec3 vC; varying float vA; uniform float uScale;
        void main() { vC = color; vA = aAlpha; vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * uScale / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying vec3 vC; varying float vA;
        void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d); if (r > 0.25) discard;
          float a = vA * smoothstep(0.25, 0.0, r); gl_FragColor = vec4(vC * a, a); }`,
    });
    this.points = new THREE.Points(g, m);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }

  emit(x: number, y: number, z: number, n: number, o: { color: THREE.Color; speed: number; up: number; life: number; size: number; grav: number; spread?: number }) {
    for (let k = 0; k < n; k++) {
      const i = this.next;
      this.next = (this.next + 1) % MAX_P;
      const a = Math.random() * Math.PI * 2;
      const s = o.speed * (0.4 + Math.random() * 0.8);
      const sp = o.spread ?? 0.4;
      this.pos.set([x + (Math.random() - 0.5) * sp, y + Math.random() * 0.4, z + (Math.random() - 0.5) * sp], i * 3);
      this.vel.set([Math.cos(a) * s, o.up * (0.5 + Math.random()), Math.sin(a) * s], i * 3);
      const j = 0.85 + Math.random() * 0.3;
      this.col.set([o.color.r * j, o.color.g * j, o.color.b * j], i * 3);
      this.life[i] = this.maxLife[i] = o.life * (0.7 + Math.random() * 0.6);
      this.size[i] = o.size * (0.6 + Math.random() * 0.8);
      this.grav[i] = o.grav;
    }
  }

  update(dt: number) {
    for (let i = 0; i < MAX_P; i++) {
      if (this.life[i] <= 0) {
        if (this.alpha[i] !== 0) this.alpha[i] = 0;
        continue;
      }
      this.life[i] -= dt;
      const k = i * 3;
      this.vel[k + 1] -= this.grav[i] * dt;
      this.vel[k] *= 1 - 1.8 * dt;
      this.vel[k + 2] *= 1 - 1.8 * dt;
      this.pos[k] += this.vel[k] * dt;
      this.pos[k + 1] += this.vel[k + 1] * dt;
      this.pos[k + 2] += this.vel[k + 2] * dt;
      const f = Math.max(0, this.life[i] / this.maxLife[i]);
      this.alpha[i] = Math.min(1, f * 1.6) * f;
    }
    const g = this.points.geometry;
    g.attributes.position.needsUpdate = true;
    g.attributes.aAlpha.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.attributes.aSize.needsUpdate = true;
  }
}

// ------------------------------------------------------------------------------------------ the diorama
export class Diorama {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  composer: EffectComposer | null = null;
  bloom: UnrealBloomPass | null = null;
  quality: Quality;
  cameraMode: CameraMode = "orbit";
  skyMode: SkyMode = "cycle";

  L = 0;
  G = 0;
  height: (x: number, y: number) => number = () => 0;
  private time = { uTime: { value: 0 } };
  private island = new THREE.Group();
  private groundTex: THREE.DataTexture | null = null;
  private groundData: Uint8Array | null = null;
  private grass: THREE.InstancedMesh | null = null;
  private grassAmount: THREE.InstancedBufferAttribute | null = null;
  private grassCell: Int32Array | null = null;
  private refuge: Uint8Array | null = null;
  private props = new THREE.Group();
  private prey: Track | null = null;
  private pred: Track | null = null;
  private preyMat: THREE.MeshStandardMaterial;
  private predMat: THREE.MeshStandardMaterial;
  private particles: Particles;
  private rings: { mesh: THREE.Mesh; t: number }[] = [];
  private sun: THREE.DirectionalLight;
  private moon: THREE.DirectionalLight;
  private hemi: THREE.HemisphereLight;
  private skyMat: THREE.ShaderMaterial;
  private stars: THREE.Points;
  private fireflies: THREE.Points;
  private clouds = new THREE.Group();
  private selRing: THREE.Mesh;
  private fan: THREE.Mesh;
  private selected: Selected | null = null;
  private selSlot = -1;
  private selSpecies = 0;
  private lastFrameAt = 0;
  private frameInterval = 1 / 30;
  private simT = 0;
  private dayPhase = 0.42;
  private shownPhase = 0.42;
  private clock = new THREE.Clock();
  private raf = 0;
  private camTween: { from: THREE.Vector3; to: THREE.Vector3; tFrom: THREE.Vector3; tTo: THREE.Vector3; t: number; dur: number } | null = null;
  private userTouched = false;
  private frameTimes: number[] = [];
  onSlowDevice: (() => void) | null = null;
  /** Called when the camera leaves a mode on its own (the viewer dragged during a tour). */
  onCameraMode: ((m: CameraMode) => void) | null = null;
  /** The tour asks for a specific animal to be selected. */
  onTourPick: ((species: number, slot: number) => void) | null = null;
  private tourPickedAt = 0;
  private disposed = false;
  private onPickCb: (x: number, y: number) => void;
  // overlays painted onto the terrain: fading movement trails, or where catches happen
  overlay: Overlay = "none";
  private ovCanvas = document.createElement("canvas");
  private ovCtx: CanvasRenderingContext2D;
  private ovTex: THREE.CanvasTexture;
  private ovMesh: THREE.Mesh | null = null;
  private heat = new Float32Array(128 * 128);
  private heatMax = 0;
  private heatDirty = false;
  private heatDrawnAt = 0;
  private trailAt = 0;
  private tourNext = 0;

  constructor(private host: HTMLElement, onPick: (x: number, y: number) => void, quality: Quality) {
    this.quality = quality;
    this.onPickCb = onPick;
    this.ovCanvas.width = this.ovCanvas.height = 1024;
    this.ovCtx = this.ovCanvas.getContext("2d")!;
    this.ovTex = new THREE.CanvasTexture(this.ovCanvas);
    this.ovTex.colorSpace = THREE.SRGBColorSpace;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = quality === "high";
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(42, 1, 1, 4000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxPolarAngle = 1.38;
    this.controls.minDistance = 12;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener("start", () => {
      this.userTouched = true;
      this.controls.autoRotate = false;
      this.camTween = null;
      if (this.cameraMode === "tour") {
        this.cameraMode = "orbit";
        this.onCameraMode?.("orbit");
      }
    });

    // sky dome
    this.skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: { uTop: { value: C("#3a7bd5") }, uHor: { value: C("#d7ecf7") }, uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSun: { value: C("#fff") } },
      vertexShader: `varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vDir; uniform vec3 uTop; uniform vec3 uHor; uniform vec3 uSunDir; uniform vec3 uSun;
        void main() { float h = clamp(vDir.y, -1.0, 1.0);
          vec3 c = mix(uHor, uTop, pow(max(h, 0.0), 0.55));
          c = mix(c, uHor * 0.55, smoothstep(0.0, -0.5, h));
          float s = max(dot(vDir, normalize(uSunDir)), 0.0);
          c += uSun * (pow(s, 900.0) * 6.0 + pow(s, 12.0) * 0.25);
          gl_FragColor = vec4(c, 1.0); }`,
    });
    const sky = new THREE.Mesh(new THREE.SphereGeometry(1800, 32, 16), this.skyMat);
    this.scene.add(sky);

    // stars
    const sp: number[] = [];
    const rnd = mulberry(7);
    for (let i = 0; i < 1600; i++) {
      const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      if (u < 0.05) continue;
      sp.push(Math.cos(th) * r * 1700, u * 1700, Math.sin(th) * r * 1700);
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
    this.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xdfe8ff, size: 2.2, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false }));
    this.scene.add(this.stars);

    // lights
    this.hemi = new THREE.HemisphereLight(0xcfe6ff, 0x3a2c1c, 0.9);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff3dd, 3);
    this.sun.castShadow = quality === "high";
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0006;
    this.sun.shadow.normalBias = 0.4;
    this.scene.add(this.sun, this.sun.target);
    this.moon = new THREE.DirectionalLight(0x9db4ff, 0);
    this.scene.add(this.moon);
    this.scene.fog = new THREE.Fog(0xd7ecf7, 400, 1500);

    // animals
    this.preyMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9 });
    this.predMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.75 });
    animateMaterial(this.preyMat, "prey", this.time);
    animateMaterial(this.predMat, "predator", this.time);

    this.particles = new Particles(this.scene);

    // fireflies
    const ff = 420;
    const fp = new Float32Array(ff * 3);
    const fs = new Float32Array(ff);
    for (let i = 0; i < ff; i++) fs[i] = rnd();
    const fg = new THREE.BufferGeometry();
    fg.setAttribute("position", new THREE.BufferAttribute(fp, 3));
    fg.setAttribute("aSeed", new THREE.BufferAttribute(fs, 1));
    this.fireflies = new THREE.Points(fg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: this.time.uTime, uNight: { value: 0 }, uL: { value: 200 } },
      vertexShader: `attribute float aSeed; uniform float uTime; uniform float uL; varying float vA; uniform float uNight;
        void main() { float s = aSeed * 1000.0; vec3 p = position;
          p.x += sin(uTime * 0.31 + s) * 4.0 + sin(uTime * 0.11 + s * 1.7) * 7.0;
          p.z += cos(uTime * 0.27 + s * 1.3) * 4.0 + cos(uTime * 0.09 + s) * 7.0;
          p.y += 1.5 + sin(uTime * 0.8 + s * 2.1) * 1.2;
          vA = uNight * (0.35 + 0.65 * pow(0.5 + 0.5 * sin(uTime * (1.3 + aSeed) + s), 3.0));
          vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_PointSize = 260.0 / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying float vA; void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
          float a = vA * smoothstep(0.25, 0.0, r); gl_FragColor = vec4(vec3(1.0, 0.92, 0.45) * a * 2.2, a); }`,
    }));
    this.fireflies.frustumCulled = false;
    this.scene.add(this.fireflies);

    // selection marker and vision fan
    this.selRing = new THREE.Mesh(new THREE.RingGeometry(1.9, 2.3, 48).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false, toneMapped: false }));
    this.selRing.visible = false;
    this.selRing.renderOrder = 5;
    this.scene.add(this.selRing);
    this.fan = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.5, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    this.fan.visible = false;
    this.fan.renderOrder = 4;
    this.scene.add(this.fan);

    this.scene.add(this.island, this.props, this.clouds);
    this.setupComposer();

    // picking: a click that isn't a drag selects the nearest animal to the clicked ground point
    let down: { x: number; y: number } | null = null;
    const ray = new THREE.Raycaster();
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) return;
      const r = el.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
      const hit = this.groundMesh ? ray.intersectObject(this.groundMesh, false)[0] : undefined;
      if (hit) onPick(hit.point.x + this.L / 2, this.L / 2 - hit.point.z);
    });

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.loop();
  }

  private setupComposer() {
    this.composer?.dispose();
    this.composer = null;
    this.bloom = null;
    if (this.quality !== "high") return;
    const c = new EffectComposer(this.renderer);
    c.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.55, 0.6, 0.86);
    c.addPass(this.bloom);
    c.addPass(new OutputPass());
    this.composer = c;
    this.resize();
  }

  setQuality(q: Quality) {
    if (q === this.quality) return;
    this.quality = q;
    this.renderer.shadowMap.enabled = q === "high";
    this.sun.castShadow = q === "high";
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, q === "high" ? 2 : 1));
    this.setupComposer();
    if (this.L) {
      const g = this.grassLast;
      this.buildGrass();
      if (g) this.setGrass(g);
    }
    this.scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | undefined;
      if (m) m.needsUpdate = true;
    });
  }

  resize() {
    if (this.disposed) return;
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.quality === "high" ? 2 : 1));
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.composer?.setSize(w, h);
    (this.particles.points.material as THREE.ShaderMaterial).uniforms.uScale.value = h * 0.9;
  }

  // ---------------------------------------------------------------------------------------- world build
  private buildWorld(L: number, G: number) {
    this.L = L;
    this.G = G;
    this.height = makeHeight(L);
    for (const ch of [...this.island.children]) {
      this.island.remove(ch);
      const m = ch as THREE.Mesh;
      m.geometry?.dispose();
    }
    const H = this.height;

    // ground texture (soil under the grass, darker loam in refuges)
    this.groundData = new Uint8Array(G * G * 4);
    this.groundTex = new THREE.DataTexture(this.groundData, G, G, THREE.RGBAFormat);
    this.groundTex.magFilter = THREE.LinearFilter;
    this.groundTex.minFilter = THREE.LinearFilter;
    this.groundTex.colorSpace = THREE.SRGBColorSpace;
    this.groundTex.wrapS = this.groundTex.wrapT = THREE.RepeatWrapping;

    const seg = Math.min(160, G * 2);
    const geo = new THREE.PlaneGeometry(L, L, seg, seg).rotateX(-Math.PI / 2);
    const p = geo.getAttribute("position");
    for (let i = 0; i < p.count; i++) p.setY(i, H(p.getX(i) + L / 2, L / 2 - p.getZ(i)));
    geo.computeVertexNormals();
    const ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: this.groundTex, roughness: 1, metalness: 0 }));
    ground.receiveShadow = true;
    ground.name = "ground";
    this.island.add(ground);
    this.groundMesh = ground;
    // overlay: the same surface lifted just above short grass; tall blades poke through it
    const og = geo.clone();
    const op = og.getAttribute("position");
    for (let i = 0; i < op.count; i++) op.setY(i, op.getY(i) + 0.32 * (L / G / 2));
    this.ovMesh = new THREE.Mesh(og, new THREE.MeshBasicMaterial({
      // tone-mapped and dimmed slightly so the overlay never trips the bloom pass
      map: this.ovTex, color: 0xd0d0d0, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
    }));
    this.ovMesh.renderOrder = 2;
    this.ovMesh.visible = this.overlay !== "none";
    this.island.add(this.ovMesh);
    this.clearOverlay();

    // the island's cut sides: soil, clay and rock strata
    const depth = L * 0.11;
    const bands: [number, string][] = [[0, "#4a3523"], [-0.9, "#5b4129"], [-1.6, "#6e4a2c"], [-depth * 0.35, "#8a5a36"],
      [-depth * 0.4, "#6d6259"], [-depth * 0.7, "#5a524c"], [-depth * 0.74, "#4a4440"], [-depth, "#3a3532"]];
    const sidePos: number[] = [];
    const sideCol: number[] = [];
    const n = seg;
    const edges: [number, number, number, number][] = [[0, 0, 1, 0], [L, 0, 0, 1], [L, L, -1, 0], [0, L, 0, -1]];
    const col = new THREE.Color();
    for (const [x0, y0, dx, dy] of edges) {
      for (let k = 0; k < n; k++) {
        const xa = x0 + dx * (L * k) / n, ya = y0 + dy * (L * k) / n;
        const xb = x0 + dx * (L * (k + 1)) / n, yb = y0 + dy * (L * (k + 1)) / n;
        const ha = H(Math.min(xa, L - 1e-3), Math.min(ya, L - 1e-3)), hb = H(Math.min(xb, L - 1e-3), Math.min(yb, L - 1e-3));
        for (let b = 0; b < bands.length - 1; b++) {
          const za = b === 0 ? ha : bands[b][0], zb = b === 0 ? hb : bands[b][0];
          const z2 = bands[b + 1][0];
          const A = [xa - L / 2, za, L / 2 - ya], B = [xb - L / 2, zb, L / 2 - yb];
          const Cc = [xb - L / 2, z2, L / 2 - yb], D = [xa - L / 2, z2, L / 2 - ya];
          sidePos.push(...A, ...D, ...B, ...B, ...D, ...Cc);
          col.set(bands[b][1]).convertSRGBToLinear();
          const jitter = 0.94 + 0.06 * Math.sin(k * 3.7 + b);
          for (let v = 0; v < 6; v++) sideCol.push(col.r * jitter, col.g * jitter, col.b * jitter);
        }
      }
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.Float32BufferAttribute(sidePos, 3));
    sg.setAttribute("color", new THREE.Float32BufferAttribute(sideCol, 3));
    sg.computeVertexNormals();
    const sides = new THREE.Mesh(sg, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, flatShading: true, side: THREE.DoubleSide }));
    sides.receiveShadow = true;
    this.scene.add(sides);
    this.islandExtras.forEach((o) => this.scene.remove(o));
    this.islandExtras = [sides];
    const bottom = new THREE.Mesh(new THREE.PlaneGeometry(L, L).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x2a2624, roughness: 1 }));
    bottom.position.y = -depth;
    this.scene.add(bottom);
    this.islandExtras.push(bottom);

    // clouds drifting around and below the island
    for (const c of [...this.clouds.children]) this.clouds.remove(c);
    const cr = mulberry(31);
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true, transparent: true, opacity: 0.92 });
    for (let i = 0; i < 14; i++) {
      const g = new THREE.Group();
      const parts = 4 + Math.floor(cr() * 4);
      for (let k = 0; k < parts; k++) {
        const m = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), cloudMat);
        m.scale.set(5 + cr() * 7, 3 + cr() * 4, 4 + cr() * 6);
        m.position.set(k * 6 - parts * 3 + cr() * 3, cr() * 2, cr() * 6 - 3);
        g.add(m);
      }
      const ang = cr() * Math.PI * 2, rad = L * (0.72 + cr() * 0.55);
      g.position.set(Math.cos(ang) * rad, -depth * (0.2 + cr() * 1.4) + (cr() > 0.6 ? depth * 1.5 : 0), Math.sin(ang) * rad);
      g.userData = { ang, rad, speed: 0.004 + cr() * 0.006 };
      g.scale.setScalar(L / 200);
      this.clouds.add(g);
    }

    (this.fireflies.material as THREE.ShaderMaterial).uniforms.uL.value = L;
    const fp = this.fireflies.geometry.getAttribute("position") as THREE.BufferAttribute;
    const fr = mulberry(99);
    for (let i = 0; i < fp.count; i++) {
      const x = fr() * L, y = fr() * L;
      fp.setXYZ(i, x - L / 2, H(x, y) + 0.5, L / 2 - y);
    }
    fp.needsUpdate = true;

    // sun shadow camera covers the island
    const s = this.sun.shadow.camera;
    s.left = -L * 0.62;
    s.right = L * 0.62;
    s.top = L * 0.62;
    s.bottom = -L * 0.62;
    s.near = 1;
    s.far = L * 4;
    s.updateProjectionMatrix();

    this.controls.maxDistance = L * 3.2;
    this.buildGrass();
    if (!this.userTouched) this.camera.position.set(L * 0.08, L * 0.5, L * 0.86);
    this.controls.target.set(0, 0, 0);
  }
  private islandExtras: THREE.Object3D[] = [];
  private groundMesh: THREE.Mesh | null = null;

  private grassLast: Float32Array | null = null;

  private buildGrass() {
    if (this.grass) {
      this.island.remove(this.grass);
      this.grass.geometry.dispose();
    }
    const L = this.L, G = this.G, H = this.height;
    const per = this.quality === "high" ? (G >= 100 ? 5 : 12) : (G >= 100 ? 2 : 4);
    const n = G * G * per;
    // a tapered blade, 1 unit tall, bent slightly
    const cs0 = L / G;
    const w = 0.42 * cs0;
    const blade = new THREE.BufferGeometry();
    const ys = [0, 0.35, 0.7, 1];
    const ws = [1, 0.8, 0.45, 0];
    const v: number[] = [];
    const hv: number[] = [];
    for (let k = 0; k < 3; k++) {
      const y0 = ys[k], y1 = ys[k + 1], w0 = (ws[k] * w) / 2, w1 = (ws[k + 1] * w) / 2;
      v.push(-w0, y0, 0, w0, y0, 0, w1, y1, 0, -w0, y0, 0, w1, y1, 0, -w1, y1, 0);
      hv.push(y0, y0, y1, y0, y1, y1);
    }
    blade.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    blade.setAttribute("aH", new THREE.Float32BufferAttribute(hv, 1));
    blade.computeVertexNormals();
    const amount = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    amount.setUsage(THREE.DynamicDrawUsage);
    const seeds = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    blade.setAttribute("aG", amount);
    blade.setAttribute("aSeed", seeds);
    const mat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide });
    const maxH = 0.6 * cs0;
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.time.uTime;
      sh.uniforms.uMaxH = { value: maxH };
      sh.vertexShader = sh.vertexShader.replace("#include <common>", `#include <common>
        attribute float aG; attribute float aSeed; attribute float aH; uniform float uTime; uniform float uMaxH;
        varying float vG; varying float vH;`).replace("#include <begin_vertex>", `#include <begin_vertex>
        vG = sqrt(clamp(aG, 0.0, 1.0)); vH = aH;
        float gv = sqrt(clamp(aG, 0.0, 1.0));
        float hgt = uMaxH * (0.18 + 0.82 * gv) * (0.6 + 0.8 * aSeed);
        transformed.y *= hgt;
        vec3 ip = vec3(instanceMatrix[3][0], 0.0, instanceMatrix[3][2]);
        float wave = sin(dot(ip.xz, vec2(0.11, 0.07)) - uTime * 1.6 + aSeed * 2.0) * 0.6
                   + sin(dot(ip.xz, vec2(-0.05, 0.13)) - uTime * 0.9) * 0.4;
        float bend = aH * aH * (0.35 + 0.25 * wave) * hgt;
        transformed.x += bend;`);
      sh.fragmentShader = sh.fragmentShader.replace("#include <common>", `#include <common>
        varying float vG; varying float vH;`).replace("vec4 diffuseColor = vec4( diffuse, opacity );", `
        vec3 dry = vec3(0.36, 0.30, 0.08), mid = vec3(0.17, 0.36, 0.06), lush = vec3(0.06, 0.30, 0.04);
        vec3 base = vG < 0.45 ? mix(dry, mid, vG / 0.45) : mix(mid, lush, (vG - 0.45) / 0.55);
        base *= mix(0.45, 1.25, vH);
        base += vec3(0.10, 0.12, 0.0) * vH * vH * vG;
        vec4 diffuseColor = vec4( base, opacity );`);
    };
    const mesh = new THREE.InstancedMesh(blade, mat, n);
    mesh.receiveShadow = this.quality === "high";
    mesh.frustumCulled = false;
    const r = mulberry(1234);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const cs = L / G;
    this.grassCell = new Int32Array(n);
    let k = 0;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        for (let b = 0; b < per; b++, k++) {
          const x = (i + r()) * cs, y = (j + r()) * cs;
          e.set(0, r() * Math.PI * 2, (r() - 0.5) * 0.3);
          q.setFromEuler(e);
          m.compose(new THREE.Vector3(x - L / 2, H(x, y) - 0.05, L / 2 - y), q, new THREE.Vector3(1, 1, 1));
          mesh.setMatrixAt(k, m);
          seeds.setX(k, r());
          this.grassCell[k] = i * G + j;
        }
      }
    }
    this.grass = mesh;
    this.grassAmount = amount;
    this.island.add(mesh);

    // wildflowers: open where the grass is thick, gone where it's grazed
    if (this.flowers) {
      this.island.remove(this.flowers.mesh);
      this.flowers.mesh.geometry.dispose();
    }
    const nf = Math.round(G * G * (this.quality === "high" ? 0.22 : 0.08));
    const fgeo = new THREE.IcosahedronGeometry(0.22 * cs0, 0);
    const fm = new THREE.InstancedMesh(fgeo, new THREE.MeshLambertMaterial({ emissive: 0x111111 }), nf);
    fm.frustumCulled = false;
    const palette = ["#fbf7ef", "#f6d94a", "#c69cf2", "#f4a7c3", "#ffffff", "#f2c14e"].map((c) => new THREE.Color(c));
    const fcell = new Int32Array(nf), fpos = new Float32Array(nf * 3);
    for (let i = 0; i < nf; i++) {
      const cx = Math.floor(r() * G), cy = Math.floor(r() * G);
      const x = (cx + r()) * cs0, y = (cy + r()) * cs0;
      fcell[i] = cx * G + cy;
      fpos.set([x - L / 2, H(x, y), L / 2 - y], i * 3);
      fm.setColorAt(i, palette[Math.floor(r() * palette.length)]);
    }
    this.flowers = { mesh: fm, cell: fcell, pos: fpos, maxH };
    this.island.add(fm);
  }
  private flowers: { mesh: THREE.InstancedMesh; cell: Int32Array; pos: Float32Array; maxH: number } | null = null;

  private buildRefuges(centres: [number, number][], radius: number) {
    for (const c of [...this.props.children]) {
      this.props.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
    const L = this.L, H = this.height;
    const r = mulberry(4242);
    const rocks: THREE.Matrix4[] = [], rockCol: THREE.Color[] = [];
    const bushes: THREE.Matrix4[] = [], bushCol: THREE.Color[] = [];
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const e = new THREE.Euler();
    const wrap = (v: number) => ((v % L) + L) % L;
    const put = (list: THREE.Matrix4[], x: number, y: number, s: THREE.Vector3, sink: number) => {
      const wx = wrap(x), wy = wrap(y);
      e.set(r() * 0.6, r() * Math.PI * 2, r() * 0.6);
      q.setFromEuler(e);
      list.push(m.compose(new THREE.Vector3(wx - L / 2, H(wx, wy) - sink, L / 2 - wy), q, s).clone());
    };
    const k = L / 200;
    for (const [cx, cy] of centres) {
      const nr = 7 + Math.floor(r() * 4);
      for (let i = 0; i < nr; i++) {
        const a = (i / nr) * Math.PI * 2 + r() * 0.5, d = radius * (0.72 + r() * 0.3);
        const s = (0.8 + r() * 1.5) * k;
        put(rocks, cx + Math.cos(a) * d, cy + Math.sin(a) * d, new THREE.Vector3(s * (1 + r()), s * (0.6 + r() * 0.6), s * (1 + r())), 0.25 * s);
        rockCol.push(new THREE.Color().setHSL(0.58 + r() * 0.06, 0.06 + r() * 0.06, 0.42 + r() * 0.16));
      }
      for (let i = 0; i < 2; i++) {
        const s = (1.4 + r() * 1.2) * k;
        put(rocks, cx + (r() - 0.5) * radius * 0.6, cy + (r() - 0.5) * radius * 0.6, new THREE.Vector3(s * 1.4, s * 0.7, s * 1.2), 0.3 * s);
        rockCol.push(new THREE.Color().setHSL(0.6, 0.08, 0.38 + r() * 0.1));
      }
      const nb = 5 + Math.floor(r() * 4);
      for (let i = 0; i < nb; i++) {
        const a = r() * Math.PI * 2, d = radius * Math.sqrt(r()) * 0.85;
        const s = (1.3 + r() * 1.4) * k;
        put(bushes, cx + Math.cos(a) * d, cy + Math.sin(a) * d, new THREE.Vector3(s, s * (0.75 + r() * 0.35), s), 0.2 * s);
        bushCol.push(new THREE.Color().setHSL(0.27 + r() * 0.08, 0.45 + r() * 0.2, 0.2 + r() * 0.1));
      }
    }
    const mk = (geo: THREE.BufferGeometry, mat: THREE.Material, list: THREE.Matrix4[], cols: THREE.Color[]) => {
      const im = new THREE.InstancedMesh(geo, mat, list.length);
      list.forEach((mm, i) => {
        im.setMatrixAt(i, mm);
        im.setColorAt(i, cols[i]);
      });
      im.castShadow = true;
      im.receiveShadow = true;
      this.props.add(im);
    };
    mk(new THREE.DodecahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.95 }), rocks, rockCol);
    mk(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.9 }), bushes, bushCol);
  }

  private setGrass(grass: Float32Array) {
    this.grassLast = grass;
    const G = this.G;
    const d = this.groundData!;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const g = grass[i * G + j];
        const ref = this.refuge?.[i * G + j];
        // soil warms toward olive where grass is thick
        const [r0, g0, b0] = turf(g);
        let r = r0, gg = g0, b = b0;
        if (ref) {
          r = r * 0.62 + 30;
          gg = gg * 0.62 + 34;
          b = b * 0.62 + 30;
        }
        const k = (j * G + i) * 4;
        d[k] = r;
        d[k + 1] = gg;
        d[k + 2] = b;
        d[k + 3] = 255;
      }
    }
    this.groundTex!.needsUpdate = true;
    if (this.grassAmount && this.grassCell) {
      const a = this.grassAmount.array as Float32Array;
      const cell = this.grassCell;
      for (let k = 0; k < a.length; k++) a[k] = grass[cell[k]];
      this.grassAmount.needsUpdate = true;
    }
    if (this.flowers) {
      const { mesh, cell, pos, maxH } = this.flowers;
      const arr = mesh.instanceMatrix.array as Float32Array;
      for (let i = 0; i < cell.length; i++) {
        const v = Math.sqrt(Math.max(0, grass[cell[i]]));
        const sc = Math.max(0, Math.min(1, (v - 0.42) / 0.25));
        const o = i * 16;
        arr.fill(0, o, o + 16);
        arr[o] = sc;
        arr[o + 5] = sc;
        arr[o + 10] = sc;
        arr[o + 12] = pos[i * 3];
        arr[o + 13] = pos[i * 3 + 1] + maxH * (0.18 + 0.82 * v) * 0.9;
        arr[o + 14] = pos[i * 3 + 2];
        arr[o + 15] = 1;
      }
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  // ---------------------------------------------------------------------------------------- frames
  update(f: Frame) {
    const now = performance.now() / 1000;
    if (f.world !== this.L || f.grid !== this.G) this.buildWorld(f.world, f.grid);
    if (f.refuge) this.refuge = f.refuge;
    if (f.refugeCentres) this.buildRefuges(f.refugeCentres, f.refugeRadius);
    if (f.grass) this.setGrass(f.grass);
    if (f.reset || !this.prey || this.prey.cap !== f.capacity[0] || this.pred!.cap !== f.capacity[1]) this.resetTracks(f.capacity);
    if (f.reset) {
      this.heat.fill(0);
      this.heatMax = 0;
      this.clearOverlay();
    }
    const dt = now - this.lastFrameAt;
    if (this.lastFrameAt > 0 && dt < 0.5) this.frameInterval = this.frameInterval * 0.8 + dt * 0.2;
    this.lastFrameAt = now;
    this.simT = f.t;

    const born = new Set<number>();
    for (let i = 0; i < f.bornSlots.length; i += 2) born.add(f.bornSlots[i] * 1e6 + f.bornSlots[i + 1]);
    const caught = new Set<number>();
    for (let i = 0; i < f.catchEvents.length; i += 4) caught.add(f.catchEvents[i + 3]);
    this.applyHerd(this.prey!, f.prey, 0, born, caught, now, f.reset);
    this.applyHerd(this.pred!, f.pred, 1, born, null, now, f.reset);

    // effects
    const H = this.height, L = this.L;
    if (!f.reset) {
      for (let i = 0; i < f.catchEvents.length; i += 4) {
        const x = f.catchEvents[i], y = f.catchEvents[i + 1];
        const X = x - L / 2, Z = L / 2 - y, Y = H(x, y);
        this.particles.emit(X, Y + 0.6, Z, 26, { color: C("#ff7a45"), speed: 7, up: 5, life: 0.9, size: 0.9, grav: 9 });
        this.particles.emit(X, Y + 0.6, Z, 14, { color: C("#f1dfb8"), speed: 4, up: 3, life: 1.3, size: 0.7, grav: 3 });
        this.shockwave(X, Y + 0.15, Z);
        this.addHeat(x, y);
      }
      for (let i = 0; i < f.bornSlots.length && i < 400; i += 2) {
        const sp = f.bornSlots[i], slot = f.bornSlots[i + 1];
        const tr = sp === 0 ? this.prey! : this.pred!;
        const x = tr.cx[slot], y = tr.cy[slot];
        this.particles.emit(x - L / 2, H(x, y) + 0.8, L / 2 - y, sp === 0 ? 8 : 16,
          { color: sp === 0 ? C("#ffe08a") : C("#ffb36b"), speed: 1.2, up: 2.4, life: 1.4, size: 0.55, grav: -0.6, spread: 1 });
      }
      for (let i = 0; i < f.deathEvents.length && i < 400; i += 4) {
        const x = f.deathEvents[i + 1], y = f.deathEvents[i + 2];
        this.particles.emit(x - L / 2, H(x, y) + 0.4, L / 2 - y, 6, { color: C("#8d8172"), speed: 1, up: 1.2, life: 1.2, size: 0.8, grav: 0.4 });
      }
    }

    this.selected = f.selected;
    if (f.selected) {
      this.selSpecies = f.selected.species === "prey" ? 0 : 1;
      this.selSlot = f.selected.slot;
    } else this.selSlot = -1;
  }

  private resetTracks(cap: [number, number]) {
    for (const t of [this.prey, this.pred]) if (t) {
      this.scene.remove(t.mesh);
      t.mesh.dispose();
    }
    const pg = preyGeometry(), qg = predatorGeometry();
    const k = this.L / 200;
    this.prey = makeTrack(cap[0], pg, this.preyMat, 1.25 * Math.max(0.7, k));
    this.pred = makeTrack(cap[1], qg, this.predMat, 1.3 * Math.max(0.7, k));
    this.scene.add(this.prey.mesh, this.pred.mesh);
  }

  private applyHerd(tr: Track, h: Herd, sp: number, born: Set<number>, caught: Set<number> | null, now: number, reset: boolean) {
    const seen = new Uint8Array(tr.cap);
    const L = this.L;
    for (let k = 0; k < h.slot.length; k++) {
      const s = h.slot[k];
      seen[s] = 1;
      const isNew = reset || !tr.alive[s] || born.has(sp * 1e6 + s) || tr.dying[s] > 0;
      if (isNew) {
        tr.px[s] = h.x[k];
        tr.py[s] = h.y[k];
        tr.ph[s] = h.h[k];
        tr.born[s] = reset ? now - 10 : now;
        tr.dying[s] = 0;
        tr.pose[s] = 0;
        tr.spdS[s] = 0;
      } else {
        tr.px[s] = tr.cx[s];
        tr.py[s] = tr.cy[s];
        tr.ph[s] = tr.ch[s];
        // wrapped across an edge of the torus: don't slide across the whole island
        if (Math.abs(h.x[k] - tr.px[s]) > L / 2 || Math.abs(h.y[k] - tr.py[s]) > L / 2) {
          tr.px[s] = h.x[k];
          tr.py[s] = h.y[k];
        }
      }
      tr.cx[s] = h.x[k];
      tr.cy[s] = h.y[k];
      tr.ch[s] = h.h[k];
      tr.spd[s] = h.spd[k];
      tr.energy[s] = h.energy[k];
      const busy = (h.flags[k] & 4) !== 0, refuge = (h.flags[k] & 2) !== 0;
      tr.poseT[s] = sp === 1 ? (busy ? 1 : 0) : refuge && h.spd[k] < 0.15 ? 0.8 : 0;
      tr.alive[s] = 1;
    }
    for (let s = 0; s < tr.cap; s++) {
      if (tr.alive[s] && !seen[s]) {
        tr.alive[s] = 0;
        // caught prey vanish in the burst; others shrink away
        tr.dying[s] = caught?.has(s) ? -1 : now;
      }
    }
  }

  private shockwave(x: number, y: number, z: number) {
    let r = this.rings.find((q) => q.t <= 0);
    if (!r) {
      if (this.rings.length > 40) return;
      const mesh = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.1, 40).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: 0xffa060, transparent: true, depthWrite: false, toneMapped: false }));
      this.scene.add(mesh);
      r = { mesh, t: 0 };
      this.rings.push(r);
    }
    r.mesh.position.set(x, y, z);
    r.t = 0.7;
    r.mesh.visible = true;
  }

  // ---------------------------------------------------------------------------------------- camera / sky
  setCameraMode(mode: CameraMode) {
    this.cameraMode = mode;
    const L = this.L || 200;
    this.controls.autoRotate = false;
    if (mode === "tour") {
      this.tourNext = 0; // pick a subject on the next frame
      return;
    }
    if (mode === "top") this.tweenTo(new THREE.Vector3(0, L * 1.55, 0.01), new THREE.Vector3(0, 0, 0));
    if (mode === "orbit") this.tweenTo(new THREE.Vector3(L * 0.08, L * 0.5, L * 0.86), new THREE.Vector3(0, 0, 0));
    if (mode === "follow" && this.selSlot >= 0) {
      const tr = this.selSpecies === 0 ? this.prey! : this.pred!;
      const x = tr.cx[this.selSlot], y = tr.cy[this.selSlot];
      const t = new THREE.Vector3(x - L / 2, this.height(x, y), L / 2 - y);
      this.tweenTo(t.clone().add(new THREE.Vector3(-14, 11, 14).multiplyScalar(L / 200)), t);
    }
  }

  setSky(mode: SkyMode) {
    this.skyMode = mode;
  }

  setOverlay(o: Overlay) {
    this.overlay = o;
    if (this.ovMesh) this.ovMesh.visible = o !== "none";
    this.clearOverlay();
    if (o === "catches") this.heatDirty = true;
  }

  private clearOverlay() {
    this.ovCtx.clearRect(0, 0, this.ovCanvas.width, this.ovCanvas.height);
    this.ovTex.needsUpdate = true;
  }

  private addHeat(x: number, y: number) {
    const N = 128, L = this.L;
    const cx = (x / L) * N, cy = (y / L) * N;
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const i = (Math.floor(cx) + dx + N) % N, j = (Math.floor(cy) + dy + N) % N;
        const w = Math.exp(-(dx * dx + dy * dy) / 3.5);
        const k = j * N + i;
        this.heat[k] += w;
        if (this.heat[k] > this.heatMax) this.heatMax = this.heat[k];
      }
    }
    this.heatDirty = true;
  }

  private drawOverlay(now: number, alpha: number) {
    if (this.overlay === "none" || !this.prey || !this.pred) return;
    const g = this.ovCtx, W = this.ovCanvas.width, L = this.L;
    if (this.overlay === "trails") {
      if (now - this.trailAt < 1 / 40) return;
      this.trailAt = now;
      // fade what's there, then stamp every animal where it is now
      g.globalCompositeOperation = "destination-out";
      g.fillStyle = "rgba(0,0,0,0.045)";
      g.fillRect(0, 0, W, W);
      g.globalCompositeOperation = "source-over";
      const stamp = (tr: Track, color: string, r: number, movingOnly: boolean) => {
        g.fillStyle = color;
        g.beginPath();
        for (let s = 0; s < tr.cap; s++) {
          if (!tr.alive[s] || (movingOnly && tr.spdS[s] < 0.2)) continue;
          const x = tr.px[s] + (tr.cx[s] - tr.px[s]) * alpha, y = tr.py[s] + (tr.cy[s] - tr.py[s]) * alpha;
          const X = (x / L) * W, Y = (1 - y / L) * W;
          g.moveTo(X + r, Y);
          g.arc(X, Y, r, 0, Math.PI * 2);
        }
        g.fill();
      };
      // prey leave faint pale paths; predators bright embers, so hunts stand out
      stamp(this.prey, "rgba(255,244,214,0.09)", 1.1, true);
      stamp(this.pred, "rgba(255,110,50,0.85)", 2.6, false);
      this.ovTex.needsUpdate = true;
      return;
    }
    // catches: a heat map of where prey were caught since the world started
    if (!this.heatDirty || now - this.heatDrawnAt < 0.25) return;
    this.heatDirty = false;
    this.heatDrawnAt = now;
    const N = 128;
    const img = g.createImageData(N, N);
    const max = Math.max(1, this.heatMax);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const v = Math.sqrt(this.heat[j * N + i] / max);
        const o = ((N - 1 - j) * N + i) * 4;
        // transparent -> deep red -> orange -> pale yellow
        const r = 255, gg = Math.round(40 + 200 * Math.max(0, v - 0.35) / 0.65), b = Math.round(30 + 150 * Math.max(0, v - 0.75) / 0.25);
        img.data[o] = r;
        img.data[o + 1] = gg;
        img.data[o + 2] = b;
        img.data[o + 3] = Math.round(235 * Math.min(1, v * 1.4));
      }
    }
    const tmp = document.createElement("canvas");
    tmp.width = tmp.height = N;
    tmp.getContext("2d")!.putImageData(img, 0, 0);
    g.clearRect(0, 0, W, W);
    g.imageSmoothingEnabled = true;
    g.drawImage(tmp, 0, 0, W, W);
    this.ovTex.needsUpdate = true;
  }

  /** Tour: every few seconds pick a predator on the move (a hunt in progress) and glide to it. */
  private updateTour(now: number) {
    if (this.cameraMode !== "tour" || !this.pred || !this.prey) return;
    // wait a moment for the selection to come back before treating the subject as lost
    const lost = now - this.tourPickedAt > 4 && (this.selSlot < 0 || (this.selected !== null && !this.selected.alive));
    if (now < this.tourNext && !lost) return;
    this.tourPickedAt = now;
    this.tourNext = now + 9;
    const pick = (tr: Track, minSpd: number) => {
      const cand: number[] = [];
      for (let s = 0; s < tr.cap; s++) if (tr.alive[s] && tr.spdS[s] >= minSpd) cand.push(s);
      return cand.length ? cand[Math.floor(Math.random() * cand.length)] : -1;
    };
    let tr = this.pred;
    let s = pick(tr, 0.6);
    if (s < 0) s = pick(tr, 0);
    if (s < 0) {
      tr = this.prey;
      s = pick(tr, 0);
    }
    if (s < 0) return;
    const x = tr.cx[s], y = tr.cy[s], L = this.L, k = L / 200;
    if (this.onTourPick) this.onTourPick(tr === this.prey ? 0 : 1, s);
    else this.onPickCb(x, y);
    const t = new THREE.Vector3(x - L / 2, this.height(x, y) + 1, L / 2 - y);
    const ang = Math.random() * Math.PI * 2;
    this.tweenTo(t.clone().add(new THREE.Vector3(Math.cos(ang) * 30 * k, 17 * k, Math.sin(ang) * 30 * k)), t, 2.4);
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.9;
  }

  /** Clock text for the current visual time of day. */
  clockText(): string {
    const day = Math.floor(this.simT / DAY_STEPS + 0.42) + 1;
    const minutes = Math.floor(this.shownPhase * 24 * 60);
    const hh = String(Math.floor(minutes / 60)).padStart(2, "0"), mm = String(minutes % 60).padStart(2, "0");
    return this.skyMode === "cycle" ? `Day ${day} · ${hh}:${mm}` : `${hh}:${mm}`;
  }

  private tweenTo(pos: THREE.Vector3, target: THREE.Vector3, dur = 1.3) {
    this.camTween = { from: this.camera.position.clone(), to: pos, tFrom: this.controls.target.clone(), tTo: target, t: 0, dur };
  }

  private updateSky(dt: number) {
    const target = this.skyMode === "cycle" ? ((this.simT / DAY_STEPS + 0.42) % 1)
      : this.skyMode === "day" ? 0.5 : this.skyMode === "golden" ? 0.735 : 0.0;
    // move the shown time toward the target the short way round
    let d = target - this.shownPhase;
    if (d > 0.5) d -= 1;
    if (d < -0.5) d += 1;
    const rate = this.skyMode === "cycle" && Math.abs(d) < 0.05 ? 1 : Math.min(1, dt * 2.2);
    this.shownPhase = (this.shownPhase + d * rate + 1) % 1;
    const ph = this.shownPhase;
    const ang = ph * Math.PI * 2 - Math.PI / 2; // -pi/2 at midnight
    const elev = Math.sin(ang);
    const s = skyAt(elev);
    const L = this.L || 200;
    const dir = new THREE.Vector3(Math.cos(ang) * 0.85, Math.max(0.05, elev), 0.45).normalize();
    this.sun.position.copy(dir).multiplyScalar(L * 1.6);
    this.sun.target.position.set(0, 0, 0);
    this.sun.color.copy(s.sun);
    this.sun.intensity = s.sunI;
    this.moon.intensity = s.moon;
    this.moon.position.set(-dir.x * L, L * 1.2, -dir.z * L);
    this.hemi.intensity = s.hemi;
    this.hemi.color.copy(s.top).lerp(C("#ffffff"), 0.5);
    this.skyMat.uniforms.uTop.value.copy(s.top);
    this.skyMat.uniforms.uHor.value.copy(s.hor);
    this.skyMat.uniforms.uSunDir.value.set(Math.cos(ang), elev, 0.45);
    this.skyMat.uniforms.uSun.value.copy(s.sun).multiplyScalar(elev > -0.05 ? 1 : 0);
    (this.scene.fog as THREE.Fog).color.copy(s.hor);
    const night = THREE.MathUtils.clamp((0.08 - elev) / 0.25, 0, 1);
    (this.stars.material as THREE.PointsMaterial).opacity = night * 0.9;
    (this.fireflies.material as THREE.ShaderMaterial).uniforms.uNight.value = night;
    this.renderer.toneMappingExposure = s.exp;
    if (this.bloom) this.bloom.strength = 0.35 + night * 0.6;
  }

  // ---------------------------------------------------------------------------------------- render loop
  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(0.1, this.clock.getDelta());
    this.time.uTime.value += dt;
    const now = performance.now() / 1000;
    this.watchSpeed(dt);

    if (this.L) {
      const alpha = THREE.MathUtils.clamp((now - this.lastFrameAt) / Math.max(this.frameInterval, 1 / 120), 0, 1);
      if (this.prey) this.drawTrack(this.prey, alpha, dt, now, 0);
      if (this.pred) this.drawTrack(this.pred, alpha, dt, now, 1);
      this.updateSelection(alpha);
      this.updateSky(dt);
      this.drawOverlay(now, alpha);
      this.updateTour(now);
      for (const c of this.clouds.children) {
        const u = c.userData as { ang: number; rad: number; speed: number };
        u.ang += u.speed * dt;
        c.position.x = Math.cos(u.ang) * u.rad;
        c.position.z = Math.sin(u.ang) * u.rad;
      }
      for (const r of this.rings) {
        if (r.t <= 0) continue;
        r.t -= dt;
        const p = 1 - r.t / 0.7;
        r.mesh.scale.setScalar(1 + p * 5);
        (r.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.8;
        if (r.t <= 0) r.mesh.visible = false;
      }
    }
    this.particles.update(dt);

    if (this.camTween) {
      const tw = this.camTween;
      tw.t += dt / tw.dur;
      const e = tw.t >= 1 ? 1 : 1 - Math.pow(1 - tw.t, 3);
      this.camera.position.lerpVectors(tw.from, tw.to, e);
      this.controls.target.lerpVectors(tw.tFrom, tw.tTo, e);
      if (tw.t >= 1) this.camTween = null;
    } else if ((this.cameraMode === "follow" || this.cameraMode === "tour") && this.selSlot >= 0 && this.selected?.alive) {
      const tr = this.selSpecies === 0 ? this.prey! : this.pred!;
      const s = this.selSlot;
      const x = tr.cx[s], y = tr.cy[s];
      const goal = new THREE.Vector3(x - this.L / 2, this.height(x, y) + 1, this.L / 2 - y);
      if (goal.distanceTo(this.controls.target) > this.L / 3) this.controls.target.copy(goal);
      const delta = goal.sub(this.controls.target).multiplyScalar(Math.min(1, dt * 3));
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    this.controls.update();
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  };

  private watchSpeed(dt: number) {
    if (this.quality !== "high" || !this.onSlowDevice) return;
    this.frameTimes.push(dt);
    if (this.frameTimes.length < 150) return;
    const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    this.frameTimes = [];
    if (avg > 1 / 24) {
      const cb = this.onSlowDevice;
      this.onSlowDevice = null;
      cb();
    }
  }

  private drawTrack(tr: Track, alpha: number, dt: number, now: number, sp: number) {
    const L = this.L, H = this.height;
    const arr = tr.mesh.instanceMatrix.array as Float32Array;
    const anim = tr.anim.array as Float32Array;
    const col = tr.mesh.instanceColor!.array as Float32Array;
    const freq = sp === 0 ? 9 : 7.5;
    for (let s = 0; s < tr.cap; s++) {
      let scale = 0;
      if (tr.alive[s]) {
        const age = now - tr.born[s];
        scale = age >= 0.5 ? 1 : easeOutBack(Math.max(0, age) / 0.5);
      } else if (tr.dying[s] > 0) {
        const d = (now - tr.dying[s]) / 0.45;
        if (d >= 1) tr.dying[s] = 0;
        else scale = 1 - d;
      }
      const o = s * 16;
      if (scale <= 0.001) {
        if (arr[o] !== 0 || arr[o + 5] !== 0 || arr[o + 10] !== 0) {
          for (let k = 0; k < 16; k++) arr[o + k] = 0;
        }
        continue;
      }
      const x = tr.px[s] + (tr.cx[s] - tr.px[s]) * alpha;
      const y = tr.py[s] + (tr.cy[s] - tr.py[s]) * alpha;
      let dh = tr.ch[s] - tr.ph[s];
      if (dh > Math.PI) dh -= 2 * Math.PI;
      if (dh < -Math.PI) dh += 2 * Math.PI;
      const h = tr.ph[s] + dh * alpha;
      tr.spdS[s] += (tr.spd[s] - tr.spdS[s]) * Math.min(1, dt * 8);
      tr.pose[s] += (tr.poseT[s] - tr.pose[s]) * Math.min(1, dt * 3);
      tr.gait[s] += tr.spdS[s] * freq * dt * (1 + 0.15 * tr.seed[s]);
      const sc = tr.base * scale * (0.9 + 0.2 * tr.seed[s]);
      const c = Math.cos(h) * sc, sn = Math.sin(h) * sc;
      // column-major: rotation about Y by heading (model faces +X), uniform scale, translation
      arr[o] = c; arr[o + 1] = 0; arr[o + 2] = -sn; arr[o + 3] = 0;
      arr[o + 4] = 0; arr[o + 5] = sc; arr[o + 6] = 0; arr[o + 7] = 0;
      arr[o + 8] = sn; arr[o + 9] = 0; arr[o + 10] = c; arr[o + 11] = 0;
      arr[o + 12] = x - L / 2; arr[o + 13] = H(x, y) - 0.05; arr[o + 14] = L / 2 - y; arr[o + 15] = 1;
      anim[s * 4] = tr.gait[s];
      anim[s * 4 + 1] = tr.spdS[s];
      anim[s * 4 + 2] = tr.pose[s];
      anim[s * 4 + 3] = tr.seed[s];
      // hungry animals look paler
      const e = THREE.MathUtils.clamp(tr.energy[s] * 1.6, 0.55, 1);
      const v = 0.85 + 0.3 * tr.seed[s];
      col[s * 3] = v * (0.75 + 0.25 * e);
      col[s * 3 + 1] = v * (0.7 + 0.3 * e);
      col[s * 3 + 2] = v * (0.7 + 0.3 * e);
    }
    tr.mesh.count = tr.cap;
    tr.mesh.instanceMatrix.needsUpdate = true;
    tr.anim.needsUpdate = true;
    tr.mesh.instanceColor!.needsUpdate = true;
  }

  private updateSelection(alpha: number) {
    const sel = this.selected;
    if (!sel || this.selSlot < 0) {
      this.selRing.visible = false;
      this.fan.visible = false;
      return;
    }
    const tr = this.selSpecies === 0 ? this.prey! : this.pred!;
    const s = this.selSlot;
    const L = this.L;
    const x = sel.alive ? tr.px[s] + (tr.cx[s] - tr.px[s]) * alpha : sel.x;
    const y = sel.alive ? tr.py[s] + (tr.cy[s] - tr.py[s]) * alpha : sel.y;
    const X = x - L / 2, Z = L / 2 - y, Y = this.height(x, y);
    const pulse = 1 + Math.sin(this.time.uTime.value * 5) * 0.08;
    this.selRing.visible = true;
    this.selRing.position.set(X, Y + 0.25, Z);
    this.selRing.scale.setScalar(pulse * (this.selSpecies === 1 ? 1.4 : 1) * (L / 200 < 0.7 ? 0.8 : 1));
    if (!sel.alive || sel.obs.length < 7) {
      this.fan.visible = false;
      return;
    }
    let dh = tr.ch[s] - tr.ph[s];
    if (dh > Math.PI) dh -= 2 * Math.PI;
    if (dh < -Math.PI) dh += 2 * Math.PI;
    const head = tr.ph[s] + dh * alpha;
    const R = (sel.obs.length - 6) / 6;
    const cone = (sel.coneDeg * Math.PI) / 180;
    const pos: number[] = [];
    const cols: number[] = [];
    const palette = [C("#9be15d"), C("#ffe7b0"), C("#ff6a3d"), C("#a9c6ff"), C("#ffffff")];
    for (let r = 0; r < R; r++) {
      const o = r * 6;
      let k = [1, 2, 3, 4, 5].findIndex((c) => sel.obs[o + c] > 0.5);
      if (k < 0) k = 4;
      const len = Math.max(0.8, sel.obs[o] * sel.range);
      const a0 = head - cone / 2 + (r / R) * cone, a1 = head - cone / 2 + ((r + 1) / R) * cone;
      const c = palette[k];
      const fade = k === 4 ? 0.18 : 0.95;
      const p0 = [X, Y + 0.5, Z];
      const pa = [X + Math.cos(a0) * len, Y + 0.5, Z - Math.sin(a0) * len];
      const pb = [X + Math.cos(a1) * len, Y + 0.5, Z - Math.sin(a1) * len];
      pos.push(...p0, ...pa, ...pb);
      for (let v = 0; v < 3; v++) cols.push(c.r * fade, c.g * fade, c.b * fade);
    }
    const g = this.fan.geometry;
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    g.computeBoundingSphere();
    this.fan.visible = true;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.controls.dispose();
    this.composer?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

// ground colour by grass amount: bare soil, straw, meadow, lush (sRGB 0-255); sqrt spreads the low end,
// where the simulation's grazed grass spends most of its time
const TURF: [number, number[]][] = [[0, [112, 88, 58]], [0.18, [156, 146, 74]], [0.38, [112, 152, 62]], [0.65, [74, 138, 48]], [1, [46, 110, 38]]];
function turf(g: number): number[] {
  const v = Math.sqrt(Math.max(0, Math.min(1, g)));
  let i = 0;
  while (i < TURF.length - 2 && v > TURF[i + 1][0]) i++;
  const [a, ca] = TURF[i], [b, cb] = TURF[i + 1];
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t];
}

function easeOutBack(t: number) {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

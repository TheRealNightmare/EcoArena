// The landing page's 3D story: a small EcoArena island with scripted scenes. Scroll position picks the camera
// shot (one per chapter); the animals live on their own clock, so every shot is always in motion.
//
// This is an illustration of the rules, not the simulation: the hunt is choreographed so it always plays out
// on screen, while the numbers quoted beside it come from the real world config.
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { animateMaterial, predatorGeometry, preyGeometry } from "../view/creatures";
import { makeHeight, mulberry, Particles, skyAt } from "../view/Diorama";

export const CHAPTERS = ["hero", "world", "prey", "predators", "refuges", "vision", "methods",
  "findings-prey", "findings-predators", "findings-reward", "enter"] as const;
const CH = (name: (typeof CHAPTERS)[number]) => CHAPTERS.indexOf(name);

const S = 80; // island size (world units)
const G = 64; // grass cells per side
const C = (h: string) => new THREE.Color(h);
const REFUGE: [number, number] = [58, 52];
const REFUGE_R = 7;
const HERD = 28;
const WOLVES = 4;

type V3 = [number, number, number];
// camera shot per chapter: position, look-at target (scene coords), sun elevation (-1 midnight .. 1 noon)
const SHOTS: { pos: V3; target: V3; elev: number }[] = [
  { pos: [62, 46, 74], target: [0, -4, 0], elev: 0.15 }, // hero: the whole island in warm late-afternoon light
  { pos: [-30, 7, 22], target: [-6, 0.5, -6], elev: 0.42 }, // world: low over the grass
  { pos: [-4, 6.5, 13], target: [-11, 0.8, 4], elev: 0.6 }, // prey: the grazing herd
  { pos: [-2, 16, 34], target: [11, 0, -5], elev: 0.55 }, // predators: the hunt lane
  { pos: [36, 13, 4], target: [18, 0.5, -12], elev: 0.35 }, // refuges: the rocks the hare runs to
  { pos: [-2, 22, 16], target: [-11, 0.5, 4], elev: 0.5 }, // vision: from above one hare
  { pos: [0, 118, 34], target: [0, 0, 0], elev: 0.7 }, // methods: map view
  { pos: [-78, 34, -52], target: [0, -4, 0], elev: 0.07 }, // findings 1: golden dusk from the far side
  { pos: [-6, 15, 46], target: [17, 0, 12], elev: 0.03 }, // findings 2: the wolves' side of the island
  { pos: [72, 42, -58], target: [0, -4, 0], elev: -0.01 }, // findings 3: sunset from the other corner
  { pos: [40, 22, 52], target: [0, 0, 0], elev: -0.05 }, // enter: twilight, fireflies
];

const smooth = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);

interface Animal {
  x: number; y: number; h: number; // world coords, heading (radians, x right / y up the map)
  spd: number; spdS: number; gait: number; pose: number; poseT: number; scale: number; seed: number;
  vx?: number; vy?: number; goal?: [number, number]; timer: number;
}

export class StoryScene {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, 1, 0.5, 3000);
  private composer: EffectComposer | null = null;
  private bloom: UnrealBloomPass | null = null;
  private time = { uTime: { value: 0 } };
  private H = makeHeight(S);
  private grass = new Float32Array(G * G);
  private grassAttr!: THREE.InstancedBufferAttribute;
  private grassCell!: Int32Array;
  private groundData = new Uint8Array(G * G * 4);
  private groundTex = new THREE.DataTexture(this.groundData, G, G, THREE.RGBAFormat);
  private herd: Animal[] = [];
  private wolves: Animal[] = [];
  private chaseHare!: Animal;
  private chaseWolf!: Animal;
  private preyMesh!: THREE.InstancedMesh;
  private predMesh!: THREE.InstancedMesh;
  private preyAnim!: THREE.InstancedBufferAttribute;
  private predAnim!: THREE.InstancedBufferAttribute;
  private particles!: Particles;
  private ring!: THREE.Mesh;
  private ringT = 0;
  private fan!: THREE.Mesh;
  private skyMat: THREE.ShaderMaterial;
  private sun: THREE.DirectionalLight;
  private hemi: THREE.HemisphereLight;
  private stars: THREE.Points;
  private fireflies!: THREE.Points;
  private clouds = new THREE.Group();
  private refugeRing!: THREE.Mesh;
  private progress = 0;
  private shown = { pos: new THREE.Vector3(...SHOTS[0].pos), target: new THREE.Vector3(...SHOTS[0].target), elev: SHOTS[0].elev };
  private clock = new THREE.Clock();
  private raf = 0;
  private disposed = false;
  private loopIndex = -1;
  private lastGrassTick = 0;
  private dive: { t0: number; from: THREE.Vector3; fromT: THREE.Vector3; done: () => void } | null = null;
  private rnd = mulberry(2026);
  private births: { a: Animal; t: number }[] = [];
  private nextBirth = 3;

  constructor(private host: HTMLElement, quality: "high" | "low") {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = quality === "high";
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, quality === "high" ? 2 : 1));
    host.appendChild(this.renderer.domElement);

    // sky dome, stars, light
    this.skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false,
      uniforms: { uTop: { value: C("#3a7bd5") }, uHor: { value: C("#d7ecf7") }, uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSun: { value: C("#fff") } },
      vertexShader: `varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `varying vec3 vDir; uniform vec3 uTop; uniform vec3 uHor; uniform vec3 uSunDir; uniform vec3 uSun;
        void main() { float h = clamp(vDir.y, -1.0, 1.0);
          vec3 c = mix(uHor, uTop, pow(max(h, 0.0), 0.5));
          c = mix(c, uHor * 0.5, smoothstep(0.0, -0.6, h));
          float s = max(dot(vDir, normalize(uSunDir)), 0.0);
          c += uSun * (pow(s, 800.0) * 5.0 + pow(s, 10.0) * 0.35);
          gl_FragColor = vec4(c, 1.0); }`,
    });
    this.scene.add(new THREE.Mesh(new THREE.SphereGeometry(1400, 32, 16), this.skyMat));
    const sp: number[] = [];
    for (let i = 0; i < 1400; i++) {
      const u = this.rnd() * 2 - 1, th = this.rnd() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      if (u > 0.05) sp.push(Math.cos(th) * r * 1300, u * 1300, Math.sin(th) * r * 1300);
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
    this.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xdfe8ff, size: 2, sizeAttenuation: false, transparent: true, opacity: 0, depthWrite: false }));
    this.scene.add(this.stars);
    this.hemi = new THREE.HemisphereLight(0xcfe6ff, 0x3a2c1c, 0.9);
    this.sun = new THREE.DirectionalLight(0xfff3dd, 3);
    this.sun.castShadow = quality === "high";
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.3;
    const sc = this.sun.shadow.camera;
    sc.left = sc.bottom = -S * 0.65;
    sc.right = sc.top = S * 0.65;
    sc.far = S * 5;
    this.scene.add(this.hemi, this.sun, this.sun.target);
    this.scene.fog = new THREE.Fog(0xd7ecf7, 250, 1100);

    if (quality === "high") {
      const c = new EffectComposer(this.renderer);
      c.addPass(new RenderPass(this.scene, this.camera));
      this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.45, 0.6, 0.86);
      c.addPass(this.bloom);
      c.addPass(new OutputPass());
      this.composer = c;
    }
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.camera.position.copy(this.shown.pos);
    this.camera.lookAt(this.shown.target);
  }

  /** Steps reported while the scene is built, for the loading screen. */
  static readonly STEPS = ["Shaping the island", "Growing the grass", "Placing refuges and trees", "Waking the animals", "Lighting the sky"];

  /** Build the scene in stages, yielding between them so the loading screen can animate and show progress. */
  async build(onStep: (done: number) => void) {
    const tick = () => new Promise<void>((r) => setTimeout(r, 0));
    this.buildIsland();
    onStep(1);
    await tick();
    this.buildGrass();
    onStep(2);
    await tick();
    this.buildRefuges();
    this.buildClouds();
    onStep(3);
    await tick();
    this.buildActors();
    onStep(4);
    await tick();
    // compile every shader now, so the first visible frames don't stutter
    this.updateCamera(0, 0);
    await this.renderer.compileAsync(this.scene, this.camera);
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
    onStep(5);
    this.loop();
  }

  /** True while the loading screen covers the scene: nothing is simulated or drawn. */
  paused = true;
  private introFly: { t0: number; ms: number; dFar: number; dHero: number; dir: THREE.Vector3; target: THREE.Vector3 } | null = null;

  /**
   * The loading screen hands over: start far out along the hero shot's line of sight, at the distance where the
   * island looks as big as the loader's island (`frac` = its on-screen radius / half the viewport height), and
   * fly in over ms. Returns how much the island grows, so the loader can grow its island in step.
   */
  reveal(frac: number, ms: number): number {
    const s = SHOTS[0];
    const target = new THREE.Vector3(...s.target), hero = new THREE.Vector3(...s.pos);
    const dir = hero.clone().sub(target);
    const dHero = dir.length();
    dir.normalize();
    const dFar = Math.max(dHero * 1.2, (S * 0.62) / (Math.max(0.05, frac) * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))));
    this.introFly = { t0: performance.now(), ms, dFar, dHero, dir, target };
    this.shown.target.copy(target);
    this.shown.pos.copy(target).addScaledVector(dir, dFar);
    this.camera.position.copy(this.shown.pos);
    this.camera.lookAt(target);
    this.paused = false;
    this.clock.getDelta(); // don't let the paused time count as one long frame
    return dFar / dHero;
  }

  private buildActors() {
    // animals
    const preyMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.9 });
    const predMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.75 });
    animateMaterial(preyMat, "prey", this.time);
    animateMaterial(predMat, "predator", this.time);
    const pg = preyGeometry(), qg = predatorGeometry();
    const nPrey = HERD + 1 + 4, nPred = WOLVES + 1;
    this.preyAnim = new THREE.InstancedBufferAttribute(new Float32Array(nPrey * 4), 4);
    this.predAnim = new THREE.InstancedBufferAttribute(new Float32Array(nPred * 4), 4);
    pg.setAttribute("aAnim", this.preyAnim);
    qg.setAttribute("aAnim", this.predAnim);
    this.preyMesh = new THREE.InstancedMesh(pg, preyMat, nPrey);
    this.predMesh = new THREE.InstancedMesh(qg, predMat, nPred);
    for (const m of [this.preyMesh, this.predMesh]) {
      m.castShadow = true;
      m.frustumCulled = false;
      this.scene.add(m);
    }
    const mk = (x: number, y: number, scale: number): Animal => ({
      x, y, h: this.rnd() * Math.PI * 2, spd: 0, spdS: 0, gait: 0, pose: 0, poseT: 0, scale, seed: this.rnd(), timer: this.rnd() * 3,
    });
    for (let i = 0; i < HERD; i++) {
      const a = this.rnd() * Math.PI * 2, d = Math.sqrt(this.rnd()) * 13;
      this.herd.push(mk(29 + Math.cos(a) * d, 36 + Math.sin(a) * d, 1.05));
    }
    this.herd[0].x = 29;
    this.herd[0].y = 36;
    for (let i = 0; i < WOLVES; i++) this.wolves.push(mk(54 + this.rnd() * 16, 14 + this.rnd() * 14, 1.05));
    this.chaseHare = mk(36, 26, 1.05);
    this.chaseWolf = mk(30, 20, 1.1);

    this.particles = new Particles(this.scene);
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.8, 1.1, 40).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xffa060, transparent: true, depthWrite: false, toneMapped: false }));
    this.ring.visible = false;
    this.scene.add(this.ring);
    this.fan = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    }));
    this.fan.renderOrder = 4;
    this.scene.add(this.fan);
    this.refugeRing = new THREE.Mesh(new THREE.RingGeometry(REFUGE_R - 0.25, REFUGE_R + 0.25, 72).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    const [rx, ry] = REFUGE;
    this.refugeRing.position.set(rx - S / 2, this.H(rx, ry) + 0.6, S / 2 - ry);
    this.scene.add(this.refugeRing);

    // fireflies over the meadow at night
    const ff = 260, fp = new Float32Array(ff * 3), fs = new Float32Array(ff);
    for (let i = 0; i < ff; i++) {
      const x = this.rnd() * S, y = this.rnd() * S;
      fp.set([x - S / 2, this.H(x, y) + 0.6, S / 2 - y], i * 3);
      fs[i] = this.rnd();
    }
    const fg = new THREE.BufferGeometry();
    fg.setAttribute("position", new THREE.BufferAttribute(fp, 3));
    fg.setAttribute("aSeed", new THREE.BufferAttribute(fs, 1));
    this.fireflies = new THREE.Points(fg, new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      uniforms: { uTime: this.time.uTime, uNight: { value: 0 } },
      vertexShader: `attribute float aSeed; uniform float uTime; uniform float uNight; varying float vA;
        void main() { float s = aSeed * 1000.0; vec3 p = position;
          p.x += sin(uTime * 0.31 + s) * 2.0 + sin(uTime * 0.11 + s * 1.7) * 3.0;
          p.z += cos(uTime * 0.27 + s * 1.3) * 2.0 + cos(uTime * 0.09 + s) * 3.0;
          p.y += 1.0 + sin(uTime * 0.8 + s * 2.1) * 0.8;
          vA = uNight * (0.35 + 0.65 * pow(0.5 + 0.5 * sin(uTime * (1.3 + aSeed) + s), 3.0));
          vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_PointSize = 140.0 / -mv.z; gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying float vA; void main() { vec2 d = gl_PointCoord - 0.5; float r = dot(d, d);
          float a = vA * smoothstep(0.25, 0.0, r); gl_FragColor = vec4(vec3(1.0, 0.92, 0.45) * a * 2.2, a); }`,
    }));
    this.fireflies.frustumCulled = false;
    this.scene.add(this.fireflies);

  }

  // ---------------------------------------------------------------------------------------- build
  private buildIsland() {
    const H = this.H;
    this.groundTex.magFilter = this.groundTex.minFilter = THREE.LinearFilter;
    this.groundTex.colorSpace = THREE.SRGBColorSpace;
    const seg = 96;
    const geo = new THREE.PlaneGeometry(S, S, seg, seg).rotateX(-Math.PI / 2);
    const p = geo.getAttribute("position");
    for (let i = 0; i < p.count; i++) p.setY(i, H(p.getX(i) + S / 2, S / 2 - p.getZ(i)));
    geo.computeVertexNormals();
    const ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: this.groundTex, roughness: 1 }));
    ground.receiveShadow = true;
    this.scene.add(ground);

    // grass field: smooth noise, a few grazed patches
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x = (i + 0.5) / G, y = (j + 0.5) / G;
        const n = 0.55 + 0.25 * Math.sin(x * 9.1 + 1.3) * Math.cos(y * 7.3 + 0.4) + 0.15 * Math.sin((x + y) * 17.0);
        this.grass[i * G + j] = Math.min(1, Math.max(0.05, n));
      }
    }
    // cut sides: soil and rock strata
    const depth = S * 0.16;
    const bands: [number, string][] = [[0, "#4a3523"], [-0.8, "#5b4129"], [-1.4, "#6e4a2c"], [-depth * 0.35, "#8a5a36"],
      [-depth * 0.4, "#6d6259"], [-depth * 0.7, "#5a524c"], [-depth * 0.74, "#4a4440"], [-depth, "#3a3532"]];
    const pos: number[] = [], col: number[] = [];
    const c = new THREE.Color();
    const edges: [number, number, number, number][] = [[0, 0, 1, 0], [S, 0, 0, 1], [S, S, -1, 0], [0, S, 0, -1]];
    for (const [x0, y0, dx, dy] of edges) {
      for (let k = 0; k < seg; k++) {
        const xa = x0 + (dx * S * k) / seg, ya = y0 + (dy * S * k) / seg;
        const xb = x0 + (dx * S * (k + 1)) / seg, yb = y0 + (dy * S * (k + 1)) / seg;
        const ha = H(Math.min(xa, S - 1e-3), Math.min(ya, S - 1e-3)), hb = H(Math.min(xb, S - 1e-3), Math.min(yb, S - 1e-3));
        for (let b = 0; b < bands.length - 1; b++) {
          const za = b === 0 ? ha : bands[b][0], zb = b === 0 ? hb : bands[b][0], z2 = bands[b + 1][0];
          // jagged lower edge so the island looks broken off, not cut
          const jag = b === bands.length - 2 ? -Math.abs(Math.sin(k * 1.7 + x0) * depth * 0.35) : 0;
          pos.push(xa - S / 2, za, S / 2 - ya, xa - S / 2, z2 + jag, S / 2 - ya, xb - S / 2, zb, S / 2 - yb,
            xb - S / 2, zb, S / 2 - yb, xa - S / 2, z2 + jag, S / 2 - ya, xb - S / 2, z2 + jag, S / 2 - yb);
          c.set(bands[b][1]).convertSRGBToLinear();
          const j = 0.93 + 0.07 * Math.sin(k * 3.7 + b);
          for (let v = 0; v < 6; v++) col.push(c.r * j, c.g * j, c.b * j);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.computeVertexNormals();
    this.scene.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 1, side: THREE.DoubleSide })));
    // underside: an inverted rocky cone
    const under = new THREE.Mesh(new THREE.ConeGeometry(S * 0.62, S * 0.55, 7, 3).rotateX(Math.PI).rotateY(0.3),
      new THREE.MeshStandardMaterial({ color: 0x3e3833, flatShading: true, roughness: 1 }));
    under.position.y = -depth - S * 0.27;
    under.scale.set(1, 1, 1);
    this.scene.add(under);
  }

  private buildGrass() {
    const H = this.H, cs = S / G, per = 5, n = G * G * per;
    const w = 0.42 * cs, maxH = 0.62 * cs;
    const blade = new THREE.BufferGeometry();
    const ys = [0, 0.35, 0.7, 1], ws = [1, 0.8, 0.45, 0];
    const v: number[] = [], hv: number[] = [];
    for (let k = 0; k < 3; k++) {
      const y0 = ys[k], y1 = ys[k + 1], w0 = (ws[k] * w) / 2, w1 = (ws[k + 1] * w) / 2;
      v.push(-w0, y0, 0, w0, y0, 0, w1, y1, 0, -w0, y0, 0, w1, y1, 0, -w1, y1, 0);
      hv.push(y0, y0, y1, y0, y1, y1);
    }
    blade.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    blade.setAttribute("aH", new THREE.Float32BufferAttribute(hv, 1));
    blade.computeVertexNormals();
    this.grassAttr = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    this.grassAttr.setUsage(THREE.DynamicDrawUsage);
    const seeds = new THREE.InstancedBufferAttribute(new Float32Array(n), 1);
    blade.setAttribute("aG", this.grassAttr);
    blade.setAttribute("aSeed", seeds);
    const mat = new THREE.MeshLambertMaterial({ side: THREE.DoubleSide });
    mat.onBeforeCompile = (sh) => {
      sh.uniforms.uTime = this.time.uTime;
      sh.uniforms.uMaxH = { value: maxH };
      sh.vertexShader = sh.vertexShader.replace("#include <common>", `#include <common>
        attribute float aG; attribute float aSeed; attribute float aH; uniform float uTime; uniform float uMaxH;
        varying float vG; varying float vH;`).replace("#include <begin_vertex>", `#include <begin_vertex>
        float gv = sqrt(clamp(aG, 0.0, 1.0)); vG = gv; vH = aH;
        float hgt = uMaxH * (0.15 + 0.85 * gv) * (0.6 + 0.8 * aSeed);
        transformed.y *= hgt;
        vec3 ip = vec3(instanceMatrix[3][0], 0.0, instanceMatrix[3][2]);
        float wave = sin(dot(ip.xz, vec2(0.21, 0.13)) - uTime * 1.7 + aSeed * 2.0) * 0.6 + sin(dot(ip.xz, vec2(-0.09, 0.24)) - uTime * 0.9) * 0.4;
        transformed.x += aH * aH * (0.4 + 0.3 * wave) * hgt;`);
      sh.fragmentShader = sh.fragmentShader.replace("#include <common>", `#include <common>
        varying float vG; varying float vH;`).replace("vec4 diffuseColor = vec4( diffuse, opacity );", `
        vec3 dry = vec3(0.40, 0.32, 0.09), mid = vec3(0.18, 0.38, 0.06), lush = vec3(0.07, 0.33, 0.05);
        vec3 base = vG < 0.45 ? mix(dry, mid, vG / 0.45) : mix(mid, lush, (vG - 0.45) / 0.55);
        base *= mix(0.45, 1.25, vH); base += vec3(0.11, 0.13, 0.0) * vH * vH * vG;
        vec4 diffuseColor = vec4( base, opacity );`);
    };
    const mesh = new THREE.InstancedMesh(blade, mat, n);
    mesh.frustumCulled = false;
    mesh.receiveShadow = true;
    const r = mulberry(77), m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    this.grassCell = new Int32Array(n);
    let k = 0;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        for (let b = 0; b < per; b++, k++) {
          const x = (i + r()) * cs, y = (j + r()) * cs;
          e.set(0, r() * Math.PI * 2, (r() - 0.5) * 0.3);
          q.setFromEuler(e);
          m.compose(new THREE.Vector3(x - S / 2, H(x, y) - 0.05, S / 2 - y), q, new THREE.Vector3(1, 1, 1));
          mesh.setMatrixAt(k, m);
          seeds.setX(k, r());
          this.grassCell[k] = i * G + j;
        }
      }
    }
    this.scene.add(mesh);
    this.pushGrass();
  }

  private inRefuge(x: number, y: number, pad = 0) {
    return Math.hypot(x - REFUGE[0], y - REFUGE[1]) < REFUGE_R + pad || Math.hypot(x - 14, y - 16) < 6 + pad;
  }

  private buildRefuges() {
    const H = this.H, r = mulberry(4242);
    const rocks: THREE.Matrix4[] = [], rockCol: THREE.Color[] = [], bushes: THREE.Matrix4[] = [], bushCol: THREE.Color[] = [];
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
    const put = (list: THREE.Matrix4[], x: number, y: number, s: THREE.Vector3, sink: number) => {
      e.set(r() * 0.6, r() * Math.PI * 2, r() * 0.6);
      q.setFromEuler(e);
      list.push(m.compose(new THREE.Vector3(x - S / 2, H(x, y) - sink, S / 2 - y), q, s).clone());
    };
    for (const [cx, cy, rad] of [[REFUGE[0], REFUGE[1], REFUGE_R], [14, 16, 6]] as const) {
      const nr = 11;
      for (let i = 0; i < nr; i++) {
        const a = (i / nr) * Math.PI * 2 + r() * 0.4, d = rad * (0.8 + r() * 0.25);
        // leave a gap facing the hunt lane so the hare can run in
        if (cx === REFUGE[0] && Math.cos(a - 3.8) > 0.8) continue;
        const s = 0.9 + r() * 1.4;
        put(rocks, cx + Math.cos(a) * d, cy + Math.sin(a) * d, new THREE.Vector3(s * (1 + r()), s * (0.7 + r() * 0.7), s * (1 + r())), 0.25 * s);
        rockCol.push(new THREE.Color().setHSL(0.58 + r() * 0.06, 0.06 + r() * 0.06, 0.42 + r() * 0.16));
      }
      for (let i = 0; i < 6; i++) {
        const a = r() * Math.PI * 2, d = rad * Math.sqrt(r()) * 0.7, s = 1.1 + r() * 1.1;
        put(bushes, cx + Math.cos(a) * d, cy + Math.sin(a) * d, new THREE.Vector3(s, s * (0.75 + r() * 0.35), s), 0.2 * s);
        bushCol.push(new THREE.Color().setHSL(0.27 + r() * 0.08, 0.45 + r() * 0.2, 0.2 + r() * 0.1));
      }
    }
    // scattered trees for scale
    const trunks: THREE.Matrix4[] = [], crowns: THREE.Matrix4[] = [], crownCol: THREE.Color[] = [];
    for (const [x, y] of [[6, 70], [12, 74], [72, 72], [76, 64], [70, 6], [40, 4], [4, 44], [62, 34]] as [number, number][]) {
      const s = 0.9 + r() * 0.6;
      trunks.push(m.compose(new THREE.Vector3(x - S / 2, H(x, y) + 1.4 * s, S / 2 - y), new THREE.Quaternion(), new THREE.Vector3(0.35 * s, 2.8 * s, 0.35 * s)).clone());
      for (let k = 0; k < 3; k++) {
        crowns.push(m.compose(new THREE.Vector3(x - S / 2, H(x, y) + (3.2 + k * 1.3) * s, S / 2 - y), new THREE.Quaternion(),
          new THREE.Vector3((2.2 - k * 0.55) * s, 1.6 * s, (2.2 - k * 0.55) * s)).clone());
        crownCol.push(new THREE.Color().setHSL(0.3 + r() * 0.05, 0.45, 0.2 + k * 0.04));
      }
    }
    const mk = (geo: THREE.BufferGeometry, list: THREE.Matrix4[], cols: THREE.Color[] | THREE.Color) => {
      const im = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.92 }), list.length);
      list.forEach((mm, i) => {
        im.setMatrixAt(i, mm);
        im.setColorAt(i, Array.isArray(cols) ? cols[i] : cols);
      });
      im.castShadow = im.receiveShadow = true;
      this.scene.add(im);
    };
    mk(new THREE.DodecahedronGeometry(1, 0), rocks, rockCol);
    mk(new THREE.IcosahedronGeometry(1, 1), bushes, bushCol);
    mk(new THREE.CylinderGeometry(1, 1.2, 1, 6), trunks, new THREE.Color("#5b3d27"));
    mk(new THREE.ConeGeometry(1, 1, 7), crowns, crownCol);
  }

  private buildClouds() {
    const cr = mulberry(31);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, flatShading: true, transparent: true, opacity: 0.93 });
    for (let i = 0; i < 12; i++) {
      const g = new THREE.Group();
      const parts = 4 + Math.floor(cr() * 4);
      for (let k = 0; k < parts; k++) {
        const m = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), mat);
        m.scale.set(3 + cr() * 4, 2 + cr() * 2.5, 2.5 + cr() * 3.5);
        m.position.set(k * 3.6 - parts * 1.8 + cr() * 2, cr() * 1.4, cr() * 3.5 - 1.7);
        g.add(m);
      }
      const ang = cr() * Math.PI * 2, rad = S * (0.75 + cr() * 0.6);
      g.position.set(Math.cos(ang) * rad, -S * (0.05 + cr() * 0.45) + (cr() > 0.65 ? S * 0.4 : 0), Math.sin(ang) * rad);
      g.userData = { ang, rad, speed: 0.01 + cr() * 0.015 };
      this.clouds.add(g);
    }
    this.scene.add(this.clouds);
  }

  private pushGrass() {
    const a = this.grassAttr.array as Float32Array;
    for (let k = 0; k < a.length; k++) a[k] = this.grass[this.grassCell[k]];
    this.grassAttr.needsUpdate = true;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const v = Math.sqrt(this.grass[i * G + j]);
        const o = (j * G + i) * 4;
        // bare soil -> straw -> meadow
        const t = Math.min(1, v / 0.75);
        this.groundData[o] = 112 + (70 - 112) * t;
        this.groundData[o + 1] = 88 + (132 - 88) * t;
        this.groundData[o + 2] = 58 + (44 - 58) * t;
        this.groundData[o + 3] = 255;
        const x = (i + 0.5) * (S / G), y = (j + 0.5) * (S / G);
        if (this.inRefuge(x, y)) {
          this.groundData[o] = this.groundData[o] * 0.6 + 34;
          this.groundData[o + 1] = this.groundData[o + 1] * 0.6 + 36;
          this.groundData[o + 2] = this.groundData[o + 2] * 0.6 + 34;
        }
      }
    }
    this.groundTex.needsUpdate = true;
  }

  // ---------------------------------------------------------------------------------------- public
  /** Chapter position: 0 = hero, 1 = first chapter, ... fractional between chapters. */
  setProgress(c: number) {
    this.glide = null;
    this.progress = Math.max(0, Math.min(SHOTS.length - 1, c));
  }

  /** Fly to chapter i over ms milliseconds (ease in-out), passing through the shots in between. */
  goTo(i: number, ms: number) {
    const to = Math.max(0, Math.min(SHOTS.length - 1, i));
    if (ms <= 0) {
      this.setProgress(to);
      return;
    }
    this.glide = { from: this.progress, to, t0: performance.now(), ms };
  }
  private glide: { from: number; to: number; t0: number; ms: number } | null = null;

  /** Swoop down into the island, then call done (the page switches to the simulation). */
  diveIn(done: () => void) {
    this.dive = { t0: performance.now(), from: this.camera.position.clone(), fromT: this.shown.target.clone(), done };
  }

  resize() {
    if (this.disposed) return;
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    // keep the island framed on tall phone screens
    this.camera.fov = w / h < 0.8 ? 58 : 40;
    this.camera.updateProjectionMatrix();
    this.composer?.setSize(w, h);
    if (this.particles) (this.particles.points.material as THREE.ShaderMaterial).uniforms.uScale.value = h * 0.9;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.composer?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---------------------------------------------------------------------------------------- behaviour
  private wander(a: Animal, dt: number, cx: number, cy: number, radius: number, maxSpd: number, graze: boolean) {
    a.timer -= dt;
    if (a.timer <= 0) {
      // new intent: graze in place, or hop to a nearby spot inside the home range
      if (graze && this.rnd() < 0.55) {
        a.goal = undefined;
        a.timer = 1.5 + this.rnd() * 3;
      } else {
        let gx = 0, gy = 0;
        for (let k = 0; k < 8; k++) {
          const ang = this.rnd() * Math.PI * 2, d = Math.sqrt(this.rnd()) * radius;
          gx = cx + Math.cos(ang) * d;
          gy = cy + Math.sin(ang) * d;
          if (!this.inRefuge(gx, gy, 1)) break;
        }
        a.goal = [gx, gy];
        a.timer = 2 + this.rnd() * 3;
      }
    }
    if (a.goal) {
      const dx = a.goal[0] - a.x, dy = a.goal[1] - a.y, d = Math.hypot(dx, dy);
      if (d < 0.4) {
        a.goal = undefined;
        a.spd = 0;
      } else {
        this.steer(a, Math.atan2(dy, dx), dt, 3.2);
        a.spd = Math.min(maxSpd, d * 0.8);
      }
    } else a.spd = 0;
    a.x += Math.cos(a.h) * a.spd * dt * 4;
    a.y += Math.sin(a.h) * a.spd * dt * 4;
  }

  private steer(a: Animal, want: number, dt: number, rate: number) {
    let d = want - a.h;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    a.h += Math.max(-rate * dt, Math.min(rate * dt, d));
  }

  /** The scripted hunt. Odd loops the wolf catches the hare in the open; even loops the hare reaches the refuge. */
  private hunt(t: number) {
    const T = 7.5;
    const loop = Math.floor(t / T), u = (t % T) / T;
    const caught = loop % 2 === 1;
    // lane: from the meadow edge to the refuge gap
    const start: [number, number] = [34, 22], gap: [number, number] = [REFUGE[0] - 4.8, REFUGE[1] - 4.2], inside: [number, number] = [REFUGE[0] - 0.5, REFUGE[1] + 0.3];
    const along = (p: number): [number, number] => {
      // gentle S-curve between start and the gap
      const x = start[0] + (gap[0] - start[0]) * p + Math.sin(p * Math.PI * 2) * 2.6;
      const y = start[1] + (gap[1] - start[1]) * p;
      return [x, y];
    };
    const hare = this.chaseHare, wolf = this.chaseWolf;
    if (loop !== this.loopIndex) {
      this.loopIndex = loop;
      hare.scale = 1.05;
    }
    let hp: [number, number], wp: [number, number], hs = 1, ws = 1;
    const escapeEnd = 0.68;
    if (u < 0.1) {
      // calm before: hare grazes, wolf stalks in
      hp = along(0);
      wp = along(-0.22 + u * 0.6);
      hs = 0;
      ws = 0.4;
    } else if (!caught) {
      const p = Math.min(1, (u - 0.1) / (escapeEnd - 0.1));
      hp = p < 1 ? along(p) : [gap[0] + (inside[0] - gap[0]) * Math.min(1, (u - escapeEnd) / 0.08), gap[1] + (inside[1] - gap[1]) * Math.min(1, (u - escapeEnd) / 0.08)];
      const wpp = Math.min(0.86, -0.16 + p * 1.02);
      wp = along(wpp);
      hs = u < escapeEnd + 0.08 ? 1 : 0;
      ws = wpp < 0.86 ? 1 : 0.15;
      hare.poseT = u > escapeEnd + 0.08 ? 0.8 : 0;
      wolf.poseT = 0;
    } else {
      const catchAt = 0.5;
      const p = Math.min(catchAt, (u - 0.1) / (escapeEnd - 0.1));
      hp = along(p);
      const wpp = Math.min(p - 0.02, -0.16 + ((u - 0.1) / (escapeEnd - 0.1)) * 1.36);
      wp = along(Math.max(-0.2, wpp));
      if (u - 0.1 >= catchAt * (escapeEnd - 0.1) && hare.scale > 0) {
        // the catch
        const [x, y] = hp, X = x - S / 2, Z = S / 2 - y, Y = this.H(x, y);
        this.particles.emit(X, Y + 0.6, Z, 28, { color: C("#ff7a45"), speed: 6, up: 4.5, life: 0.9, size: 0.8, grav: 9 });
        this.particles.emit(X, Y + 0.6, Z, 14, { color: C("#f1dfb8"), speed: 3.5, up: 3, life: 1.3, size: 0.6, grav: 3 });
        this.ring.position.set(X, Y + 0.15, Z);
        this.ringT = 0.7;
        hare.scale = 0;
      }
      hs = 1;
      ws = hare.scale > 0 ? 1 : 0;
      wolf.poseT = hare.scale > 0 ? 0 : 1; // lies down to digest
      hare.poseT = 0;
    }
    const prevH = [hare.x, hare.y], prevW = [wolf.x, wolf.y];
    [hare.x, hare.y] = hp;
    [wolf.x, wolf.y] = wp;
    const mv = (a: Animal, px: number, py: number, s: number) => {
      const dx = a.x - px, dy = a.y - py;
      if (Math.hypot(dx, dy) > 0.01) a.h = Math.atan2(dy, dx);
      a.spd = s;
    };
    mv(hare, prevH[0], prevH[1], hs);
    mv(wolf, prevW[0], prevW[1], ws);
    if (!caught && u > 0.5) this.steer(wolf, Math.atan2(REFUGE[1] - wolf.y, REFUGE[0] - wolf.x), 0.05, 1);
  }

  private tickGrass(dt: number) {
    // regrow (logistic) everywhere, grazed where hares stand still
    for (let c = 0; c < this.grass.length; c++) {
      const g = this.grass[c];
      this.grass[c] = Math.min(1, g + 0.05 * dt * g * (1 - g) + 0.002 * dt);
    }
    for (const a of this.herd) {
      if (a.spd > 0.05) continue;
      const i = Math.min(G - 1, Math.max(0, Math.floor((a.x / S) * G))), j = Math.min(G - 1, Math.max(0, Math.floor((a.y / S) * G)));
      this.grass[i * G + j] = Math.max(0.03, this.grass[i * G + j] - 0.35 * dt);
    }
    this.pushGrass();
  }

  // ---------------------------------------------------------------------------------------- frame
  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    if (this.paused) return;
    const dt = Math.min(0.05, this.clock.getDelta());
    const t = (this.time.uTime.value += dt);

    for (const a of this.herd) this.wander(a, dt, 29, 36, 14, 0.75, true);
    for (const a of this.wolves) this.wander(a, dt, 62, 20, 10, 0.55, false);
    this.hunt(t);
    this.updateBirths(dt);
    if (t - this.lastGrassTick > 0.2) {
      this.tickGrass(t - this.lastGrassTick);
      this.lastGrassTick = t;
    }
    this.drawAnimals(dt);
    this.particles.update(dt);
    if (this.ringT > 0) {
      this.ringT -= dt;
      const p = 1 - this.ringT / 0.7;
      this.ring.visible = this.ringT > 0;
      this.ring.scale.setScalar(1 + p * 5);
      (this.ring.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.8;
    }
    for (const c of this.clouds.children) {
      const u = c.userData as { ang: number; rad: number; speed: number };
      u.ang += u.speed * dt;
      c.position.x = Math.cos(u.ang) * u.rad;
      c.position.z = Math.sin(u.ang) * u.rad;
    }
    this.updateCamera(dt, t);
    this.updateOverlays(t);
    if (this.composer) this.composer.render();
    else this.renderer.render(this.scene, this.camera);
  };

  private updateBirths(dt: number) {
    this.nextBirth -= dt;
    if (this.nextBirth <= 0) {
      this.nextBirth = 4 + this.rnd() * 3;
      const parent = this.herd[1 + Math.floor(this.rnd() * (HERD - 1))];
      const free = this.births.length < 4;
      if (free) {
        const a: Animal = { ...parent, scale: 0.01, goal: undefined, timer: 2, seed: this.rnd() };
        a.x += Math.cos(parent.h + 2) * 1.2;
        a.y += Math.sin(parent.h + 2) * 1.2;
        this.births.push({ a, t: 0 });
        const X = a.x - S / 2, Z = S / 2 - a.y, Y = this.H(a.x, a.y);
        this.particles.emit(X, Y + 0.8, Z, 14, { color: C("#ffe08a"), speed: 1.2, up: 2.4, life: 1.4, size: 0.5, grav: -0.6, spread: 1 });
      }
    }
    for (const b of this.births) {
      b.t += dt;
      b.a.scale = Math.min(0.75, 0.75 * Math.min(1, b.t / 0.6)) * (b.t > 9 ? Math.max(0, 1 - (b.t - 9) / 0.6) : 1);
      this.wander(b.a, dt, 29, 36, 14, 0.7, true);
    }
    this.births = this.births.filter((b) => b.t < 9.6);
  }

  private drawAnimals(dt: number) {
    const H = this.H;
    const put = (mesh: THREE.InstancedMesh, anim: THREE.InstancedBufferAttribute, list: Animal[], freq: number) => {
      const arr = mesh.instanceMatrix.array as Float32Array, an = anim.array as Float32Array;
      for (let i = 0; i < mesh.count; i++) {
        const a = list[i];
        const o = i * 16;
        if (!a || a.scale <= 0.001) {
          arr.fill(0, o, o + 16);
          continue;
        }
        a.spdS += (a.spd - a.spdS) * Math.min(1, dt * 8);
        a.pose += (a.poseT - a.pose) * Math.min(1, dt * 3);
        a.gait += a.spdS * freq * dt * (1 + 0.15 * a.seed);
        const sc = a.scale, c = Math.cos(a.h) * sc, sn = Math.sin(a.h) * sc;
        arr[o] = c; arr[o + 1] = 0; arr[o + 2] = -sn; arr[o + 3] = 0;
        arr[o + 4] = 0; arr[o + 5] = sc; arr[o + 6] = 0; arr[o + 7] = 0;
        arr[o + 8] = sn; arr[o + 9] = 0; arr[o + 10] = c; arr[o + 11] = 0;
        arr[o + 12] = a.x - S / 2; arr[o + 13] = H(a.x, a.y) - 0.05; arr[o + 14] = S / 2 - a.y; arr[o + 15] = 1;
        an[i * 4] = a.gait;
        an[i * 4 + 1] = a.spdS;
        an[i * 4 + 2] = a.pose;
        an[i * 4 + 3] = a.seed;
      }
      mesh.instanceMatrix.needsUpdate = true;
      anim.needsUpdate = true;
    };
    put(this.preyMesh, this.preyAnim, [...this.herd, this.chaseHare, ...this.births.map((b) => b.a)], 9);
    put(this.predMesh, this.predAnim, [...this.wolves, this.chaseWolf], 7.5);
  }

  /** Weight (0..1) of how much chapter i is on screen right now. */
  private weight(i: number) {
    return Math.max(0, 1 - Math.abs(this.progress - i));
  }

  private updateOverlays(t: number) {
    // refuge outline glows in the refuges chapter
    (this.refugeRing.material as THREE.MeshBasicMaterial).opacity = this.weight(CH("refuges")) * (0.55 + 0.25 * Math.sin(t * 3));
    // vision fan around the focus hare in the vision chapter
    const w = this.weight(CH("vision"));
    const mat = this.fan.material as THREE.MeshBasicMaterial;
    mat.opacity = w * 0.6;
    this.fan.visible = w > 0.01;
    if (!this.fan.visible) return;
    const me = this.herd[0], R = 16, cone = (240 * Math.PI) / 180, range = 20;
    const X = me.x - S / 2, Z = S / 2 - me.y, Y = this.H(me.x, me.y) + 0.45;
    const others: { x: number; y: number; kind: number }[] = [
      ...this.herd.slice(1).map((a) => ({ x: a.x, y: a.y, kind: 1 })),
      ...this.wolves.map((a) => ({ x: a.x, y: a.y, kind: 2 })),
      { x: this.chaseWolf.x, y: this.chaseWolf.y, kind: 2 },
    ];
    const palette = [C("#9be15d"), C("#ffe7b0"), C("#ff6a3d"), C("#a9c6ff"), C("#ffffff")];
    const pos: number[] = [], col: number[] = [];
    for (let r = 0; r < R; r++) {
      const a0 = me.h - cone / 2 + (r / R) * cone, a1 = me.h - cone / 2 + ((r + 1) / R) * cone;
      let best = range, kind = 4;
      for (const o of others) {
        const dx = o.x - me.x, dy = o.y - me.y, d = Math.hypot(dx, dy);
        if (d > range || d < 0.01) continue;
        let b = Math.atan2(dy, dx) - me.h;
        while (b > Math.PI) b -= 2 * Math.PI;
        while (b < -Math.PI) b += 2 * Math.PI;
        const lo = -cone / 2 + (r / R) * cone, hi = lo + cone / R;
        if (b >= lo && b < hi && d - 0.5 < best) {
          best = d - 0.5;
          kind = o.kind;
        }
      }
      if (kind === 4) {
        // first tall grass (or the refuge) along the sector's centre line
        const ac = (a0 + a1) / 2;
        for (let s = 0; s < 8; s++) {
          const d = (s + 0.5) * (range / 8), x = me.x + Math.cos(ac) * d, y = me.y + Math.sin(ac) * d;
          if (x < 0 || y < 0 || x >= S || y >= S) break;
          if (this.inRefuge(x, y)) { best = d; kind = 3; break; }
          const g = this.grass[Math.floor((x / S) * G) * G + Math.floor((y / S) * G)];
          if (g > 0.5) { best = d; kind = 0; break; }
        }
      }
      const len = Math.max(0.8, best);
      pos.push(X, Y, Z, X + Math.cos(a0) * len, Y, Z - Math.sin(a0) * len, X + Math.cos(a1) * len, Y, Z - Math.sin(a1) * len);
      const c = palette[kind], f = kind === 4 ? 0.2 : 0.95;
      for (let v = 0; v < 3; v++) col.push(c.r * f, c.g * f, c.b * f);
    }
    const g = this.fan.geometry;
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.computeBoundingSphere();
  }

  private updateCamera(dt: number, t: number) {
    if (this.introFly) {
      // apparent size (1/distance) follows the same ease-in-out curve as the loader's island
      const f = this.introFly;
      const u = Math.min(1, (performance.now() - f.t0) / f.ms);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      const inv = 1 / f.dFar + (1 / f.dHero - 1 / f.dFar) * e;
      this.shown.pos.copy(f.target).addScaledVector(f.dir, 1 / inv);
      this.shown.target.copy(f.target);
      this.camera.position.copy(this.shown.pos);
      this.camera.lookAt(f.target);
      this.applySky(this.shown.elev);
      if (u >= 1) this.introFly = null;
      return;
    }
    if (this.glide) {
      const g = this.glide;
      const u = Math.min(1, (performance.now() - g.t0) / g.ms);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      this.progress = g.from + (g.to - g.from) * e;
      if (u >= 1) this.glide = null;
    }
    // a glide already eases in and out, so between shots it moves linearly (no stop at each shot on long jumps)
    const c = this.progress, i = Math.floor(c), f = this.glide ? c - i : smooth(c - i);
    const a = SHOTS[i], b = SHOTS[Math.min(SHOTS.length - 1, i + 1)];
    const pos = new THREE.Vector3(...a.pos).lerp(new THREE.Vector3(...b.pos), f);
    const target = new THREE.Vector3(...a.target).lerp(new THREE.Vector3(...b.target), f);
    // slow drift in the hero and at the end so the island always feels alive
    const drift = Math.max(this.weight(CH("hero")), this.weight(CH("enter")), this.weight(CH("methods")) * 0.4);
    if (drift > 0) {
      const ang = Math.sin(t * 0.07) * 0.35 * drift;
      const rel = pos.clone().sub(target);
      rel.applyAxisAngle(new THREE.Vector3(0, 1, 0), ang);
      pos.copy(target).add(rel);
    }
    const elev = a.elev + (b.elev - a.elev) * f;
    const k = 1 - Math.exp(-dt * (this.glide ? 10 : 4));
    if (this.dive) {
      const d = this.dive;
      const u = Math.min(1, (performance.now() - d.t0) / 1600);
      const e = u * u * (3 - 2 * u);
      const goal = new THREE.Vector3(-10, 2.2, 6), goalT = new THREE.Vector3(-14, 0.6, 2);
      this.camera.position.lerpVectors(d.from, goal, e);
      this.shown.target.lerpVectors(d.fromT, goalT, e);
      this.camera.lookAt(this.shown.target);
      if (u >= 1) {
        const done = d.done;
        this.dive = null;
        done();
      }
    } else {
      this.shown.pos.lerp(pos, k);
      this.shown.target.lerp(target, k);
      this.camera.position.copy(this.shown.pos);
      this.camera.lookAt(this.shown.target);
    }
    this.shown.elev += (elev - this.shown.elev) * k;
    this.applySky(this.shown.elev);
  }

  private applySky(elev: number) {
    const s = skyAt(elev);
    const ang = Math.asin(Math.max(-1, Math.min(1, elev)));
    const dir = new THREE.Vector3(Math.cos(ang) * 0.85, Math.max(0.06, elev), 0.5).normalize();
    this.sun.position.copy(dir).multiplyScalar(S * 2);
    this.sun.color.copy(s.sun);
    this.sun.intensity = Math.max(s.sunI, 0.25) * 1.15;
    // generous sky fill so the island's shaded sides never go black
    this.hemi.intensity = Math.max(s.hemi, 0.45) * 1.25;
    this.hemi.color.copy(s.top).lerp(C("#ffffff"), 0.5);
    this.skyMat.uniforms.uTop.value.copy(s.top);
    this.skyMat.uniforms.uHor.value.copy(s.hor);
    this.skyMat.uniforms.uSunDir.value.set(Math.cos(ang), elev, 0.5);
    this.skyMat.uniforms.uSun.value.copy(s.sun).multiplyScalar(elev > -0.05 ? 1 : 0);
    (this.scene.fog as THREE.Fog).color.copy(s.hor);
    const night = Math.min(1, Math.max(0, (0.08 - elev) / 0.2));
    (this.stars.material as THREE.PointsMaterial).opacity = night * 0.9;
    (this.fireflies.material as THREE.ShaderMaterial).uniforms.uNight.value = night;
    this.renderer.toneMappingExposure = s.exp + night * 0.15;
    if (this.bloom) this.bloom.strength = 0.3 + night * 0.6;
  }
}

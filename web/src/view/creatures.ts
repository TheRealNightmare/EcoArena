// Low-poly prey (hare-like) and predator (wolf-like) built from primitives, with per-vertex animation tags.
//
// Every vertex carries: colour, a part id (0 body, 1 front leg, 2 back leg, 3 ear, 4 tail, 5 head), a pivot
// (the joint the part swings about) and a leg phase. The vertex shader in Diorama.ts swings legs, bobs the body,
// flicks ears and wags tails from these, so thousands of animals animate on the GPU without skeletons.
import * as THREE from "three";

export const PART = { body: 0, front: 1, back: 2, ear: 3, tail: 4, head: 5 } as const;

interface Piece {
  geo: THREE.BufferGeometry;
  color: string;
  part: number;
  pivot: [number, number, number];
  phase: number;
}

function piece(geo: THREE.BufferGeometry, at: [number, number, number], color: string, part: number,
  opts: { scale?: [number, number, number]; rot?: [number, number, number]; pivot?: [number, number, number]; phase?: number } = {}): Piece {
  const g = geo.toNonIndexed();
  if (opts.scale) g.scale(...opts.scale);
  if (opts.rot) {
    g.rotateX(opts.rot[0]);
    g.rotateY(opts.rot[1]);
    g.rotateZ(opts.rot[2]);
  }
  g.translate(...at);
  return { geo: g, color, part, pivot: opts.pivot ?? at, phase: opts.phase ?? 0 };
}

function merge(pieces: Piece[]): THREE.BufferGeometry {
  let n = 0;
  for (const p of pieces) n += p.geo.getAttribute("position").count;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const part = new Float32Array(n);
  const pivot = new Float32Array(n * 3);
  const phase = new Float32Array(n);
  let o = 0;
  const c = new THREE.Color();
  for (const p of pieces) {
    const a = p.geo.getAttribute("position");
    c.set(p.color).convertSRGBToLinear();
    for (let i = 0; i < a.count; i++, o++) {
      pos.set([a.getX(i), a.getY(i), a.getZ(i)], o * 3);
      // a little per-vertex shade variation reads as fur/facets under flat shading
      const k = 0.92 + 0.08 * Math.sin(o * 12.9898);
      col.set([c.r * k, c.g * k, c.b * k], o * 3);
      part[o] = p.part;
      pivot.set(p.pivot, o * 3);
      phase[o] = p.phase;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setAttribute("aPart", new THREE.BufferAttribute(part, 1));
  g.setAttribute("aPivot", new THREE.BufferAttribute(pivot, 3));
  g.setAttribute("aLegPhase", new THREE.BufferAttribute(phase, 1));
  g.computeVertexNormals();
  return g;
}

const ico = (r: number, d = 1) => new THREE.IcosahedronGeometry(r, d);
const cone = (r: number, h: number, s = 6) => new THREE.ConeGeometry(r, h, s);
const cyl = (r0: number, r1: number, h: number, s = 6) => new THREE.CylinderGeometry(r0, r1, h, s);

/** Hare: faces +X, about 1.7 long. Hops: back legs drive together, front legs together. */
export function preyGeometry(): THREE.BufferGeometry {
  const fur = "#e6d6b8", belly = "#fbf4e6", dark = "#8a6a4a", pink = "#e7a49a";
  const P: Piece[] = [
    piece(ico(0.55), [0, 0.68, 0], fur, PART.body, { scale: [1.35, 0.92, 0.88] }),
    piece(ico(0.42), [-0.28, 0.6, 0], fur, PART.body, { scale: [1.1, 1, 1.05] }),
    piece(ico(0.38), [0.1, 0.5, 0], belly, PART.body, { scale: [1.3, 0.6, 0.8] }),
    piece(ico(0.34), [0.68, 1.02, 0], fur, PART.head, { scale: [1.15, 0.98, 0.92], pivot: [0.45, 0.9, 0] }),
    piece(ico(0.07, 0), [0.99, 1.0, 0], pink, PART.head, { pivot: [0.45, 0.9, 0] }),
    piece(ico(0.06, 0), [0.86, 1.1, 0.17], "#1d1410", PART.head, { pivot: [0.45, 0.9, 0] }),
    piece(ico(0.06, 0), [0.86, 1.1, -0.17], "#1d1410", PART.head, { pivot: [0.45, 0.9, 0] }),
    piece(cone(0.1, 0.66, 5), [0.52, 1.5, 0.12], fur, PART.ear, { rot: [0.25, 0, 0.42], pivot: [0.6, 1.2, 0.1] }),
    piece(cone(0.1, 0.66, 5), [0.52, 1.5, -0.12], fur, PART.ear, { rot: [-0.25, 0, 0.42], pivot: [0.6, 1.2, -0.1] }),
    piece(cone(0.06, 0.4, 4), [0.5, 1.6, 0.13], dark, PART.ear, { rot: [0.25, 0, 0.42], pivot: [0.6, 1.2, 0.1] }),
    piece(cone(0.06, 0.4, 4), [0.5, 1.6, -0.13], dark, PART.ear, { rot: [-0.25, 0, 0.42], pivot: [0.6, 1.2, -0.1] }),
    piece(ico(0.17, 0), [-0.82, 0.78, 0], "#f6efe2", PART.tail, { pivot: [-0.72, 0.75, 0] }),
    piece(cyl(0.07, 0.06, 0.48), [0.38, 0.24, 0.17], fur, PART.front, { pivot: [0.38, 0.48, 0.17], phase: 0 }),
    piece(cyl(0.07, 0.06, 0.48), [0.38, 0.24, -0.17], fur, PART.front, { pivot: [0.38, 0.48, -0.17], phase: 0.4 }),
    piece(new THREE.BoxGeometry(0.62, 0.22, 0.2), [-0.38, 0.12, 0.24], belly, PART.back, { pivot: [-0.3, 0.45, 0.24], phase: Math.PI }),
    piece(new THREE.BoxGeometry(0.62, 0.22, 0.2), [-0.38, 0.12, -0.24], belly, PART.back, { pivot: [-0.3, 0.45, -0.24], phase: Math.PI + 0.4 }),
    piece(ico(0.3), [-0.36, 0.42, 0.24], fur, PART.back, { scale: [1.2, 1, 0.6], pivot: [-0.3, 0.5, 0.24], phase: Math.PI }),
    piece(ico(0.3), [-0.36, 0.42, -0.24], fur, PART.back, { scale: [1.2, 1, 0.6], pivot: [-0.3, 0.5, -0.24], phase: Math.PI + 0.4 }),
  ];
  return merge(P);
}

/** Wolf: faces +X, about 3.2 long. Trots: diagonal legs move together. */
export function predatorGeometry(): THREE.BufferGeometry {
  const coat = "#b8532c", back = "#7e3a22", cream = "#ecd7b6", dark = "#2c201a";
  const P: Piece[] = [
    piece(ico(0.6), [-0.1, 1.08, 0], coat, PART.body, { scale: [1.85, 0.82, 0.78] }),
    piece(ico(0.5), [-0.15, 1.38, 0], back, PART.body, { scale: [1.7, 0.4, 0.62] }),
    piece(ico(0.52), [0.62, 1.16, 0], coat, PART.body, { scale: [1, 1.08, 0.95] }),
    piece(ico(0.4), [0.7, 0.92, 0], cream, PART.body, { scale: [0.9, 0.8, 0.8] }),
    piece(ico(0.38), [1.22, 1.5, 0], coat, PART.head, { scale: [1.12, 0.95, 0.9], pivot: [0.9, 1.35, 0] }),
    piece(cone(0.19, 0.58, 6), [1.66, 1.4, 0], cream, PART.head, { rot: [0, 0, -Math.PI / 2], pivot: [0.9, 1.35, 0] }),
    piece(ico(0.07, 0), [1.95, 1.4, 0], dark, PART.head, { pivot: [0.9, 1.35, 0] }),
    piece(ico(0.06, 0), [1.42, 1.62, 0.18], "#f2c94c", PART.head, { pivot: [0.9, 1.35, 0] }),
    piece(ico(0.06, 0), [1.42, 1.62, -0.18], "#f2c94c", PART.head, { pivot: [0.9, 1.35, 0] }),
    piece(cone(0.13, 0.36, 4), [1.12, 1.92, 0.18], back, PART.ear, { rot: [0.2, 0, 0.1], pivot: [1.12, 1.75, 0.18] }),
    piece(cone(0.13, 0.36, 4), [1.12, 1.92, -0.18], back, PART.ear, { rot: [-0.2, 0, 0.1], pivot: [1.12, 1.75, -0.18] }),
    piece(cone(0.17, 1.05, 6), [-1.45, 1.05, 0], back, PART.tail, { rot: [0, 0, 1.95], pivot: [-1.05, 1.25, 0] }),
    piece(cone(0.1, 0.3, 6), [-1.92, 0.86, 0], cream, PART.tail, { rot: [0, 0, 1.95], pivot: [-1.05, 1.25, 0] }),
    piece(cyl(0.1, 0.07, 0.9), [0.7, 0.45, 0.22], coat, PART.front, { pivot: [0.7, 0.9, 0.22], phase: 0 }),
    piece(cyl(0.1, 0.07, 0.9), [0.7, 0.45, -0.22], coat, PART.front, { pivot: [0.7, 0.9, -0.22], phase: Math.PI }),
    piece(cyl(0.12, 0.07, 0.9), [-0.78, 0.45, 0.22], coat, PART.back, { pivot: [-0.78, 0.92, 0.22], phase: Math.PI }),
    piece(cyl(0.12, 0.07, 0.9), [-0.78, 0.45, -0.22], coat, PART.back, { pivot: [-0.78, 0.92, -0.22], phase: 0 }),
    piece(ico(0.09, 0), [0.72, 0.03, 0.22], dark, PART.front, { pivot: [0.7, 0.9, 0.22], phase: 0 }),
    piece(ico(0.09, 0), [0.72, 0.03, -0.22], dark, PART.front, { pivot: [0.7, 0.9, -0.22], phase: Math.PI }),
    piece(ico(0.09, 0), [-0.76, 0.03, 0.22], dark, PART.back, { pivot: [-0.78, 0.92, 0.22], phase: Math.PI }),
    piece(ico(0.09, 0), [-0.76, 0.03, -0.22], dark, PART.back, { pivot: [-0.78, 0.92, -0.22], phase: 0 }),
  ];
  return merge(P);
}

/**
 * Shader patch shared by both species. Instance attribute aAnim = (gait phase, speed 0..1, pose 0..1, seed).
 * pose 0 = standing/moving, 1 = lying down (predator digesting, prey hiding in a refuge).
 */
export function animateMaterial(mat: THREE.Material, kind: "prey" | "predator", uniforms: { uTime: { value: number } }) {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = uniforms.uTime;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
        attribute float aPart;
        attribute vec3 aPivot;
        attribute float aLegPhase;
        attribute vec4 aAnim;
        uniform float uTime;
        vec3 rotZ(vec3 p, vec3 o, float a) { p -= o; float c = cos(a), s = sin(a); return vec3(c*p.x - s*p.y, s*p.x + c*p.y, p.z) + o; }
        vec3 rotY(vec3 p, vec3 o, float a) { p -= o; float c = cos(a), s = sin(a); return vec3(c*p.x + s*p.z, p.y, -s*p.x + c*p.z) + o; }
        vec3 rotX(vec3 p, vec3 o, float a) { p -= o; float c = cos(a), s = sin(a); return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z) + o; }`,
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
        {
          float gait = aAnim.x, spd = aAnim.y, pose = aAnim.z, seed = aAnim.w;
          float t = uTime + seed * 17.0;
          bool isPrey = ${kind === "prey" ? "true" : "false"};
          float legAmp = (isPrey ? 0.9 : 0.75) * spd * (1.0 - pose);
          if (aPart == 1.0 || aPart == 2.0) {
            float swing = sin(gait + aLegPhase) * legAmp;
            transformed = rotZ(transformed, aPivot, swing);
            // fold legs under the body when lying down
            transformed.y = mix(transformed.y, aPivot.y - (aPivot.y - transformed.y) * 0.25, pose);
          }
          if (aPart == 3.0) {
            float flick = sin(t * 3.1) * 0.08 + sin(gait) * 0.18 * spd;
            transformed = rotX(transformed, aPivot, flick * sign(transformed.z + 0.0001));
            transformed = rotZ(transformed, aPivot, isPrey ? 0.35 * pose : 0.0);
          }
          if (aPart == 4.0) {
            float wag = isPrey ? sin(gait * 2.0) * 0.25 * spd : sin(t * (4.0 + 6.0 * spd)) * 0.35;
            transformed = rotY(transformed, aPivot, wag);
          }
          if (aPart == 5.0) {
            // head nods with the gait; heads go down to eat / rest
            float nod = sin(gait * 2.0) * 0.08 * spd + pose * 0.45 + (1.0 - spd) * sin(t * 0.7) * 0.06;
            transformed = rotZ(transformed, aPivot, -nod);
          }
          // whole-body motion: prey hop, predators bob; lying down lowers and breathes
          float hop = isPrey ? max(0.0, sin(gait)) * 0.42 * spd : abs(sin(gait)) * 0.1 * spd;
          float breathe = sin(t * 2.2) * 0.025 * (0.4 + pose);
          transformed.y += hop * (1.0 - pose);
          if (aPart == 0.0 || aPart == 5.0 || aPart == 3.0 || aPart == 4.0) transformed.y *= 1.0 + breathe;
          transformed.y -= pose * (isPrey ? 0.22 : 0.48);
          // stretch into a lunge at full speed
          transformed.x *= 1.0 + 0.12 * smoothstep(0.75, 1.0, spd) * (1.0 - pose);
        }`,
      );
  };
}

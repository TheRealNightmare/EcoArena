import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import Loader from "../loader/Loader";
import { CHAPTERS, StoryScene } from "./StoryScene";
import "./landing.css";

// ------------------------------------------------------------------------------------------ findings data
// Benchmark results: every method controls both species, 5 training seeds x 150k world-steps each, evaluated on
// 5 fresh worlds x 10,000 steps in each of 5 scenarios. Source: results/bench/STATS.md in the repo.
type Bar = { label: string; v: number; tone?: "rl" | "evo" | "base" | "ppo" | "es" | "ga" | "neat" | "hybrid" };
// share of worlds where both species were still alive after 10,000 steps (S1-S5 pooled; 125 worlds per method)
const SURVIVAL: Bar[] = [
  { label: "Random", v: 92, tone: "base" },
  { label: "ES", v: 77, tone: "evo" },
  { label: "Hybrid", v: 64, tone: "evo" },
  { label: "Hand-coded", v: 64, tone: "base" },
  { label: "PPO", v: 22, tone: "rl" },
  { label: "GA", v: 10, tone: "evo" },
  { label: "NEAT", v: 0, tone: "evo" },
];
// S1: mean predator lifespan in steps (mean of 5 training seeds)
const PRED_LIFE: Bar[] = [
  { label: "PPO", v: 1196, tone: "rl" },
  { label: "Random", v: 1072, tone: "base" },
  { label: "ES", v: 1059, tone: "evo" },
  { label: "Hand-coded", v: 1056, tone: "base" },
  { label: "Hybrid", v: 1055, tone: "evo" },
  { label: "GA", v: 782, tone: "evo" },
  { label: "NEAT", v: 633, tone: "evo" },
];

function BarChart({ id, bars, max, unit, decimals = 1, labelW = 86 }: { id: string; bars: Bar[]; max: number; unit: string; decimals?: number; labelW?: number }) {
  const rowH = 26, W = 360, H = bars.length * rowH + 22;
  const ticks = [0, max / 4, max / 2, (3 * max) / 4, max];
  return (
    <svg className="bars" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={`${id}-t`}>
      <title id={`${id}-t`}>{bars.map((b) => `${b.label} ${b.v}`).join(", ")} {unit}</title>
      {ticks.map((t) => {
        const x = labelW + ((W - labelW - 44) * t) / max;
        return (
          <g key={t}>
            <line x1={x} x2={x} y1={0} y2={H - 18} className="grid" />
            <text x={x} y={H - 4} className="tick" textAnchor="middle">{+t.toFixed(2)}</text>
          </g>
        );
      })}
      {bars.map((b, i) => {
        const y = i * rowH + 4, w = ((W - labelW - 44) * b.v) / max;
        return (
          <g key={b.label} className={`bar ${b.tone}`} style={{ ["--i" as string]: i }}>
            <text x={labelW - 8} y={y + 13} textAnchor="end" className="lab">{b.label}</text>
            <rect x={labelW} y={y + 2} width={w} height={rowH - 10} rx={4} />
            <text x={labelW + w + 6} y={y + 13} className="val">{b.v.toFixed(decimals)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Survival (% of 25 test worlds with both species alive after 10,000 steps) per scenario, from results/bench/STATS.md
const SCEN = ["S1", "S2", "S3", "S4", "S5"] as const;
const SCEN_NAME = ["Baseline", "Slow grass", "Fast prey", "Crowded start", "Many predators"];
const BY_SCENARIO: { label: string; tone: Bar["tone"]; all: number; s: number[] }[] = [
  { label: "Random", tone: "base", all: 92, s: [100, 60, 100, 100, 100] },
  { label: "ES", tone: "evo", all: 77, s: [80, 60, 84, 80, 80] },
  { label: "Hybrid", tone: "evo", all: 64, s: [88, 40, 20, 76, 96] },
  { label: "Hand-coded", tone: "base", all: 64, s: [100, 100, 100, 0, 20] },
  { label: "PPO", tone: "rl", all: 22, s: [80, 0, 0, 4, 24] },
  { label: "GA", tone: "evo", all: 10, s: [4, 0, 8, 8, 28] },
  { label: "NEAT", tone: "evo", all: 0, s: [0, 0, 0, 0, 0] },
];

function Heatmap() {
  return (
    <div className="heat" role="table" aria-label="Survival by scenario, percent of worlds">
      <div className="heat-row head" role="row">
        <span role="columnheader">Method</span>
        {SCEN.map((c, i) => <span key={c} role="columnheader" title={SCEN_NAME[i]}><b>{c}</b><small>{SCEN_NAME[i]}</small></span>)}
        <span role="columnheader"><b>All</b><small>125 worlds</small></span>
      </div>
      {BY_SCENARIO.map((r) => (
        <div key={r.label} className={`heat-row ${r.tone}`} role="row">
          <span role="rowheader" className="name">{r.label}</span>
          {[...r.s, r.all].map((v, i) => (
            <span key={i} role="cell" className={`cell mono ${i === 5 ? "total" : ""}`} style={{ ["--v" as string]: v / 100 }}>{v}%</span>
          ))}
        </div>
      ))}
    </div>
  );
}

// Median predators per world over time, S1 (25 worlds per method), sampled every 250 steps.
// Source: results/bench/eval/S1/*.pops.npy
const PRED_SERIES: { label: string; key: string; d: number[] }[] = [
  { label: "PPO", key: "ppo", d: [121, 161, 199, 241, 244, 245, 255, 260, 263, 269, 269, 266, 277, 268, 269, 262, 264, 263, 263, 251, 241, 236, 205, 212, 228, 225, 234, 251, 252, 250, 263, 264, 268, 268, 269, 269, 264, 269, 271, 267] },
  { label: "ES", key: "es", d: [121, 161, 194, 214, 188, 169, 159, 148, 139, 134, 134, 124, 123, 114, 121, 120, 111, 114, 115, 112, 118, 122, 125, 116, 121, 123, 114, 114, 111, 113, 118, 113, 116, 121, 121, 120, 122, 120, 109, 118] },
  { label: "Hybrid", key: "hybrid", d: [121, 161, 192, 211, 187, 166, 153, 143, 130, 128, 128, 118, 114, 117, 101, 101, 99, 104, 99, 94, 93, 93, 84, 87, 79, 87, 85, 88, 88, 80, 79, 82, 77, 71, 80, 68, 73, 69, 67, 60] },
  { label: "GA", key: "ga", d: [121, 142, 146, 132, 95, 68, 55, 43, 34, 28, 23, 21, 16, 13, 10, 7, 6, 5, 5, 4, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { label: "NEAT", key: "neat", d: [121, 127, 110, 82, 46, 27, 18, 11, 7, 4, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
];

function PredatorLines() {
  const W = 380, H = 200, L = 34, R = 56, T = 10, B = 26;
  const X = (i: number) => L + ((W - L - R) * i) / 39, Y = (v: number) => T + (H - T - B) * (1 - v / 300);
  return (
    <svg className="lines" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label="Median predators per world over 10,000 steps: PPO about 265, ES about 118, Hybrid falls to 60, GA reaches 0 by step 6,500, NEAT by step 3,250">
      {[0, 100, 200, 300].map((v) => (
        <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="grid" /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text></g>
      ))}
      {[0, 2, 4, 6, 8, 10].map((k) => <text key={k} x={X((k * 1000) / 250)} y={H - 8} textAnchor="middle" className="tick">{k}k</text>)}
      {PRED_SERIES.map((s) => (
        <g key={s.key} className={`ln ${s.key}`}>
          <polyline points={s.d.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")} />
          {s.key === "ga" ? <text x={X(10) + 6} y={Y(s.d[10]) - 6}>GA</text>
            : s.key === "neat" ? <text x={X(4) - 7} y={Y(s.d[4]) + 4} textAnchor="end">NEAT</text>
            : <text x={W - R + 6} y={Y(s.d[39]) + 4}>{s.label}</text>}
        </g>
      ))}
    </svg>
  );
}

// Median prey still alive at the moment the last predator died, over every collapsed test world (all scenarios)
const PREY_AT_END: Bar[] = [
  { label: "ES", v: 571, tone: "es" },
  { label: "PPO", v: 565, tone: "ppo" },
  { label: "GA", v: 539, tone: "ga" },
  { label: "Hybrid", v: 539, tone: "hybrid" },
  { label: "NEAT", v: 504, tone: "neat" },
  { label: "Hand-coded", v: 48, tone: "base" },
];

function Dumbbell() {
  // PPO with the old +1 birth bonus vs the same run without it
  const rows = [
    { label: "Prey: caught per encounter", from: 11.4, to: 8.2, max: 16, better: "lower" },
    { label: "Predators: catches per 1k steps", from: 5.3, to: 7.45, max: 10, better: "higher" },
  ];
  const W = 420, left = 16, right = W - 16;
  return (
    <div className="dumbbells">
      {rows.map((r) => {
        const X = (v: number) => left + ((right - left) * v) / r.max;
        return (
          <figure key={r.label} className="db">
            <figcaption>{r.label} <span className="dim">({r.better} is better)</span></figcaption>
            <svg viewBox={`0 0 ${W} 58`} role="img" aria-label={`${r.label}: ${r.from} with the bonus, ${r.to} without`}>
              <line x1={left} x2={right} y1={28} y2={28} className="axis" />
              <line x1={X(r.from)} x2={X(r.to)} y1={28} y2={28} className="span" />
              <circle cx={X(r.from)} cy={28} r={7} className="from" />
              <circle cx={X(r.to)} cy={28} r={8} className="to" />
              <text x={X(r.from)} y={11} textAnchor="middle" className="val dim">{r.from} with bonus</text>
              <text x={X(r.to)} y={54} textAnchor="middle" className="val strong">{r.to} without</text>
            </svg>
          </figure>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------------------------------ study 2 data
// Each trained brain's predators in the baseline world: [thrust squared, catches per 1,000 predator-steps,
// died out (1) or survived (0), method]. 42 brains: the benchmark's 27 plus the movement-cost runs.
// Source: results/study/leak_model.json
type Pt = [number, number, 0 | 1, string];
const LEAK: Pt[] = [
  [0.333, 6.46, 0, "base"], [0.197, 6.14, 0, "base"],
  [0.022, 5.49, 0, "ppo"], [0.056, 5.62, 0, "ppo"], [0.054, 4.98, 1, "ppo"], [0.053, 5.69, 0, "ppo"], [0.029, 5.4, 0, "ppo"],
  [0.569, 6.53, 1, "ga"], [0.619, 6.76, 1, "ga"], [0.642, 6.65, 1, "ga"], [0.431, 5.88, 1, "ga"], [0.549, 6.38, 1, "ga"],
  [0.234, 6.06, 0, "es"], [0.161, 5.53, 0, "es"], [0.173, 5.7, 0, "es"], [0.16, 4.56, 1, "es"], [0.176, 5.95, 0, "es"],
  [0.568, 5.61, 1, "neat"], [0.995, 7.29, 1, "neat"], [0.644, 6.14, 1, "neat"], [0.694, 6.22, 1, "neat"], [0.768, 6.79, 1, "neat"],
  [0.196, 5.33, 1, "hybrid"], [0.288, 5.89, 0, "hybrid"], [0.256, 5.77, 0, "hybrid"], [0.252, 5.91, 0, "hybrid"], [0.156, 5.82, 0, "hybrid"],
  [0.269, 5.85, 0, "ga"], [0.322, 6.5, 0, "ga"], [0.347, 6.58, 0, "ga"], [0.237, 5.86, 0, "ga"], [0.562, 6.08, 1, "ga"],
  [0.381, 5.53, 1, "neat"], [0.242, 2.99, 1, "neat"], [0.304, 4.55, 1, "neat"], [0.581, 6.53, 1, "neat"], [0.74, 7.15, 1, "neat"],
  [0.065, 5.75, 0, "es"], [0.077, 5.94, 0, "es"], [0.058, 5.74, 0, "es"], [0.063, 5.74, 0, "es"], [0.078, 5.34, 0, "es"],
];
// catch rate at which a predator expects exactly one child in its lifetime (R0 = 1), at thrust squared 0, 0.1, ... 1
const R0_LINE = [4.89, 5.37, 5.69, 6.09, 6.57, 6.89, 7.29, 7.77, 8.05, 8.49, 8.9];

function LeakScatter() {
  const W = 400, H = 262, L = 40, R = 12, T = 12, B = 38, y0 = 2.5, y1 = 7.8;
  const X = (v: number) => L + (W - L - R) * v, Y = (v: number) => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
  // the R0 = 1 line, cut where it leaves the top of the plot
  const line: string[] = [];
  for (let i = 0; i < R0_LINE.length; i++) {
    const v = R0_LINE[i];
    if (v <= y1) line.push(`${X(i / 10).toFixed(1)},${Y(v).toFixed(1)}`);
    else {
      const p = R0_LINE[i - 1], f = (y1 - p) / (v - p);
      line.push(`${X((i - 1 + f) / 10).toFixed(1)},${Y(y1).toFixed(1)}`);
      break;
    }
  }
  return (
    <svg className="scatter" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label="Each trained brain's predators by wasted movement and catch rate. Those below the break-even line died out; those above it survived.">
      <polygon className="doom" points={`${X(0)},${Y(y0)} ${line.join(" ")} ${X(1)},${Y(y1)} ${X(1)},${Y(y0)}`} />
      {[3, 4, 5, 6, 7].map((v) => (
        <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="grid" /><text x={L - 6} y={Y(v) + 4} textAnchor="end" className="tick">{v}</text></g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => <text key={v} x={X(v)} y={H - B + 14} textAnchor="middle" className="tick">{v}</text>)}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="axis-label">energy wasted on movement (thrust²)</text>
      <text x={11} y={(T + H - B) / 2} textAnchor="middle" className="axis-label" transform={`rotate(-90 11 ${(T + H - B) / 2})`}>catches per 1,000 steps</text>
      <polyline className="r0-line" points={line.join(" ")} />
      <text x={X(0.98)} y={Y(2.85)} textAnchor="end" className="zone">can't replace themselves</text>
      <text x={X(0.02)} y={Y(7.45)} className="zone ok">can replace themselves</text>
      {LEAK.map(([th, c, dead, m], i) => (
        <circle key={i} cx={X(th)} cy={Y(c)} r={4.2} className={`pt ${m} ${dead ? "dead" : "alive"}`} style={{ ["--i" as string]: i }} />
      ))}
    </svg>
  );
}

// How clearly selection can see the energy that coasting saves, against the luck of catches, by fitness window.
// Source: results/study/energy_signal.json
const SIGNAL: Bar[] = [
  { label: "128 steps", v: 0.24, tone: "ga" },
  { label: "512 steps", v: 0.59, tone: "ga" },
  { label: "1,024 steps", v: 0.93, tone: "ga" },
];

// Share of test worlds with both species alive, before and after one change to training (S1, S2 and S4 pooled:
// 75 worlds per bar, 5 training seeds). Source: results/bench/eval and results/study/eval
const FIXES: { label: string; what: string; from: number; to: number; ok: boolean }[] = [
  { label: "ES", what: "prey learn 4× slower", from: 73, to: 100, ok: true },
  { label: "ES", what: "movement costs 5×", from: 73, to: 88, ok: true },
  { label: "GA", what: "movement costs 5×", from: 4, to: 72, ok: true },
  { label: "PPO", what: "sees 1,000 steps ahead", from: 28, to: 72, ok: true },
  { label: "PPO", what: "prey learn 4× slower", from: 28, to: 51, ok: true },
  { label: "PPO", what: "paid for a healthy world", from: 28, to: 35, ok: false },
  { label: "NEAT", what: "movement costs 5×", from: 0, to: 4, ok: false },
  { label: "PPO", what: "predators learn 4× slower", from: 28, to: 0, ok: false },
];

function Slopes() {
  const W = 210, left = 26, right = W - 36;
  const X = (v: number) => left + ((right - left) * v) / 100;
  return (
    <div className="slopes" role="table" aria-label="Survival before and after each change to training">
      {FIXES.map((r, i) => (
        <div key={i} className={`slope ${r.ok ? "ok" : "no"}`} role="row" style={{ ["--i" as string]: i }}>
          <span role="rowheader" className="who"><b>{r.label}</b>{r.what}</span>
          <svg viewBox={`0 0 ${W} 26`} role="cell" aria-label={`${r.from}% before, ${r.to}% after`}>
            <line x1={left} x2={right} y1={13} y2={13} className="axis" />
            <line x1={X(r.from)} x2={X(r.to)} y1={13} y2={13} className="span" />
            <circle cx={X(r.from)} cy={13} r={4.5} className="from" />
            <circle cx={X(r.to)} cy={13} r={6} className="to" />
            {r.to >= r.from
              ? <><text x={X(r.to) + 10} y={17} className="val">{r.to}%</text><text x={X(r.from) - 8} y={17} textAnchor="end" className="val dim">{r.from}%</text></>
              : <><text x={X(r.to) - 10} y={17} textAnchor="end" className="val">{r.to}%</text><text x={X(r.from) + 8} y={17} className="val dim">{r.from}%</text></>}
          </svg>
        </div>
      ))}
      <p className="slope-key"><span className="k from" />before<span className="k to" />after one change<span className="k no" />did not help</p>
    </div>
  );
}

// Cross-play in the baseline world: prey of one method with predators of another, percent of 25 worlds where
// both species lasted 10,000 steps. Source: results/study/REPORT.md
const XP = ["PPO", "GA", "ES", "NEAT", "Hybrid"];
const XPLAY: number[][] = [ // rows: prey brain; columns: predator brain
  [80, 0, 0, 0, 28],
  [60, 4, 80, 12, 92],
  [88, 0, 80, 4, 100],
  [80, 40, 100, 0, 80],
  [100, 0, 28, 0, 88],
];
// Skill against the other methods' animals, 1.0 = an average method. Source: results/study/OPT.md
const SKILL: Bar[] = [
  { label: "PPO", v: 1.48, tone: "ppo" },
  { label: "Hybrid", v: 1.17, tone: "hybrid" },
  { label: "ES", v: 0.98, tone: "es" },
  { label: "GA", v: 0.8, tone: "ga" },
  { label: "NEAT", v: 0.77, tone: "neat" },
];

function CrossPlay() {
  return (
    <div className="heat xplay" role="table" aria-label="Worlds with both species alive when prey of one method meet predators of another, percent">
      <div className="heat-row head" role="row">
        <span role="columnheader" className="corner">predators →<br />prey ↓</span>
        {XP.map((c) => <span key={c} role="columnheader"><b>{c}</b></span>)}
      </div>
      {XPLAY.map((row, r) => (
        <div key={XP[r]} className="heat-row" role="row">
          <span role="rowheader" className="name">{XP[r]}</span>
          {row.map((v, c) => (
            <span key={c} role="cell" className={`cell mono ${r === c ? "total" : ""}`} style={{ ["--v" as string]: v / 100 }}>{v}%</span>
          ))}
        </div>
      ))}
    </div>
  );
}

// The drag world: share of test worlds with both species alive (S1, S2 and S4 pooled). Source: results/bench_drag/eval
const DRAG: Bar[] = [
  { label: "Hand-coded", v: 100, tone: "base" },
  { label: "ES", v: 3, tone: "es" },
  { label: "PPO", v: 0, tone: "ppo" },
  { label: "GA", v: 0, tone: "ga" },
  { label: "NEAT", v: 0, tone: "neat" },
  { label: "Hybrid", v: 0, tone: "hybrid" },
  { label: "Random", v: 0, tone: "base" },
];

const TEAM = ["Mirazul Islam", "Fairoze Fatema Alam", "Mehejabin Prova", "Afnan Zaman Mohammad Sajid", "Julkar Naim Shishir"];
const BIBTEX = `@misc{ecoarena2026,
  title  = {EcoArena: Learned Predators and Prey in a World That Never Resets},
  author = {Islam, Mirazul and Alam, Fairoze Fatema and Prova, Mehejabin and Sajid, Afnan Zaman Mohammad and Shishir, Julkar Naim},
  year   = {2026},
  note   = {United International University. http://eco.mirazulislamnahid.com/}
}`;

function Cite() {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(BIBTEX).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }).catch(() => undefined);
  };
  return (
    <div className="cite">
      <pre className="mono" tabIndex={0}>{BIBTEX}</pre>
      <button type="button" className="ghost-dark" onClick={copy}>{copied ? "Copied" : "Copy BibTeX"}</button>
    </div>
  );
}

// ------------------------------------------------------------------------------------------ story chapters
const STORY = [
  {
    kicker: "The world",
    title: "An ecosystem that never resets",
    body: "There are no episodes. Grass regrows, prey eat it, predators eat prey, and every animal is born, ages and dies inside one continuous run. Animals appear and vanish mid-run, which most reinforcement-learning tools are not built for, so EcoArena keeps them in fixed slots on the GPU.",
    stats: ["200 × 200 world", "up to 2,400 prey and 600 predators", "10,000-step test runs"],
    side: "left",
  },
  {
    kicker: "Prey",
    title: "Graze, grow, split",
    body: "A prey animal standing on grass gains energy every step. At 70 energy it can give birth, handing 30 to its child. Moving costs energy, and so does simply being alive, so a good prey brain balances eating against running.",
    stats: ["+2 energy a step on rich grass", "breeds at 70 energy", "lives up to 1,500 steps"],
    side: "right",
  },
  {
    kicker: "Predators",
    title: "Hunt, then rest",
    body: "A predator that lunges at prey within 1.5 units catches it half the time. A catch gives 35 energy and 60 steps of digesting; a miss leaves it stunned for 40. Predators can't breed when more than two others are within 15 units, so their numbers can't explode.",
    stats: ["catch radius 1.5", "50% of lunges succeed", "60 steps digesting after a catch"],
    side: "left",
  },
  {
    kicker: "Refuges",
    title: "Rocks where nothing can catch you",
    body: "Forty-eight rocky patches cover 13% of the map. Prey inside can't be caught, but grass there never grows past 0.35, so hiding means going hungry. Before we added them, our test worlds kept collapsing: predators ate every prey, then starved.",
    stats: ["48 refuges", "13% of the map", "grass capped at 0.35 inside"],
    side: "right",
  },
  {
    kicker: "Senses",
    title: "102 numbers in, 2 out",
    body: "Each animal sees through 16 vision sectors: prey across 240°, predators across a narrower 120°. Every sector reports how far away the nearest thing is and what it is: grass, prey, predator, refuge or nothing. Add energy, speed, age, resting, grass underfoot and whether it's in a refuge. The brain answers with how much to turn and how hard to push.",
    stats: ["16 sectors × 6 values", "+ 6 about itself", "→ turn and thrust"],
    side: "left",
  },
] as const;

const METHODS = [
  { name: "PPO", family: "Reinforcement learning", body: "Learns from every step. A critic estimates how good each moment is, and the policy is nudged toward actions that did better than expected.", tag: "22% of worlds survive · largest predator packs" },
  { name: "GA", family: "Genetic algorithm", body: "Fifty networks share the world at once. The five that gather the most reward survive each generation; the rest are mutated copies of them.", tag: "10% survive · most catches, shortest lives" },
  { name: "ES", family: "Evolution strategies", body: "Tries 25 pairs of opposite random nudges to the weights, then moves the network in the direction the whole population's scores point.", tag: "77% survive · the steadiest learner" },
  { name: "NEAT", family: "Evolving topologies", body: "Starts with every input wired straight to the outputs and grows new neurons and connections as it evolves. Our networks grew to about 17 hidden neurons.", tag: "0% survive · predators gone by step 3,000" },
  { name: "Hybrid", family: "Evolution, then RL", body: "Spends half the budget evolving with GA, then hands the best network to PPO to fine-tune for the other half.", tag: "64% survive · second steadiest" },
] as const;

const RULES = [
  { k: "Same reward", v: "energy gained − energy spent, −1 for dying", icon: "reward" },
  { k: "Same brain", v: "102 inputs → 64 → 64 → 2 outputs", icon: "brain" },
  { k: "Same budget", v: "150,000 world-steps of training", icon: "clock" },
  { k: "Same test", v: "5 seeds × 5 worlds × 10,000 steps × 5 scenarios", icon: "flask" },
] as const;

const DEPTH = [
  { name: "PPO", learns: "Every step's reward, through a critic that predicts the future", round: "Collect 64 steps in 4 worlds, then 4 passes of small, clipped updates", knobs: "discount 0.99 · ~590 updates" },
  { name: "GA", learns: "One score per network per generation", round: "50 networks share the world for 128 steps; the top 5 survive, the rest are mutated copies", knobs: "noise 0.02 · 293 generations" },
  { name: "ES", learns: "The scores of 50 nudged copies of one network", round: "25 mirrored pairs of random nudges; step toward the ones that scored higher", knobs: "noise 0.02 · Adam 0.01" },
  { name: "NEAT", learns: "Scores, while also growing the wiring", round: "Species compete among themselves; mutations add neurons and links", knobs: "up to 48 neurons · 5 species" },
  { name: "Hybrid", learns: "GA first, then PPO", round: "75k world-steps of GA, then PPO starts from the best network", knobs: "critic warm-up 20 updates" },
] as const;

// ------------------------------------------------------------------------------------------ chapter icons
function Icon({ name }: { name: string }) {
  const p = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const g = (() => {
    switch (name) {
      case "hero": return <><path d="M3 18h18" /><path d="M5.5 18c1.6-4.4 4-7 6.5-7s4.9 2.6 6.5 7" /><path d="M15 11V6.5m0 0-2.2 2.2M15 6.5l2.2 2.2" /></>;
      case "world": return <><rect x="3.5" y="3.5" width="17" height="17" rx="2.5" /><path d="M3.5 9.2h17M3.5 14.8h17M9.2 3.5v17M14.8 3.5v17" opacity=".55" /></>;
      case "prey": return <><ellipse cx="9.6" cy="6.2" rx="1.5" ry="3.6" /><ellipse cx="14.4" cy="6.2" rx="1.5" ry="3.6" /><circle cx="12" cy="12.6" r="3" /><path d="M6.6 20.5c.4-3 2.6-4.9 5.4-4.9s5 1.9 5.4 4.9" /></>;
      case "predators": return <><path d="M4.5 4 8 9.5h8L19.5 4l-.8 8.6L12 20l-6.7-7.4Z" /><path d="M9.6 12.6h.01M14.4 12.6h.01" strokeWidth={2.4} /></>;
      case "refuges": return <><path d="M2.5 18.5 6 11l4-3.2 5 1 3.5 4.2 3 5.5Z" /><path d="M10 7.8 11.5 13l3.5-4" opacity=".55" /></>;
      case "vision": return <><path d="M2.5 12S6 6 12 6s9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" /><circle cx="12" cy="12" r="2.6" /></>;
      case "methods": return <><circle cx="5.5" cy="7" r="1.8" /><circle cx="5.5" cy="17" r="1.8" /><circle cx="12" cy="12" r="1.8" /><circle cx="18.5" cy="7" r="1.8" /><circle cx="18.5" cy="17" r="1.8" /><path d="M7.1 7.9 10.4 11M7.1 16.1l3.3-3.1M13.6 11l3.3-3.1M13.6 13l3.3 3.1" /></>;
      case "fair": return <><path d="M12 4v16M8 20h8M4.5 7.5h15" /><path d="M6.5 7.5 3.8 13.5h5.4ZM17.5 7.5l-2.7 6h5.4Z" /></>;
      case "findings-prey": return <><path d="M3.5 20.5h17" /><path d="M6.5 20.5V13M12 20.5V5.5M17.5 20.5v-10" /></>;
      case "findings-scenarios": return <><rect x="3.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.5" opacity=".55" /><rect x="3.5" y="13.5" width="7" height="7" rx="1.5" opacity=".55" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.5" /></>;
      case "findings-collapse": return <><path d="M3.5 6.5 9 12l4-3.5 7.5 9" /><path d="M15.5 17.5h5v-5" /></>;
      case "findings-predators": return <><path d="M6.5 3.5h11M6.5 20.5h11" /><path d="M7.5 3.5c0 5 9 5.6 9 8.5s-9 3.5-9 8.5M16.5 3.5c0 5-9 5.6-9 8.5s9 3.5 9 8.5" /></>;
      case "findings-reward": return <><path d="M6 8.5h12.5m0 0-3-3m3 3-3 3M18 15.5H5.5m0 0 3-3m-3 3 3 3" /></>;
      case "why-leak": return <><path d="M12 3.5c3 3.6 5.5 6.4 5.5 9.8a5.5 5.5 0 0 1-11 0c0-3.4 2.5-6.2 5.5-9.8Z" /><path d="M9.5 14.2a2.6 2.6 0 0 0 2.5 2.3" opacity=".55" /></>;
      case "why-window": return <><circle cx="10.5" cy="10.5" r="6" /><path d="m15 15 5.5 5.5" /><path d="M8 10.5h5" opacity=".55" /></>;
      case "fixes": return <><path d="M14.5 5.5a4 4 0 0 0-5.200 5.200L3.5 16.500v4h4l5.800-5.800a4 4 0 0 0 5.200-5.200l-2.7 2.7-2.3-.7-.7-2.3Z" /></>;
      case "arms": return <><path d="M4 4l9 9M4 4v4M4 4h4M20 4l-9 9M20 4v4M20 4h-4" /><path d="M7 17l-3 3M17 17l3 3M9.5 14.5 7 17M14.5 14.5 17 17" opacity=".55" /></>;
      case "drag": return <><path d="M3 8h11a2.5 2.5 0 1 0-2.5-2.5M3 12h15.5a2.5 2.5 0 1 1-2.5 2.5M3 16h8a2.5 2.5 0 1 1-2.5 2.5" /></>;
      case "team": return <><circle cx="9" cy="8" r="3" /><path d="M3.5 19.5c.4-3.200 2.600-5 5.5-5s5.100 1.800 5.5 5" /><path d="M16 5.200a3 3 0 0 1 0 5.600M17.500 14.800c1.7.6 2.800 2.200 3 4.7" opacity=".55" /></>;
      case "enter": return <path d="M8 5.2v13.6L19 12Z" />;
      case "reward": return <><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5v9M8.5 12h7" /></>;
      case "brain": return <><circle cx="6" cy="8" r="1.7" /><circle cx="6" cy="16" r="1.7" /><circle cx="18" cy="12" r="1.7" /><path d="M7.6 8.6 16.4 11.4M7.6 15.4l8.8-2.8" /></>;
      case "clock": return <><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></>;
      case "flask": return <><path d="M9.5 3.5h5M10.5 3.5v5.2L5 18.4c-.6 1.1.2 2.1 1.4 2.1h11.2c1.2 0 2-1 1.4-2.1l-5.5-9.7V3.5" /><path d="M7.4 15h9.2" opacity=".55" /></>;
      default: return <circle cx="12" cy="12" r="3" />;
    }
  })();
  return <svg className="ico" viewBox="0 0 24 24" aria-hidden="true" {...p}>{g}</svg>;
}

// ------------------------------------------------------------------------------------------ page
const LABELS: Record<(typeof CHAPTERS)[number], string> = {
  hero: "Start", world: "The world", prey: "Prey", predators: "Predators", refuges: "Refuges", vision: "Senses",
  methods: "Methods", fair: "A fair fight", "findings-prey": "Survival", "findings-scenarios": "By scenario",
  "findings-collapse": "How it collapses", "findings-predators": "Predator lifespans", "findings-reward": "The reward",
  "why-leak": "The energy leak", "why-window": "Why evolution misses it", fixes: "The fixes", arms: "Arms race",
  drag: "A second world", team: "Paper & team", enter: "Simulation",
};
const N = CHAPTERS.length;
const at = (name: (typeof CHAPTERS)[number]) => CHAPTERS.indexOf(name);

/** Glide length for a jump of `n` chapters (longer jumps fly a little longer). */
const glideMs = (n: number) => Math.min(2200, 1200 + 250 * (Math.abs(n) - 1));

export default function Landing({ onEnter }: { onEnter: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const story = useRef<StoryScene | null>(null);
  const [index, setIndex] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const [loaded, setLoaded] = useState(0);
  const [loading, setLoading] = useState(true);
  const live = useRef({ index: 0, lockedUntil: 0, needQuiet: false, lastWheel: 0, wheelSum: 0, loading: true });
  live.current.loading = loading;

  useEffect(() => {
    let q: "high" | "low" = window.innerWidth < 700 ? "low" : "high";
    try {
      const saved = localStorage.getItem("ecoarena.quality");
      if (saved) q = JSON.parse(saved);
    } catch {
      /* storage unavailable */
    }
    const s = new StoryScene(host.current!, q);
    story.current = s;
    s.build(setLoaded);
    return () => s.dispose();
  }, []);

  /** Fly to chapter i. Gestures that arrive during the flight are ignored. */
  const go = useCallback((i: number) => {
    const L = live.current;
    const to = Math.max(0, Math.min(N - 1, i));
    if (to === L.index) return;
    const ms = glideMs(to - L.index);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    story.current?.goTo(to, reduce ? 0 : ms);
    L.index = to;
    L.lockedUntil = performance.now() + (reduce ? 300 : ms * 0.45); // the camera is a spring, so the next gesture can bend its path
    L.needQuiet = true;
    L.wheelSum = 0;
    setIndex(to);
  }, []);

  const step = useCallback((d: number) => {
    const L = live.current;
    if (L.loading || performance.now() < L.lockedUntil) return;
    go(L.index + d);
  }, [go]);

  // Inner panels that are taller than the screen scroll first; the page moves on at their edge.
  const innerCanScroll = (target: EventTarget | null, dir: number) => {
    let el = target as HTMLElement | null;
    while (el && !el.classList?.contains("slide")) {
      if (el.scrollHeight > el.clientHeight + 2 && getComputedStyle(el).overflowY !== "visible") {
        if (dir > 0 && el.scrollTop + el.clientHeight < el.scrollHeight - 2) return true;
        if (dir < 0 && el.scrollTop > 2) return true;
      }
      el = el.parentElement;
    }
    return false;
  };

  useEffect(() => {
    const L = live.current;
    const onWheel = (e: WheelEvent) => {
      const dir = Math.sign(e.deltaY);
      if (!dir || innerCanScroll(e.target, dir)) return;
      e.preventDefault();
      const now = performance.now();
      // a trackpad keeps firing after the flick: wait for a short pause before the next gesture counts
      if (L.needQuiet) {
        const quiet = now - L.lastWheel > 160;
        L.lastWheel = now;
        if (!quiet || now < L.lockedUntil) return;
        L.needQuiet = false;
      }
      L.lastWheel = now;
      L.wheelSum += e.deltaY;
      if (Math.abs(L.wheelSum) >= 24) step(Math.sign(L.wheelSum));
    };
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const k = e.key;
      let d = 0;
      if (k === "ArrowDown" || k === "PageDown" || (k === " " && !e.shiftKey && tag !== "BUTTON")) d = 1;
      else if (k === "ArrowUp" || k === "PageUp" || (k === " " && e.shiftKey && tag !== "BUTTON")) d = -1;
      else if (k === "Home") return go(0), e.preventDefault();
      else if (k === "End") return go(N - 1), e.preventDefault();
      if (!d) return;
      e.preventDefault();
      step(d);
    };
    let ty = 0, tx = 0, tTarget: EventTarget | null = null;
    const onTouchStart = (e: TouchEvent) => {
      ty = e.touches[0].clientY;
      tx = e.touches[0].clientX;
      tTarget = e.target;
    };
    const onTouchEnd = (e: TouchEvent) => {
      const dy = ty - e.changedTouches[0].clientY, dx = tx - e.changedTouches[0].clientX;
      if (Math.abs(dy) < 50 || Math.abs(dy) < Math.abs(dx) * 1.2) return;
      const dir = Math.sign(dy);
      if (innerCanScroll(tTarget, dir)) return;
      step(dir);
    };
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("keydown", onKey);
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
    };
  }, [go, step]);

  const enter = () => {
    if (leaving) return;
    setLeaving(true);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !story.current) onEnter();
    else story.current.diveIn(onEnter);
  };

  const slide = (i: number, cls: string, children: ReactNode) => (
    <section key={i} className={`slide ${cls} ${i === index ? "active" : i < index ? "past" : "future"}`}
      aria-hidden={i !== index} {...(i !== index ? ({ inert: "" } as Record<string, string>) : {})} aria-label={LABELS[CHAPTERS[i]]}>
      {children}
    </section>
  );

  const caveat = <p className="caveat">Benchmark: every method controls both species; 5 training seeds per method, each tested on 5 fresh worlds for 10,000 steps in 5 scenarios.</p>;
  const worldOn = index >= at("world") && index <= at("vision");
  const findOn = index >= at("findings-prey") && index <= at("findings-reward");
  const methodsOn = index === at("methods") || index === at("fair");
  const whyOn = index >= at("why-leak") && index <= at("drag");
  const study = <p className="caveat">Follow-up study: 5 training seeds per setting, tested on 5 fresh worlds for 10,000 steps in the baseline, slow-grass and crowded scenarios.</p>;

  return (
    <div className={`landing ${leaving ? "leaving" : ""} ${loading ? "" : "revealed"}`}>
      {loading ? (
        <Loader title="EcoArena" steps={StoryScene.STEPS} done={loaded} minMs={2000}
          onReveal={(frac, ms) => story.current?.reveal(frac, ms)} onFinished={() => setLoading(false)} />
      ) : null}
      <div className="story-canvas" ref={host} aria-hidden="true" />
      <div className="veil" aria-hidden="true" />
      <div className="dusk-cover" aria-hidden="true" />
      <div className="progress" aria-hidden="true"><span style={{ transform: `scaleX(${index / (N - 1)})` }} /></div>

      <header className={`lnav ${index > 0 ? "solid" : ""}`}>
        <button type="button" className="wordmark" onClick={() => go(0)}>EcoArena</button>
        <nav aria-label="Sections">
          <button type="button" onClick={() => go(at("world"))} className={worldOn ? "on" : ""}>The world</button>
          <button type="button" onClick={() => go(at("methods"))} className={methodsOn ? "on" : ""}>Methods</button>
          <button type="button" onClick={() => go(at("findings-prey"))} className={findOn ? "on" : ""}>Findings</button>
          <button type="button" onClick={() => go(at("why-leak"))} className={whyOn ? "on" : ""}>Why &amp; fixes</button>
          <button type="button" onClick={() => go(at("team"))} className={index === at("team") ? "on" : ""}>Team</button>
        </nav>
        <button type="button" className="cta small" onClick={enter}>Open the simulation</button>
      </header>

      <ol className="chapters-nav" aria-label="Chapters">
        {CHAPTERS.map((c, i) => (
          <li key={c} className={c === "methods" || c === "findings-prey" || c === "why-leak" || c === "team" || c === "enter" ? "group-start" : ""}>
            <button type="button" className={index === i ? "on" : i < index ? "seen" : ""} aria-current={index === i ? "step" : undefined}
              aria-label={LABELS[c]} onClick={() => go(i)}>
              <span className="lbl" aria-hidden="true">{LABELS[c]}</span>
              <Icon name={c} />
            </button>
          </li>
        ))}
      </ol>

      <main className="slides">
        {slide(0, "hero", (
          <>
            <p className="eyebrow">A benchmark for learning in a living world</p>
            <h1>EcoArena</h1>
            <p className="lede">
              Predators and prey share one world that never resets. We give each species a brain trained by a different
              machine-learning method and ask a simple question: does the ecosystem survive?
            </p>
            <div className="hero-actions">
              <button type="button" className="cta" onClick={enter}>Open the simulation</button>
              <button type="button" className="ghost" onClick={() => go(1)}>See how it works</button>
            </div>
            <p className="scroll-cue" aria-hidden="true"><span />Scroll</p>
          </>
        ))}

        {STORY.map((s, k) => slide(k + 1, `story ${s.side}`, (
          <article className="card glass">
            <p className="kicker"><span className="num mono">{String(k + 1).padStart(2, "0")}</span>{s.kicker}</p>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
            <ul className="stats">
              {s.stats.map((x) => <li key={x}>{x}</li>)}
            </ul>
          </article>
        )))}

        {slide(at("methods"), "methods", (
          <div className="slide-body">
            <div className="section-head glass">
              <p className="kicker">Methods</p>
              <h2>Five ways to grow a brain</h2>
              <p>Every method gets the same network (two layers of 64), the same reward (energy gained, minus a penalty for dying) and the same budget of 150,000 world-steps. Only the way it learns differs.</p>
            </div>
            <div className="method-grid">
              {METHODS.map((m) => (
                <article key={m.name} className="method glass">
                  <p className="family">{m.family}</p>
                  <h3>{m.name}</h3>
                  <p>{m.body}</p>
                  <p className="tag">{m.tag}</p>
                </article>
              ))}
            </div>
          </div>
        ))}

        {slide(at("fair"), "fair", (
          <div className="slide-body">
            <div className="section-head glass">
              <p className="kicker">Methods in depth</p>
              <h2>A fair fight</h2>
              <p>Comparing reinforcement learning with evolution is easy to get wrong: give one a better reward, a bigger brain or more compute and it wins for the wrong reason. Here everything except the learning rule is identical.</p>
              <ul className="rules">
                {RULES.map((r) => (
                  <li key={r.k}><Icon name={r.icon} /><span><b>{r.k}</b>{r.v}</span></li>
                ))}
              </ul>
            </div>
            <div className="depth glass" role="table" aria-label="How each method learns">
              <div className="depth-row head" role="row">
                <span role="columnheader">Method</span><span role="columnheader">Learns from</span><span role="columnheader">One round</span><span role="columnheader">Settings</span>
              </div>
              {DEPTH.map((d) => (
                <div key={d.name} className="depth-row" role="row">
                  <span role="rowheader" className="name">{d.name}</span>
                  <span role="cell">{d.learns}</span>
                  <span role="cell">{d.round}</span>
                  <span role="cell" className="mono knobs">{d.knobs}</span>
                </div>
              ))}
            </div>
          </div>
        ))}

        {slide(at("findings-prey"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings · 1 of 5</p>
            <h2 className="scene-title">Trained animals break the ecosystem</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>Worlds where both species survive</h3>
                <p>Percent of test worlds with prey and predators still alive after 10,000 steps, across all five scenarios. Random animals keep the balance best. Of the trained methods only ES and the hybrid usually do, and every collapse is the same: the predators starve. Prey never died out.</p>
              </figcaption>
              <BarChart id="surv" bars={SURVIVAL} max={100} unit="percent of worlds" decimals={0} />
              <p className="foot">Training rewards each animal for its own energy over a short window, so nothing in it rewards keeping the other species alive. Seeds matter too: ES kept every world alive on two of its five training seeds and lost almost all of them on another.</p>
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-scenarios"), "finding-slide wide", (
          <div className="slide-body">
            <p className="kicker on-scene">Findings · 2 of 5</p>
            <h2 className="scene-title">Harder worlds expose every method</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>Survival in each scenario</h3>
                <p>Share of test worlds with both species alive after 10,000 steps. Every brain was trained only on the baseline world, then tested on four variations it never saw. The hand-coded rules survive poor grass and fast prey but fail when the world starts crowded; ES is the only learner that holds up everywhere.</p>
              </figcaption>
              <Heatmap />
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-collapse"), "finding-slide right wide", (
          <div className="slide-body">
            <p className="kicker on-scene">Findings · 3 of 5</p>
            <h2 className="scene-title">Predators starve with food all around them</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How the predators disappear</h3>
                <p>GA and NEAT predators dwindle from the first thousand steps. PPO keeps the biggest packs in the baseline world but fades in harsher ones. Yet when the last predator died, about 500 to 570 prey were still alive: the learned prey simply got too hard to catch. Only the hand-coded world collapses the textbook way, with prey eaten down to a few dozen first.</p>
              </figcaption>
              <div className="pair">
                <div>
                  <p className="chart-title">Predators per world over time <span className="dim">(baseline world, median of 25)</span></p>
                  <PredatorLines />
                </div>
                <div>
                  <p className="chart-title">Prey alive when the last predator died <span className="dim">(median)</span></p>
                  <BarChart id="preyend" bars={PREY_AT_END} max={600} unit="prey" decimals={0} />
                </div>
              </div>
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-predators"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings · 4 of 5</p>
            <h2 className="scene-title">The keenest hunters die youngest</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How long a predator lives</h3>
                <p>Mean predator lifespan in steps, baseline world. GA and NEAT predators catch the most prey per step, yet live the shortest lives: they burn energy faster than their catches replace it, and their populations dwindle within a few thousand steps. Their prey, meanwhile, live about twice as long as PPO prey.</p>
              </figcaption>
              <BarChart id="pred" bars={PRED_LIFE} max={1400} unit="steps" decimals={0} />
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-reward"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings · 5 of 5</p>
            <h2 className="scene-title">One reward term flipped the result</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>PPO with and without a birth bonus</h3>
                <p>PPO first earned a +1 bonus for each birth. Removing it, so every method optimised exactly the same thing, made PPO the best prey and put its predators level with evolution. A small reward detail decided which family of methods looked better.</p>
              </figcaption>
              <Dumbbell />
            </figure>
            <p className="caveat">Earlier controlled experiment: one PPO run per setting, each species trained against hand-coded opponents. The benchmark uses no birth bonus for any method.</p>
          </div>
        ))}

        {slide(at("why-leak"), "finding-slide wide", (
          <div className="slide-body">
            <p className="kicker on-scene">Why it collapses · 1 of 2</p>
            <h2 className="scene-title">The predators spend more than they earn</h2>
            <figure className="finding glass">
              <div className="pair even">
                <figcaption>
                  <h3>An energy leak</h3>
                  <p>Pushing forward costs energy, and the cost grows with the square of the push. GA and NEAT predators never learn to coast: they push hard all the time, and their catches can't pay for it. So they starve slowly in a world full of prey.</p>
                  <p>The line is a formula built only from the world's rules. Below it, a predator can't bank enough energy in its lifetime to raise one child, so the pack shrinks. Ranked by this formula, every brain whose predators died out scores below every brain whose predators survived.</p>
                  <ul className="legend">
                    <li><span className="sw ppo" />PPO</li><li><span className="sw ga" />GA</li><li><span className="sw es" />ES</li>
                    <li><span className="sw neat" />NEAT</li><li><span className="sw hybrid" />Hybrid</li><li><span className="sw base" />Baselines</li>
                    <li><span className="sw hollow" />predators died out</li>
                  </ul>
                </figcaption>
                <div>
                  <p className="chart-title">Each dot is one trained brain's predators <span className="dim">(baseline world)</span></p>
                  <LeakScatter />
                </div>
              </div>
              <p className="foot">42 brains: the benchmark's plus the movement-cost runs. The plain rule "below the line dies" gets 83% right; every miss is a survivor sitting just under the line.</p>
            </figure>
          </div>
        ))}

        {slide(at("why-window"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Why it collapses · 2 of 2</p>
            <h2 className="scene-title">Evolution judges too quickly to notice</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How clearly the saving shows through the luck</h3>
                <p>GA, ES and NEAT score each brain on just 128 steps. In so short a window, whether a predator happened to catch something matters about four times more than the energy it saved by coasting. At 1.0 the saving and the luck are equal; that takes a window of about 1,100 steps.</p>
              </figcaption>
              <BarChart id="signal" bars={SIGNAL} max={1} unit="signal divided by luck" decimals={2} />
              <ul className="stats">
                <li>GA, ES and NEAT judge on 128 steps</li>
                <li>needed: about 1,100</li>
                <li>luck shrinks as window<sup>−0.66</sup></li>
              </ul>
              <p className="foot">Measured on the GA champion: half the copies were made to coast, and we checked how often selection picked them. With 128 steps they were picked 71% of the time, where 50% would be pure chance. PPO learns from every step, and its predators do learn to coast.</p>
            </figure>
          </div>
        ))}

        {slide(at("fixes"), "finding-slide right wide", (
          <div className="slide-body">
            <p className="kicker on-scene">The fixes</p>
            <h2 className="scene-title">One change to training can save the world</h2>
            <figure className="finding glass">
              <div className="pair even">
                <figcaption>
                  <h3>What worked, and what didn't</h3>
                  <p>Each row changes one thing in training and nothing in the test. Making movement cost more plugs GA's energy leak. Letting PPO look further ahead, or slowing the prey so they can't win the arms race, rescues worlds too.</p>
                  <p>Three ideas failed. Paying every animal for a healthy ecosystem gives no single animal a reason to act differently. NEAT's predators got thriftier but also caught less. And slowing the predators' learning lost every world.</p>
                  <p className="foot">Homeostatic Co-Training, our rule that pauses a species' learning while the other is rare, gave a smaller gain: ES 73% → 87%, PPO 28% → 39%.</p>
                </figcaption>
                <div>
                  <p className="chart-title">Worlds with both species alive <span className="dim">(0 to 100%)</span></p>
                  <Slopes />
                </div>
              </div>
            </figure>
            {study}
          </div>
        ))}

        {slide(at("arms"), "finding-slide wide", (
          <div className="slide-body">
            <p className="kicker on-scene">Arms race</p>
            <h2 className="scene-title">The most skilled animals keep the worst world</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>Prey of one method meet predators of another</h3>
                <p>PPO breeds the best hunters and the best escapers, yet its own worlds collapse most. Its prey starve almost every other method's predators (top row), while its predators live off anyone's prey (first column). Skill and survival pull apart: the steadiest method, ES, is only average at hunting and escaping.</p>
              </figcaption>
              <div className="pair">
                <div>
                  <p className="chart-title">Worlds with both species alive <span className="dim">(baseline world, 25 per cell)</span></p>
                  <CrossPlay />
                </div>
                <div>
                  <p className="chart-title">Skill against the other methods <span className="dim">(1.0 = average)</span></p>
                  <BarChart id="skill" bars={SKILL} max={1.6} unit="relative skill" decimals={2} />
                </div>
              </div>
              <p className="foot">Outlined cells pair a method with itself, as in the benchmark. Skill combines how well a method's predators catch the others' prey and how well its prey avoid the others' predators.</p>
            </figure>
          </div>
        ))}

        {slide(at("drag"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">A second world</p>
            <h2 className="scene-title">Add friction, and learning fails harder</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>The drag world</h3>
                <p>To check that none of this depends on free coasting, we built a second world where every animal loses 10% of its speed each step. The hand-coded animals keep it alive every time. Retrained from scratch, the five learning methods kept 2 of 375 test worlds, and again it is always the predators that die.</p>
              </figcaption>
              <BarChart id="dragw" bars={DRAG} max={100} unit="percent of worlds" decimals={0} />
              <p className="foot">The extinction formula, fixed on the first world and applied here unchanged, ranked all 32 of these unseen brains correctly. Only two of them kept their predators, so this confirms the ranking more than the exact threshold. The movement-cost fix helps less here: GA reaches 20% in the baseline scenario.</p>
            </figure>
            {study}
          </div>
        ))}

        {slide(at("team"), "team", (
          <div className="slide-body narrow">
            <article className="card glass team-card">
              <p className="kicker">Paper &amp; team</p>
              <h2>Who built this</h2>
              <p>EcoArena began as a Machine Learning course project at United International University, Dhaka, and grew into a research paper, now in preparation.</p>
              <ul className="people">
                {TEAM.map((n) => <li key={n}>{n}</li>)}
              </ul>
              <p className="aff">Department of Computer Science and Engineering, United International University, Dhaka, Bangladesh</p>
              <Cite />
            </article>
          </div>
        ))}

        {slide(at("enter"), "enter", (
          <>
            <h2>See it for yourself</h2>
            <p>Pick a brain for each species, swap a broken brain for its repaired version, switch on the drag world, and watch the predators' energy budget decide their fate live.</p>
            <button type="button" className="cta big" onClick={enter}>Enter the arena</button>
            <p className="fine">Runs entirely in your browser. The same rules as the Python benchmark, checked against it step by step.</p>
          </>
        ))}
      </main>
    </div>
  );
}

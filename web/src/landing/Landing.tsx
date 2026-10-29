import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { CHAPTERS, StoryScene } from "./StoryScene";
import "./landing.css";

// ------------------------------------------------------------------------------------------ findings data
// Preliminary results: one training run per method (150k world-steps), evaluated on 4 fresh worlds x 3,000 steps
// against hand-coded opponents. Sources: results/{ppo,ga,es,neat,hybrid}/SUMMARY.md in the repo.
type Bar = { label: string; v: number; tone?: "rl" | "evo" | "base" };
const PREY_ATTACK: Bar[] = [
  { label: "PPO", v: 8.2, tone: "rl" },
  { label: "Random", v: 14.0, tone: "base" },
  { label: "Hand-coded", v: 14.2, tone: "base" },
  { label: "ES", v: 14.8, tone: "evo" },
  { label: "Hybrid", v: 15.0, tone: "evo" },
  { label: "GA", v: 15.2, tone: "evo" },
  { label: "NEAT", v: 16.0, tone: "evo" },
];
const PRED_CATCH: Bar[] = [
  { label: "GA", v: 8.18, tone: "evo" },
  { label: "ES", v: 8.03, tone: "evo" },
  { label: "Hybrid", v: 7.64, tone: "evo" },
  { label: "NEAT", v: 7.59, tone: "evo" },
  { label: "PPO", v: 7.45, tone: "rl" },
  { label: "Hand-coded", v: 5.92, tone: "base" },
  { label: "Random", v: 4.78, tone: "base" },
];

function BarChart({ id, bars, max, unit, decimals = 1 }: { id: string; bars: Bar[]; max: number; unit: string; decimals?: number }) {
  const rowH = 26, labelW = 86, W = 360, H = bars.length * rowH + 22;
  const ticks = [0, max / 4, max / 2, (3 * max) / 4, max];
  return (
    <svg className="bars" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={`${id}-t`}>
      <title id={`${id}-t`}>{bars.map((b) => `${b.label} ${b.v}`).join(", ")} {unit}</title>
      {ticks.map((t) => {
        const x = labelW + ((W - labelW - 44) * t) / max;
        return (
          <g key={t}>
            <line x1={x} x2={x} y1={0} y2={H - 18} className="grid" />
            <text x={x} y={H - 4} className="tick" textAnchor="middle">{Number.isInteger(t) ? t : t.toFixed(1)}</text>
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
  { name: "PPO", family: "Reinforcement learning", body: "Learns from every step. A critic estimates how good each moment is, and the policy is nudged toward actions that did better than expected.", tag: "Best prey so far" },
  { name: "GA", family: "Genetic algorithm", body: "Fifty networks share the world at once. The five that gather the most reward survive each generation; the rest are mutated copies of them.", tag: "Most catches per predator" },
  { name: "ES", family: "Evolution strategies", body: "Tries 25 pairs of opposite random nudges to the weights, then moves the network in the direction the whole population's scores point.", tag: "Longest-lived predators" },
  { name: "NEAT", family: "Evolving topologies", body: "Starts with every input wired straight to the outputs and grows new neurons and connections as it evolves. Our networks grew to about 17 hidden neurons.", tag: "Grows its own network" },
  { name: "Hybrid", family: "Evolution, then RL", body: "Spends half the budget evolving with GA, then hands the best network to PPO to fine-tune for the other half.", tag: "Kept GA's hunting skill" },
] as const;

// ------------------------------------------------------------------------------------------ page
const LABELS: Record<(typeof CHAPTERS)[number], string> = {
  hero: "Start", world: "The world", prey: "Prey", predators: "Predators", refuges: "Refuges", vision: "Senses",
  methods: "Methods", "findings-prey": "Findings: prey", "findings-predators": "Findings: predators",
  "findings-reward": "Findings: reward", enter: "Simulation",
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
  const live = useRef({ index: 0, lockedUntil: 0, needQuiet: false, lastWheel: 0, wheelSum: 0 });

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
    L.lockedUntil = performance.now() + (reduce ? 300 : ms * 0.85);
    L.needQuiet = true;
    L.wheelSum = 0;
    setIndex(to);
  }, []);

  const step = useCallback((d: number) => {
    const L = live.current;
    if (performance.now() < L.lockedUntil) return;
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

  const caveat = <p className="caveat">Preliminary: one training run per method, each species trained against hand-coded opponents and tested on 4 fresh worlds for 3,000 steps. The full benchmark, 5 seeds across 5 scenarios, is running now.</p>;
  const worldOn = index >= at("world") && index <= at("vision");
  const findOn = index >= at("findings-prey") && index <= at("findings-reward");

  return (
    <div className={`landing ${leaving ? "leaving" : ""}`}>
      <div className="story-canvas" ref={host} aria-hidden="true" />
      <div className="veil" aria-hidden="true" />
      <div className="progress" aria-hidden="true"><span style={{ transform: `scaleX(${index / (N - 1)})` }} /></div>

      <header className={`lnav ${index > 0 ? "solid" : ""}`}>
        <button type="button" className="wordmark" onClick={() => go(0)}>EcoArena</button>
        <nav aria-label="Sections">
          <button type="button" onClick={() => go(at("world"))} className={worldOn ? "on" : ""}>The world</button>
          <button type="button" onClick={() => go(at("methods"))} className={index === at("methods") ? "on" : ""}>Methods</button>
          <button type="button" onClick={() => go(at("findings-prey"))} className={findOn ? "on" : ""}>Findings</button>
        </nav>
        <button type="button" className="cta small" onClick={enter}>Open the simulation</button>
      </header>

      <ol className="dots" aria-label="Chapters">
        {CHAPTERS.map((c, i) => (
          <li key={c}>
            <button type="button" className={index === i ? "on" : ""} aria-current={index === i ? "step" : undefined} onClick={() => go(i)}>
              <span className="dot" aria-hidden="true" />
              <span className="lbl">{LABELS[c]}</span>
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

        {slide(at("findings-prey"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings so far · 1 of 3</p>
            <h2 className="scene-title">Only reinforcement learning learns to survive as prey</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How often prey get caught</h3>
                <p>Catches per prey, per predator, per step (×10⁶). Lower is better. PPO prey learned to stay out of reach; every evolutionary method ended close to, or worse than, the hand-coded rules.</p>
              </figcaption>
              <BarChart id="prey" bars={PREY_ATTACK} max={16} unit="catches per million encounters" />
              <p className="foot">Random prey look good here only because they barely meet predators: they starve, and none of their test worlds kept both species.</p>
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-predators"), "finding-slide right", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings so far · 2 of 3</p>
            <h2 className="scene-title">Every method learns to hunt</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How often predators eat</h3>
                <p>Catches per 1,000 predator-steps. Higher is better. Every learned predator beats the hand-coded hunter by 25 to 40 percent, and they ambush at the refuge edges: prey have to leave the rocks to eat, and that is where they are caught.</p>
              </figcaption>
              <BarChart id="pred" bars={PRED_CATCH} max={10} unit="catches per 1,000 steps" decimals={2} />
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-reward"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings so far · 3 of 3</p>
            <h2 className="scene-title">One reward term flipped the result</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>PPO with and without a birth bonus</h3>
                <p>PPO first earned a +1 bonus for each birth. Removing it, so every method optimised exactly the same thing, made PPO the best prey and put its predators level with evolution. A small reward detail decided which family of methods looked better.</p>
              </figcaption>
              <Dumbbell />
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("enter"), "enter", (
          <>
            <h2>See it for yourself</h2>
            <p>Pick a brain for each species, change the scenario, follow a single animal, or let the camera tour the hunts. Turn on the Catches map to watch the refuge-edge ambushes form.</p>
            <button type="button" className="cta big" onClick={enter}>Enter the arena</button>
            <p className="fine">Runs entirely in your browser. The same rules as the Python benchmark, checked against it step by step.</p>
          </>
        ))}
      </main>
    </div>
  );
}

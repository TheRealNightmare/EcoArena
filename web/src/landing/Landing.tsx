import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import Loader from "../loader/Loader";
import { CHAPTERS, StoryScene } from "./StoryScene";
import "./landing.css";

// ------------------------------------------------------------------------------------------ findings data
// Benchmark results: every method controls both species, 5 training seeds x 150k world-steps each, evaluated on
// 5 fresh worlds x 10,000 steps in each of 5 scenarios. Source: results/bench/STATS.md in the repo.
type Bar = { label: string; v: number; tone?: "rl" | "evo" | "base" };
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
    L.lockedUntil = performance.now() + (reduce ? 300 : ms * 0.85);
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
            <p className="kicker on-scene">Findings · 1 of 3</p>
            <h2 className="scene-title">Trained animals break the ecosystem</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>Worlds where both species survive</h3>
                <p>Percent of test worlds with prey and predators still alive after 10,000 steps, across all five scenarios. Random animals keep the balance best. Of the trained methods only ES and the hybrid usually do, and every collapse is the same: the predators starve. Prey never died out.</p>
              </figcaption>
              <BarChart id="surv" bars={SURVIVAL} max={100} unit="percent of worlds" decimals={0} />
              <p className="foot">Training rewards each animal for its own energy over a short window, so nothing rewards keeping the other species alive. Even ES ranges from a perfect record on four seeds to collapse on the fifth.</p>
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-predators"), "finding-slide right", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings · 2 of 3</p>
            <h2 className="scene-title">The keenest hunters die youngest</h2>
            <figure className="finding glass">
              <figcaption>
                <h3>How long a predator lives</h3>
                <p>Mean predator lifespan in steps, baseline world. GA and NEAT predators catch the most prey per step, yet live the shortest lives: they spend more energy chasing than they win back, and their populations dwindle within a few thousand steps. Their prey, meanwhile, live about twice as long as PPO prey.</p>
              </figcaption>
              <BarChart id="pred" bars={PRED_LIFE} max={1400} unit="steps" decimals={0} />
            </figure>
            {caveat}
          </div>
        ))}

        {slide(at("findings-reward"), "finding-slide", (
          <div className="slide-body narrow">
            <p className="kicker on-scene">Findings · 3 of 3</p>
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

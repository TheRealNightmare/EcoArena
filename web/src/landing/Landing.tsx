import { useEffect, useRef, useState } from "react";
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
export default function Landing({ onEnter }: { onEnter: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const story = useRef<StoryScene | null>(null);
  const [chapter, setChapter] = useState(0);
  const [leaving, setLeaving] = useState(false);

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

  // scroll position -> chapter progress (0 = hero centred, 1 = first chapter centred, ...)
  useEffect(() => {
    const el = scroller.current!;
    let raf = 0;
    const measure = () => {
      raf = 0;
      const secs = Array.from(el.querySelectorAll<HTMLElement>("[data-ch]"));
      const mid = el.scrollTop + el.clientHeight / 2;
      const centres = secs.map((s) => s.offsetTop + s.offsetHeight / 2);
      let p = 0;
      if (mid >= centres[centres.length - 1]) p = centres.length - 1;
      else if (mid > centres[0]) {
        const i = centres.findIndex((c, k) => mid >= c && mid < centres[k + 1]);
        p = i + (mid - centres[i]) / (centres[i + 1] - centres[i]);
      }
      story.current?.setProgress(p);
      setChapter(Math.round(p));
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    measure();
    return () => {
      el.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const enter = () => {
    if (leaving) return;
    setLeaving(true);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !story.current) onEnter();
    else story.current.diveIn(onEnter);
  };

  const goTo = (i: number) => {
    const el = scroller.current!;
    const sec = el.querySelector<HTMLElement>(`[data-ch="${i}"]`);
    if (sec) el.scrollTo({ top: sec.offsetTop + sec.offsetHeight / 2 - el.clientHeight / 2, behavior: "smooth" });
  };

  const labels: Record<(typeof CHAPTERS)[number], string> = {
    hero: "Start", world: "The world", prey: "Prey", predators: "Predators", refuges: "Refuges", vision: "Senses",
    methods: "Methods", findings: "Findings", enter: "Simulation",
  };

  return (
    <div className={`landing ${leaving ? "leaving" : ""}`}>
      <div className="story-canvas" ref={host} aria-hidden="true" />
      <div className="veil" aria-hidden="true" />

      <header className={`lnav ${chapter > 0 ? "solid" : ""}`}>
        <button type="button" className="wordmark" onClick={() => goTo(0)}>EcoArena</button>
        <nav aria-label="Sections">
          <button type="button" onClick={() => goTo(1)} className={chapter >= 1 && chapter <= 5 ? "on" : ""}>The world</button>
          <button type="button" onClick={() => goTo(6)} className={chapter === 6 ? "on" : ""}>Methods</button>
          <button type="button" onClick={() => goTo(7)} className={chapter === 7 ? "on" : ""}>Findings</button>
        </nav>
        <button type="button" className="cta small" onClick={enter}>Open the simulation</button>
      </header>

      <ol className="dots" aria-label="Chapters">
        {CHAPTERS.map((c, i) => (
          <li key={c}>
            <button type="button" className={chapter === i ? "on" : ""} aria-current={chapter === i ? "step" : undefined} onClick={() => goTo(i)}>
              <span className="dot" aria-hidden="true" />
              <span className="lbl">{labels[c]}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="scroller" ref={scroller}>
        <section className="ch hero" data-ch={0}>
          <p className="eyebrow">A benchmark for learning in a living world</p>
          <h1>EcoArena</h1>
          <p className="lede">
            Predators and prey share one world that never resets. We give each species a brain trained by a different
            machine-learning method and ask a simple question: does the ecosystem survive?
          </p>
          <div className="hero-actions">
            <button type="button" className="cta" onClick={enter}>Open the simulation</button>
            <button type="button" className="ghost" onClick={() => goTo(1)}>See how it works</button>
          </div>
          <p className="scroll-cue" aria-hidden="true"><span />Scroll</p>
        </section>

        {STORY.map((s, k) => (
          <section key={s.kicker} className={`ch story ${s.side}`} data-ch={k + 1}>
            <article className="card glass">
              <p className="kicker"><span className="num mono">{String(k + 1).padStart(2, "0")}</span>{s.kicker}</p>
              <h2>{s.title}</h2>
              <p>{s.body}</p>
              <ul className="stats">
                {s.stats.map((x) => <li key={x}>{x}</li>)}
              </ul>
            </article>
          </section>
        ))}

        <section className="ch methods" data-ch={6}>
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
        </section>

        <section className={`ch findings ${chapter === 7 ? "active" : ""}`} data-ch={7}>
          <div className="section-head glass">
            <p className="kicker">Findings so far</p>
            <h2>Every method learns to hunt. Only reinforcement learning learns to survive as prey.</h2>
          </div>
          <div className="finding-grid">
            <figure className="finding glass">
              <figcaption>
                <h3>Prey: how often they get caught</h3>
                <p>Catches per prey, per predator, per step (×10⁶). Lower is better. PPO prey learned to stay out of reach; every evolutionary method ended close to, or worse than, the hand-coded rules.</p>
              </figcaption>
              <BarChart id="prey" bars={PREY_ATTACK} max={16} unit="catches per million encounters" />
              <p className="foot">Random prey look good here only because they barely meet predators: they starve, and none of their test worlds kept both species.</p>
            </figure>
            <figure className="finding glass">
              <figcaption>
                <h3>Predators: how often they eat</h3>
                <p>Catches per 1,000 predator-steps. Higher is better. Every learned predator beats the hand-coded hunter by 25 to 40 percent.</p>
              </figcaption>
              <BarChart id="pred" bars={PRED_CATCH} max={10} unit="catches per 1,000 steps" decimals={2} />
            </figure>
            <figure className="finding glass">
              <figcaption>
                <h3>One reward term flipped the result</h3>
                <p>PPO first earned a +1 bonus for each birth. Removing it, so every method optimised exactly the same thing, made PPO the best prey and put its predators level with evolution.</p>
              </figcaption>
              <Dumbbell />
            </figure>
            <article className="finding glass insight">
              <h3>Predators wait at the refuge edges</h3>
              <p>Mapping where catches happen shows hot rings around the rocks: prey have to leave the refuges to eat, and that is where they are caught. Turn on the Catches map in the simulation to watch it form.</p>
            </article>
          </div>
          <p className="caveat glass">Preliminary: one training run per method, each species trained against hand-coded opponents and tested on 4 fresh worlds for 3,000 steps. The full benchmark, 5 seeds across 5 scenarios, is running now.</p>
        </section>

        <section className="ch enter" data-ch={8}>
          <h2>See it for yourself</h2>
          <p>Pick a brain for each species, change the scenario, follow a single animal, or let the camera tour the hunts.</p>
          <button type="button" className="cta big" onClick={enter}>Enter the arena</button>
          <p className="fine">Runs entirely in your browser. The same rules as the Python benchmark, checked against it step by step.</p>
        </section>
      </div>
    </div>
  );
}

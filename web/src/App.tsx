import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { BRAIN_INDEX } from "./brainIndex";
import Loader from "./loader/Loader";
import type { BrainName, Frame, Selected, SimSettings, ToWorker } from "./protocol";
import { drawChart } from "./view/Chart";
import { Diorama, type CameraMode, type Overlay, type Quality, type SkyMode } from "./view/Diorama";
import SimWorker from "./worker?worker&inline";

const BRAINS: { id: BrainName; label: string }[] = [
  { id: "random", label: "Random" },
  { id: "heuristic", label: "Hand-coded" },
  { id: "ppo", label: "PPO" },
  { id: "ga", label: "GA" },
  { id: "es", label: "ES" },
  { id: "neat", label: "NEAT" },
  { id: "hybrid", label: "Hybrid" },
];

const ABOUT: Record<BrainName, string> = {
  random: "Uniform random turning and thrust. The floor every method has to beat.",
  heuristic: "Hand-written rules on the same 102 inputs: prey graze and run for refuges, predators chase and rest while digesting.",
  ppo: "Reinforcement learning (PPO): one shared 64-64 network per species, improved by gradient steps on every animal's reward.",
  ga: "Genetic algorithm: 50 networks share the world, the best 5 survive each generation and the rest are mutated copies.",
  es: "Evolution strategies: 25 pairs of opposite weight nudges estimate which direction raises fitness.",
  neat: "NEAT: starts from direct input-to-output wiring and grows hidden neurons and connections as it evolves.",
  hybrid: "Half the budget evolving with GA, then PPO fine-tunes the best evolved network.",
};

const SCENARIOS: [string, string][] = [
  ["S1", "Baseline rules"],
  ["S2", "Grass regrows at half speed"],
  ["S3", "Prey 30% faster"],
  ["S4", "Twice as many of both at start"],
  ["S5", "Twice as many predators at start"],
];

const SPEEDS = [5, 15, 30, 60, 0];
const HISTORY = 4000;
const KINDS = ["grass", "prey", "predator", "refuge", "nothing"];

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(`ecoarena.${key}`);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, v: unknown) {
  try {
    localStorage.setItem(`ecoarena.${key}`, JSON.stringify(v));
  } catch {
    /* storage unavailable: fine */
  }
}

/** Number that glides to its new value. */
function useTween(target: number, ms = 450) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  const start = useRef(0);
  const cur = useRef(target);
  useEffect(() => {
    from.current = cur.current;
    start.current = performance.now();
    let raf = 0;
    const tick = () => {
      const p = Math.min(1, (performance.now() - start.current) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      cur.current = from.current + (target - from.current) * e;
      setV(cur.current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

const Icon = ({ d, size = 18 }: { d: string; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);
const I = {
  play: "M7 4.5v15l12-7.5z",
  pause: "M8 5v14M16 5v14",
  step: "M6 5l9 7-9 7zM18 5v14",
  orbit: "M12 4a8 8 0 1 0 8 8M20 4v4h-4",
  top: "M4 4h16v16H4zM4 12h16M12 4v16",
  follow: "M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.8 12 21 12 21zM12 12a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5",
  tour: "M3 7h12a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h12M18 16l3 3-3 3",
  sun: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  globe: "M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18",
  brain: "M9 4a3 3 0 0 0-3 3v1a3 3 0 0 0-2 5 3 3 0 0 0 3 4 3 3 0 0 0 5 1V5a2 2 0 0 0-3-1zM15 4a3 3 0 0 1 3 3v1a3 3 0 0 1 2 5 3 3 0 0 1-3 4 3 3 0 0 1-5 1",
  chart: "M4 19h16M6 16l4-5 3 3 5-7",
  close: "M6 6l12 12M18 6L6 18",
  chevron: "M6 9l6 6 6-6",
};

function Panel({ id, title, icon, children, open, onToggle, className = "", extra }: {
  id: string; title: string; icon: string; children: ReactNode; open: boolean; onToggle: () => void; className?: string; extra?: ReactNode;
}) {
  return (
    <section className={`glass panel ${className} ${open ? "open" : "closed"}`} aria-labelledby={`${id}-title`}>
      <div className="panel-head">
        <button type="button" className="panel-toggle" id={`${id}-toggle`} aria-expanded={open} onClick={onToggle}>
          <Icon d={I[icon as keyof typeof I]} size={16} />
          <h2 id={`${id}-title`}>{title}</h2>
          <span className="chev"><Icon d={I.chevron} size={14} /></span>
        </button>
        {extra}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export default function App() {
  const [settings, setSettings] = useState<SimSettings>({ base: "s1_large.json", scenario: "S1", seed: 1, prey: "es", predator: "es" });
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(30);
  const [chartMode, setChartMode] = useState<"time" | "phase">("time");
  const [stats, setStats] = useState({ t: 0, prey: 0, pred: 0, sps: 0, caught: 0 });
  const [extinct, setExtinct] = useState<Frame["extinct"]>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [seedText, setSeedText] = useState("1");
  const [camMode, setCamMode] = useState<CameraMode>("orbit");
  const [sky, setSky] = useState<SkyMode>(() => load<SkyMode>("sky", "cycle"));
  const [quality, setQuality] = useState<Quality>(() => load<Quality>("quality", "high"));
  const [overlay, setOverlay] = useState<Overlay>("none");
  const [open, setOpen] = useState(() => load("panels", { brains: true, chart: true }));
  const [sheet, setSheet] = useState<"none" | "brains" | "chart" | "world">("none");
  const [worldOpen, setWorldOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  // loading screen: steps tick off as the worker, world, shaders and first frame really finish
  const [loading, setLoading] = useState(true);
  const [loadDone, setLoadDone] = useState(0);
  const [animalsLabel, setAnimalsLabel] = useState("Growing grass and waking the animals");
  const loadingRef = useRef(true);
  loadingRef.current = loading;
  const [clock, setClock] = useState("");

  const worker = useRef<Worker | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const appRef = useRef<HTMLDivElement>(null);
  const brandRef = useRef<HTMLElement>(null);
  const view = useRef<Diorama | null>(null);
  const chart = useRef<HTMLCanvasElement>(null);
  const sparkPrey = useRef<HTMLCanvasElement>(null);
  const sparkPred = useRef<HTMLCanvasElement>(null);
  const hist = useRef<{ prey: number[]; pred: number[]; t0: number; caught: number[] }>({ prey: [], pred: [], t0: 0, caught: [] });
  const live = useRef({ running, speed });
  live.current = { running, speed };
  const chartModeRef = useRef(chartMode);
  chartModeRef.current = chartMode;

  const send = useCallback((m: ToWorker) => worker.current?.postMessage(m), []);

  const redrawCharts = useCallback(() => {
    const h = hist.current;
    const colors = { fg: cssVar("--fg"), muted: cssVar("--muted"), line: cssVar("--line"), prey: cssVar("--prey"), predator: cssVar("--predator") };
    if (chart.current && chart.current.offsetParent) drawChart(chart.current, h.prey, h.pred, h.t0, chartModeRef.current, colors);
    const spark = (cv: HTMLCanvasElement | null, vals: number[], color: string) => {
      if (!cv || vals.length < 2) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = cv.clientWidth, H = cv.clientHeight;
      if (cv.width !== Math.round(W * dpr)) {
        cv.width = Math.round(W * dpr);
        cv.height = Math.round(H * dpr);
      }
      const g = cv.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      const v = vals.slice(-600);
      const max = Math.max(...v, 1), min = Math.min(...v);
      const span = Math.max(1, max - min);
      g.beginPath();
      v.forEach((y, i) => {
        const X = (i / (v.length - 1)) * W, Y = H - 2 - ((y - min) / span) * (H - 4);
        if (i) g.lineTo(X, Y);
        else g.moveTo(X, Y);
      });
      g.strokeStyle = color;
      g.lineWidth = 1.5;
      g.stroke();
    };
    spark(sparkPrey.current, h.prey, colors.prey);
    spark(sparkPred.current, h.pred, colors.predator);
  }, []);

  // 3D view + worker, created once (after the loading screen has painted, so it animates from the start)
  const [canBuild, setCanBuild] = useState(false);
  useEffect(() => {
    let a = 0, b = 0;
    a = requestAnimationFrame(() => (b = requestAnimationFrame(() => setTimeout(() => setCanBuild(true), 60))));
    return () => { cancelAnimationFrame(a); cancelAnimationFrame(b); };
  }, []);
  useEffect(() => {
    if (!canBuild) return;
    const v = new Diorama(stage.current!, (x, y) => send({ type: "select", x, y }), quality);
    v.onSlowDevice = () => {
      setQuality("low");
      setToast("Switched to Low quality for smoother motion. You can change it in the dock.");
    };
    v.paused = true; // hidden behind the loading screen until the hand-over
    v.onCameraMode = (m) => setCamMode(m);
    v.onTourPick = (species, slot) => send({ type: "selectSlot", species, slot });
    view.current = v;
    const w = new SimWorker();
    worker.current = w;
    setLoadDone(1);
    let lastChart = 0;
    let boot: "waiting" | "building" | "done" = "waiting";
    const handle = (f: Frame) => {
      const h = hist.current;
      if (f.reset) {
        h.prey = [];
        h.pred = [];
        h.caught = [];
        h.t0 = f.t;
      }
      for (let i = 0; i < f.popHistory.length; i += 2) {
        h.prey.push(f.popHistory[i]);
        h.pred.push(f.popHistory[i + 1]);
      }
      if (h.prey.length > HISTORY) {
        const cut = h.prey.length - HISTORY;
        h.prey.splice(0, cut);
        h.pred.splice(0, cut);
        h.t0 += cut;
      }
      h.caught.push(f.captures);
      if (h.caught.length > 30) h.caught.shift();
      v.update(f);
      setStats({ t: f.t, prey: f.prey.slot.length, pred: f.pred.slot.length, sps: f.stepsPerSec, caught: h.caught.reduce((a, b) => a + b, 0) });
      setExtinct(f.extinct);
      setSelected(f.selected);
      setClock(v.clockText());
      const now = performance.now();
      if (now - lastChart > 120) {
        lastChart = now;
        redrawCharts();
      }
    };
    w.onmessage = (e: MessageEvent<Frame>) => {
      const f = e.data;
      if (boot === "done") return handle(f);
      if (boot === "building") return; // the world is being built from the first frame
      // first frame: the worker has built the world and loaded the brains
      boot = "building";
      setLoadDone(2);
      setAnimalsLabel(`Growing grass, waking ${f.prey.slot.length.toLocaleString()} prey and ${f.pred.slot.length.toLocaleString()} predators`);
      setTimeout(() => {
        handle(f); // builds terrain, grass, refuges and animals
        setLoadDone(3);
        v.compile().then(() => {
          setLoadDone(4);
          requestAnimationFrame(() => requestAnimationFrame(() => {
            boot = "done";
            setLoadDone(5);
          }));
        });
      }, 40);
    };
    return () => {
      w.terminate();
      v.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canBuild, send, redrawCharts]);

  useEffect(() => {
    if (!canBuild) return;
    send({ type: "init", settings, running: live.current.running && !loadingRef.current, speed: live.current.speed });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canBuild, settings.base, settings.scenario, settings.seed, send]);
  useEffect(() => send({ type: "brain", species: "prey", brain: settings.prey }), [settings.prey, send]);
  useEffect(() => send({ type: "brain", species: "predator", brain: settings.predator }), [settings.predator, send]);
  useEffect(() => send({ type: "run", running: running && !loading }), [running, loading, send]);
  useEffect(() => send({ type: "speed", stepsPerSec: speed }), [speed, send]);
  useEffect(() => redrawCharts(), [chartMode, open, sheet, redrawCharts]);
  useEffect(() => {
    view.current?.setSky(sky);
    save("sky", sky);
  }, [sky]);
  useEffect(() => {
    view.current?.setQuality(quality);
    save("quality", quality);
  }, [quality]);
  useEffect(() => save("panels", open), [open]);
  useEffect(() => view.current?.setOverlay(overlay), [overlay]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);
  useEffect(() => {
    if (!selected && camMode === "follow") {
      setCamMode("orbit");
      view.current?.setCameraMode("orbit");
    }
  }, [selected, camMode]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || (tag === "BUTTON" && e.code === "Space")) return;
      if (e.code === "Space") {
        e.preventDefault();
        setRunning((r) => !r);
      } else if (e.key === ".") {
        setRunning(false);
        send({ type: "step" });
      } else if (e.key === "Escape") {
        send({ type: "deselect" });
        setWorldOpen(false);
        setSheet("none");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send]);

  // the left column starts 12 px below the title card, whatever height the card ends up (font, wrapping, width)
  useEffect(() => {
    const b = brandRef.current, a = appRef.current;
    if (!b || !a) return;
    const ro = new ResizeObserver(() => a.style.setProperty("--brand-bottom", `${b.offsetTop + b.offsetHeight + 12}px`));
    ro.observe(b);
    return () => ro.disconnect();
  }, []);

  const notes = useMemo(() => {
    const n: Record<string, string> = {};
    for (const b of BRAIN_INDEX) if (b.note) n[`${b.method}-${b.species}`] = b.note;
    return n;
  }, []);

  const camera = (m: CameraMode) => {
    if (m === "follow" && !selected) {
      setToast("Click an animal first, then Follow keeps the camera on it.");
      return;
    }
    setCamMode(m);
    view.current?.setCameraMode(m);
  };

  const restart = () => {
    const s = parseInt(seedText, 10);
    const seed = Number.isFinite(s) && s >= 0 ? s : settings.seed;
    setSeedText(String(seed));
    if (seed === settings.seed) send({ type: "init", settings, running, speed });
    else setSettings((cur) => ({ ...cur, seed }));
  };

  const preyN = useTween(stats.prey);
  const predN = useTween(stats.pred);

  const brainRow = (sp: "prey" | "predator") => (
    <fieldset className="brain-row" aria-label={`${sp} brain`}>
      <legend><span className={`dot ${sp}`} aria-hidden="true" />{sp === "prey" ? "Prey" : "Predators"}</legend>
      <div className="chips">
        {BRAINS.map((b) => (
          <button key={b.id} type="button" id={`${sp}-${b.id}`} className={`chip ${settings[sp] === b.id ? "on" : ""}`}
            aria-pressed={settings[sp] === b.id} onClick={() => setSettings((s) => ({ ...s, [sp]: b.id }))}>
            {b.label}
          </button>
        ))}
      </div>
      <p className="about">
        {ABOUT[settings[sp]]}
        {notes[`${settings[sp]}-${sp}`] ? <span className="note"> {notes[`${settings[sp]}-${sp}`]}.</span> : null}
      </p>
    </fieldset>
  );

  const brainsBody = (
    <>
      {brainRow("prey")}
      {brainRow("predator")}
      <p className="caption">Each method trained its prey and predators together; mixing two methods pairs animals that never met in training. A switch takes effect on the next step.</p>
    </>
  );

  const chartBody = (
    <>
      <div className="seg-group" role="radiogroup" aria-label="Chart type">
        <button type="button" role="radio" id="chart-time" aria-checked={chartMode === "time"} className={`seg ${chartMode === "time" ? "on" : ""}`} onClick={() => setChartMode("time")}>Over time</button>
        <button type="button" role="radio" id="chart-phase" aria-checked={chartMode === "phase"} className={`seg ${chartMode === "phase" ? "on" : ""}`} onClick={() => setChartMode("phase")}>Prey vs predators</button>
      </div>
      <canvas ref={chart} className="chart" aria-label="Population chart" />
      <p className="caption">
        {chartMode === "time" ? "Last 4,000 steps. Prey on the left axis, predators on the right." : "Prey across, predators up, one point per step, newest darkest. Lotka–Volterra cycles trace loops."}
      </p>
    </>
  );

  const worldBody = (
    <>
      <div className="scenarios" role="radiogroup" aria-label="Scenario">
        {SCENARIOS.map(([k, label]) => (
          <button key={k} type="button" id={`scenario-${k}`} role="radio" aria-checked={settings.scenario === k}
            className={`scenario ${settings.scenario === k ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, scenario: k }))}>
            <span className="mono">{k}</span><span>{label}</span>
          </button>
        ))}
      </div>
      <div className="row">
        <div className="seg-group" role="radiogroup" aria-label="World size">
          <button type="button" role="radio" id="size-large" aria-checked={settings.base === "s1_large.json"} className={`seg ${settings.base === "s1_large.json" ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, base: "s1_large.json" }))}>200 × 200</button>
          <button type="button" role="radio" id="size-small" aria-checked={settings.base === "s1_balanced.json"} className={`seg ${settings.base === "s1_balanced.json" ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, base: "s1_balanced.json" }))}>100 × 100</button>
        </div>
        <label className="seed" htmlFor="seed">Seed
          <input id="seed" inputMode="numeric" value={seedText} onChange={(e) => setSeedText(e.target.value.replace(/[^0-9]/g, ""))} onKeyDown={(e) => { if (e.key === "Enter") restart(); }} />
        </label>
        <button type="button" id="restart" className="btn" onClick={restart}>Restart</button>
      </div>
      <p className="caption">The benchmark world is 200 × 200 with 48 rocky refuges where prey can't be caught. This browser version follows the same rules as the Python benchmark and is checked against it step by step.</p>
    </>
  );

  const busy = selected && selected.digest > 0;
  const inspector = selected ? (
    <section className="glass inspector" aria-label="Selected animal">
      <div className="panel-head">
        <h2 className="insp-title"><span className={`dot ${selected.species}`} aria-hidden="true" />{selected.species === "prey" ? "Prey" : "Predator"} <span className="mono dim">#{selected.slot}</span></h2>
        <div className="row tight">
          {selected.alive ? <button type="button" className={`btn small ${camMode === "follow" ? "on" : ""}`} onClick={() => camera(camMode === "follow" ? "orbit" : "follow")}>{camMode === "follow" ? "Following" : "Follow"}</button> : null}
          <button type="button" className="icon-btn" aria-label="Close" onClick={() => send({ type: "deselect" })}><Icon d={I.close} size={15} /></button>
        </div>
      </div>
      {selected.alive ? (
        <>
          <div className="meter-row">
            <span className="dim">Energy</span>
            <span className="bar"><span style={{ width: `${Math.round((100 * selected.energy) / selected.energyCap)}%` }} /></span>
            <span className="mono">{selected.energy.toFixed(0)}</span>
          </div>
          <div className="meter-row">
            <span className="dim">Age</span>
            <span className="bar age"><span style={{ width: `${Math.round((100 * selected.age) / selected.maxAge)}%` }} /></span>
            <span className="mono">{selected.age}</span>
          </div>
          <dl className="facts">
            <div><dt>State</dt><dd>{selected.inRefuge ? "Hiding in a refuge" : busy ? (selected.digest > 41 ? "Eating, can't hunt" : "Recovering from a miss") : selected.species === "predator" ? "Hunting" : selected.speed < 0.15 ? "Grazing" : "On the move"}</dd></div>
            <div><dt>Speed</dt><dd className="mono">{selected.speed.toFixed(2)} / {selected.maxSpeed}</dd></div>
            <div><dt>Action</dt><dd className="mono">turn {selected.action[0].toFixed(2)} · thrust {selected.action[1].toFixed(2)}</dd></div>
          </dl>
          <div className="sectors" aria-label="What each vision sector reports, left to right">
            {Array.from({ length: (selected.obs.length - 6) / 6 }, (_, r) => {
              const o = r * 6;
              const k = [1, 2, 3, 4, 5].findIndex((c) => selected.obs[o + c] > 0.5);
              const kind = KINDS[k < 0 ? 4 : k];
              const near = kind === "nothing" ? 0 : 1 - selected.obs[o];
              return <span key={r} className={`sector k-${kind}`} style={{ "--near": near.toFixed(2) } as CSSProperties} title={`${kind}${kind === "nothing" ? "" : ` at ${(selected.obs[o] * selected.range).toFixed(1)}`}`} />;
            }).reverse()}
          </div>
          <p className="caption">Its 16 vision sectors, left to right, brighter when closer. With its energy, speed, age, rest state, grass underfoot and refuge, that is the 102 numbers its brain reads.</p>
        </>
      ) : (
        <p className="about">Died at step {selected.diedAt?.toLocaleString()}.</p>
      )}
    </section>
  ) : null;

  return (
    <div className="app3d" ref={appRef}>
      {loading ? (
        <Loader title="Entering the arena"
          steps={["Starting the simulation", "Building the world and loading the brains", animalsLabel, "Compiling shaders", "Rendering the first frame"]}
          done={loadDone} minMs={1800} onReveal={(frac, ms) => view.current?.reveal(frac, ms)} onFinished={() => setLoading(false)} />
      ) : null}
      <div className="scene" ref={stage} />

      <header className="glass brand" ref={brandRef}>
        <a className="back" href="#" onClick={(e) => { e.preventDefault(); location.hash = ""; }}>← About the project</a>
        <h1>EcoArena</h1>
        <p>Predators and prey sharing one world, each species driven by the brain you choose.</p>
        <div className="clock mono">{clock}<span className="dim"> · step {stats.t.toLocaleString()}</span></div>
      </header>

      <div className="pills" aria-label="Populations">
        <div className="glass pill">
          <span className="label"><span className="dot prey" aria-hidden="true" />Prey</span>
          <span className="num">{Math.round(preyN).toLocaleString()}</span>
          <canvas ref={sparkPrey} className="spark" aria-hidden="true" />
        </div>
        <div className="glass pill">
          <span className="label"><span className="dot predator" aria-hidden="true" />Predators</span>
          <span className="num">{Math.round(predN).toLocaleString()}</span>
          <canvas ref={sparkPred} className="spark" aria-hidden="true" />
        </div>
        <div className="glass pill small">
          <span className="label">Caught / s</span>
          <span className="num">{stats.caught}</span>
        </div>
      </div>

      <div className="side left">
        <Panel id="brains" title="Brains" icon="brain" open={open.brains} onToggle={() => setOpen((o: typeof open) => ({ ...o, brains: !o.brains }))}>{brainsBody}</Panel>
      </div>
      <div className="side right">
        {inspector}
        <Panel id="chart" title="Populations" icon="chart" open={open.chart} onToggle={() => setOpen((o: typeof open) => ({ ...o, chart: !o.chart }))}>{chartBody}</Panel>
      </div>

      {worldOpen ? (
        <section className="glass world-pop" aria-label="World settings">
          <div className="panel-head">
            <h2>World</h2>
            <button type="button" className="icon-btn" aria-label="Close" onClick={() => setWorldOpen(false)}><Icon d={I.close} size={15} /></button>
          </div>
          {worldBody}
        </section>
      ) : null}

      {sheet !== "none" ? (
        <section className="glass sheet" aria-label={sheet}>
          <div className="panel-head">
            <h2>{sheet === "brains" ? "Brains" : sheet === "chart" ? "Populations" : "World"}</h2>
            <button type="button" className="icon-btn" aria-label="Close" onClick={() => setSheet("none")}><Icon d={I.close} size={15} /></button>
          </div>
          {sheet === "brains" ? brainsBody : sheet === "chart" ? chartBody : worldBody}
        </section>
      ) : null}

      {extinct ? (
        <div className="glass toast warn" role="status">
          {extinct.species === "prey" ? "Prey" : "Predators"} died out at step {extinct.t.toLocaleString()}.
          <button type="button" className="btn small" onClick={() => { setSeedText(String(settings.seed + 1)); setSettings((s) => ({ ...s, seed: s.seed + 1 })); }}>New world</button>
        </div>
      ) : toast ? (
        <div className="glass toast" role="status">{toast}</div>
      ) : null}

      <nav className="glass dock" aria-label="Controls">
        <button type="button" id="play" className="dock-play" aria-label={running ? "Pause" : "Play"} onClick={() => setRunning((r) => !r)}>
          <Icon d={running ? I.pause : I.play} size={20} />
        </button>
        <button type="button" id="step" className="icon-btn" aria-label="Step once" title="Step once (.)" onClick={() => { setRunning(false); send({ type: "step" }); }}>
          <Icon d={I.step} />
        </button>
        <div className="seg-group speeds" role="radiogroup" aria-label="Steps per second">
          {SPEEDS.map((v) => (
            <button key={v} type="button" role="radio" id={`speed-${v || "max"}`} aria-checked={speed === v} className={`seg ${speed === v ? "on" : ""}`} onClick={() => setSpeed(v)}>
              {v || "Max"}
            </button>
          ))}
        </div>
        <span className="sps mono dim" aria-live="off">{running && stats.sps > 0 ? `${Math.round(stats.sps)}/s` : running ? "" : "paused"}</span>
        <span className="divider" aria-hidden="true" />
        <div className="seg-group" role="radiogroup" aria-label="Camera">
          {(["orbit", "top", "follow", "tour"] as CameraMode[]).map((m) => (
            <button key={m} type="button" role="radio" id={`cam-${m}`} aria-checked={camMode === m}
              title={m === "orbit" ? "Orbit: drag to turn, scroll to zoom" : m === "top" ? "Top-down" : m === "follow" ? "Follow the selected animal" : "Tour: the camera visits hunts on its own"}
              className={`seg icon ${camMode === m ? "on" : ""}`} onClick={() => camera(m)}>
              <Icon d={I[m]} size={16} /><span className="seg-label">{m === "orbit" ? "Orbit" : m === "top" ? "Top" : m === "follow" ? "Follow" : "Tour"}</span>
            </button>
          ))}
        </div>
        <div className="seg-group" role="radiogroup" aria-label="Time of day">
          {(["cycle", "day", "golden", "night"] as SkyMode[]).map((m) => (
            <button key={m} type="button" role="radio" id={`sky-${m}`} aria-checked={sky === m} className={`seg ${sky === m ? "on" : ""}`} onClick={() => setSky(m)}>
              {m === "cycle" ? "Cycle" : m === "day" ? "Day" : m === "golden" ? "Dusk" : "Night"}
            </button>
          ))}
        </div>
        <div className="seg-group" role="radiogroup" aria-label="Map overlay">
          {(["none", "trails", "catches"] as Overlay[]).map((m) => (
            <button key={m} type="button" role="radio" id={`overlay-${m}`} aria-checked={overlay === m}
              title={m === "none" ? "No overlay" : m === "trails" ? "Fading trails of where animals walked" : "Heat map of where prey were caught"}
              className={`seg ${overlay === m ? "on" : ""}`} onClick={() => setOverlay(m)}>
              {m === "none" ? "Plain" : m === "trails" ? "Trails" : "Catches"}
            </button>
          ))}
        </div>
        <button type="button" id="quality" className="seg-solo" title="Rendering quality" onClick={() => setQuality((q) => (q === "high" ? "low" : "high"))}>
          {quality === "high" ? "High" : "Low"}
        </button>
        <span className="divider" aria-hidden="true" />
        <button type="button" id="world" className={`dock-btn ${worldOpen ? "on" : ""}`} onClick={() => setWorldOpen((o) => !o)}>
          <Icon d={I.globe} size={16} /><span>World <span className="mono dim">{settings.scenario}</span></span>
        </button>
        <div className="phone-tabs">
          <button type="button" className={`dock-btn ${sheet === "brains" ? "on" : ""}`} onClick={() => setSheet((s) => (s === "brains" ? "none" : "brains"))}><Icon d={I.brain} size={16} /><span>Brains</span></button>
          <button type="button" className={`dock-btn ${sheet === "chart" ? "on" : ""}`} onClick={() => setSheet((s) => (s === "chart" ? "none" : "chart"))}><Icon d={I.chart} size={16} /><span>Chart</span></button>
          <button type="button" className={`dock-btn ${sheet === "world" ? "on" : ""}`} onClick={() => setSheet((s) => (s === "world" ? "none" : "world"))}><Icon d={I.globe} size={16} /><span>World</span></button>
        </div>
      </nav>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BRAIN_INDEX } from "./brainIndex";
import type { BrainName, Frame, Selected, SimSettings, ToWorker } from "./protocol";
import { drawChart } from "./view/Chart";
import { WorldView } from "./view/WorldView";
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
  ppo: "Reinforcement learning (PPO): one shared 64-64 network per species, trained by gradient steps on every animal's reward.",
  ga: "Genetic algorithm: 50 networks share the world, the best 5 survive each generation and the rest are mutated copies.",
  es: "Evolution strategies: 25 pairs of opposite weight nudges estimate which direction raises fitness.",
  neat: "NEAT: starts from direct input-to-output wiring and grows hidden neurons and connections as it evolves.",
  hybrid: "Half the budget evolving with GA, then PPO fine-tunes the best evolved network.",
};

const SCENARIO_INFO: Record<string, string> = {
  S1: "Baseline rules",
  S2: "Grass regrows at half speed",
  S3: "Prey 30% faster",
  S4: "Twice as many of both at start",
  S5: "Twice as many predators at start",
};

const SPEEDS = [
  { v: 5, label: "5" },
  { v: 15, label: "15" },
  { v: 30, label: "30" },
  { v: 60, label: "60" },
  { v: 0, label: "Max" },
];

const HISTORY = 4000;
const KINDS = ["grass", "prey", "predator", "refuge", "nothing"];

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export default function App() {
  const [settings, setSettings] = useState<SimSettings>({
    base: "s1_large.json", scenario: "S1", seed: 1, prey: "ppo", predator: "ppo",
  });
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(30);
  const [chartMode, setChartMode] = useState<"time" | "phase">("time");
  const [stats, setStats] = useState({ t: 0, prey: 0, pred: 0, sps: 0, caught: 0 });
  const [extinct, setExtinct] = useState<Frame["extinct"]>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [seedText, setSeedText] = useState("1");

  const worker = useRef<Worker | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const view = useRef<WorldView | null>(null);
  const chart = useRef<HTMLCanvasElement>(null);
  const hist = useRef<{ prey: number[]; pred: number[]; t0: number; caught: number[] }>({ prey: [], pred: [], t0: 0, caught: [] });
  const live = useRef({ running, speed });
  live.current = { running, speed };

  const send = useCallback((m: ToWorker) => worker.current?.postMessage(m), []);

  const redrawChart = useCallback(() => {
    const cv = chart.current;
    if (!cv) return;
    const h = hist.current;
    drawChart(cv, h.prey, h.pred, h.t0, chartModeRef.current, {
      fg: cssVar("--fg"), muted: cssVar("--muted"), line: cssVar("--line"), prey: cssVar("--prey"), predator: cssVar("--predator"),
    });
  }, []);
  const chartModeRef = useRef(chartMode);
  chartModeRef.current = chartMode;

  // world view + worker, created once
  useEffect(() => {
    const v = new WorldView(stage.current!, (x, y) => send({ type: "select", x, y }));
    view.current = v;
    const w = new SimWorker();
    worker.current = w;
    let pending: Frame | null = null;
    let raf = 0;
    const paint = () => {
      raf = 0;
      const f = pending;
      if (!f) return;
      pending = null;
      v.update(f);
    };
    w.onmessage = (e: MessageEvent<Frame>) => {
      const f = e.data;
      const h = hist.current;
      if (f.t === 0 || f.t < h.t0 + h.prey.length - 1) {
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
      setStats({
        t: f.t, prey: f.prey.x.length, pred: f.pred.x.length, sps: f.stepsPerSec,
        caught: h.caught.reduce((a, b) => a + b, 0),
      });
      setExtinct(f.extinct);
      setSelected(f.selected);
      // keep the latest frame only; fold any grass/refuge updates forward so none are lost
      if (pending) {
        f.grass ??= pending.grass;
        f.refuge ??= pending.refuge;
        f.refugeCentres ??= pending.refugeCentres;
      }
      pending = f;
      if (!raf) raf = requestAnimationFrame(paint);
      redrawChart();
    };
    return () => {
      w.terminate();
      v.dispose();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [send, redrawChart]);

  // (re)start the world whenever world-level settings change
  useEffect(() => {
    send({ type: "init", settings, running: live.current.running, speed: live.current.speed });
    // brains are swapped live below; only world settings restart
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.base, settings.scenario, settings.seed, send]);

  useEffect(() => send({ type: "brain", species: "prey", brain: settings.prey }), [settings.prey, send]);
  useEffect(() => send({ type: "brain", species: "predator", brain: settings.predator }), [settings.predator, send]);
  useEffect(() => send({ type: "run", running }), [running, send]);
  useEffect(() => send({ type: "speed", stepsPerSec: speed }), [speed, send]);
  useEffect(() => redrawChart(), [chartMode, redrawChart]);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const h = () => redrawChart();
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, [redrawChart]);

  // keyboard: space play/pause, "." single step
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
      } else if (e.key === "Escape") send({ type: "deselect" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send]);

  const notes = useMemo(() => {
    const n: Record<string, string> = {};
    for (const b of BRAIN_INDEX) if (b.note) n[`${b.method}-${b.species}`] = b.note;
    return n;
  }, []);

  const reset = () => {
    const s = parseInt(seedText, 10);
    const seed = Number.isFinite(s) && s >= 0 ? s : settings.seed;
    setSeedText(String(seed));
    setSettings((cur) => ({ ...cur, seed }));
    if (seed === settings.seed) send({ type: "init", settings: { ...settings, seed }, running, speed });
  };

  const brainRow = (sp: "prey" | "predator") => (
    <fieldset className="brain-row" aria-label={`${sp} controller`}>
      <legend>
        <span className={`swatch ${sp}`} aria-hidden="true" />
        {sp === "prey" ? "Prey" : "Predators"}
      </legend>
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

  const busy = selected && selected.digest > 0;
  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <h1>EcoArena</h1>
          <p>Predators and prey in one shared world, each species run by the brain you pick.</p>
        </div>
        <div className="transport" role="group" aria-label="Playback">
          <button type="button" id="play" className="btn primary" onClick={() => setRunning((r) => !r)}>
            {running ? "Pause" : "Play"}
          </button>
          <button type="button" id="step" className="btn" onClick={() => { setRunning(false); send({ type: "step" }); }}>
            Step
          </button>
          <div className="speed" role="radiogroup" aria-label="Steps per second">
            {SPEEDS.map((s) => (
              <button key={s.v} type="button" id={`speed-${s.label}`} role="radio" aria-checked={speed === s.v}
                className={`seg ${speed === s.v ? "on" : ""}`} onClick={() => setSpeed(s.v)}>
                {s.label}
              </button>
            ))}
            <span className="unit">steps/s</span>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="stage-wrap" aria-label="World">
          <div className="stage" ref={stage} />
          <div className="hud">
            <span className="mono">t = {stats.t.toLocaleString()}</span>
            <span className="mono dim">{stats.sps > 0 && running ? `${Math.round(stats.sps)} steps/s` : running ? "" : "paused"}</span>
          </div>
          {extinct ? (
            <div className="banner" role="status">
              {extinct.species === "prey" ? "Prey" : "Predators"} died out at step {extinct.t.toLocaleString()}.
              <button type="button" className="btn small" onClick={() => { setSeedText(String(settings.seed + 1)); setSettings((s) => ({ ...s, seed: s.seed + 1 })); }}>
                New world
              </button>
            </div>
          ) : null}
          <ul className="legend" aria-label="Map key">
            <li><span className="key tri prey" aria-hidden="true" />Prey</li>
            <li><span className="key tri predator" aria-hidden="true" />Predator</li>
            <li><span className="key tri busy" aria-hidden="true" />Predator digesting or resting</li>
            <li><span className="key refuge" aria-hidden="true" />Refuge</li>
            <li><span className="key grass" aria-hidden="true" />Grass</li>
            <li className="hint">Click an animal to see what it sees</li>
          </ul>
        </section>

        <aside className="panel">
          <section className="block counts" aria-label="Populations">
            <div className="count">
              <span className="label"><span className="swatch prey" aria-hidden="true" />Prey</span>
              <span className="num">{stats.prey.toLocaleString()}</span>
            </div>
            <div className="count">
              <span className="label"><span className="swatch predator" aria-hidden="true" />Predators</span>
              <span className="num">{stats.pred.toLocaleString()}</span>
            </div>
            <div className="count">
              <span className="label">Caught, last second</span>
              <span className="num small">{stats.caught}</span>
            </div>
          </section>

          <section className="block">
            <div className="block-head">
              <h2>Populations</h2>
              <div className="speed" role="radiogroup" aria-label="Chart type">
                <button type="button" id="chart-time" role="radio" aria-checked={chartMode === "time"} className={`seg ${chartMode === "time" ? "on" : ""}`} onClick={() => setChartMode("time")}>Over time</button>
                <button type="button" id="chart-phase" role="radio" aria-checked={chartMode === "phase"} className={`seg ${chartMode === "phase" ? "on" : ""}`} onClick={() => setChartMode("phase")}>Prey vs predators</button>
              </div>
            </div>
            <canvas ref={chart} className="chart" aria-label="Population chart" />
            <p className="caption">
              {chartMode === "time"
                ? "Last 4,000 steps. Prey on the left axis, predators on the right."
                : "Prey across, predators up, one point per step, newest darkest. Lotka–Volterra cycles trace loops."}
            </p>
          </section>

          {selected ? (
            <section className="block inspect" aria-label="Selected animal">
              <div className="block-head">
                <h2>
                  <span className={`swatch ${selected.species}`} aria-hidden="true" />
                  {selected.species === "prey" ? "Prey" : "Predator"} <span className="mono dim">#{selected.slot}</span>
                </h2>
                <button type="button" className="btn small" onClick={() => send({ type: "deselect" })}>Close</button>
              </div>
              {selected.alive ? (
                <>
                  <dl className="facts">
                    <div><dt>Energy</dt><dd><meter min={0} max={selected.energyCap} value={selected.energy} /> <span className="mono">{selected.energy.toFixed(0)}</span></dd></div>
                    <div><dt>Age</dt><dd className="mono">{selected.age} / {selected.maxAge}</dd></div>
                    <div><dt>Speed</dt><dd className="mono">{selected.speed.toFixed(2)} / {selected.maxSpeed}</dd></div>
                    <div><dt>State</dt><dd>{selected.inRefuge ? "In a refuge, safe" : busy ? (selected.digest > 41 ? "Digesting, can't hunt" : "Resting after a lunge, can't hunt") : selected.species === "predator" ? "Hunting" : "Out in the open"}</dd></div>
                    <div><dt>Action</dt><dd className="mono">turn {selected.action[0].toFixed(2)}, thrust {selected.action[1].toFixed(2)}</dd></div>
                  </dl>
                  <div className="sectors" aria-label="What each vision sector reports, left to right">
                    {Array.from({ length: (selected.obs.length - 6) / 6 }, (_, r) => {
                      const o = r * 6;
                      const k = [1, 2, 3, 4, 5].findIndex((c) => selected.obs[o + c] > 0.5);
                      const kind = KINDS[k < 0 ? 4 : k];
                      const near = kind === "nothing" ? 0 : 1 - selected.obs[o];
                      return <span key={r} className={`sector k-${kind}`} style={{ "--near": near.toFixed(2) } as React.CSSProperties} title={`${kind}${kind === "nothing" ? "" : ` at ${(selected.obs[o] * selected.range).toFixed(1)}`}`} />;
                    }).reverse()}
                  </div>
                  <p className="caption">Its 16 vision sectors, left to right. Brighter means closer. This strip plus energy, speed, age, rest state, grass underfoot and refuge make up the 102 numbers its brain reads.</p>
                </>
              ) : (
                <p className="about">Died at step {selected.diedAt?.toLocaleString()}.</p>
              )}
            </section>
          ) : null}

          <section className="block">
            <h2>Brains</h2>
            {brainRow("prey")}
            {brainRow("predator")}
            <p className="caption">
              Each trained network learned against hand-coded opponents, so two trained species meeting here is new to both.
              Switching a brain takes effect on the next step; the world keeps going.
            </p>
          </section>

          <section className="block">
            <h2>World</h2>
            <div className="scenarios" role="radiogroup" aria-label="Scenario">
              {Object.entries(SCENARIO_INFO).map(([k, label]) => (
                <button key={k} type="button" id={`scenario-${k}`} role="radio" aria-checked={settings.scenario === k}
                  className={`scenario ${settings.scenario === k ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, scenario: k }))}>
                  <span className="mono">{k}</span>
                  <span>{label}</span>
                </button>
              ))}
            </div>
            <div className="row">
              <div className="speed" role="radiogroup" aria-label="World size">
                <button type="button" id="size-large" role="radio" aria-checked={settings.base === "s1_large.json"} className={`seg ${settings.base === "s1_large.json" ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, base: "s1_large.json" }))}>200 × 200</button>
                <button type="button" id="size-small" role="radio" aria-checked={settings.base === "s1_balanced.json"} className={`seg ${settings.base === "s1_balanced.json" ? "on" : ""}`} onClick={() => setSettings((s) => ({ ...s, base: "s1_balanced.json" }))}>100 × 100</button>
              </div>
              <label className="seed" htmlFor="seed">
                Seed
                <input id="seed" inputMode="numeric" value={seedText} onChange={(e) => setSeedText(e.target.value.replace(/[^0-9]/g, ""))}
                  onKeyDown={(e) => { if (e.key === "Enter") reset(); }} />
              </label>
              <button type="button" id="reset" className="btn" onClick={reset}>Restart</button>
            </div>
            <p className="caption">
              The benchmark world is 200 × 200 with 48 refuges where prey can't be caught. This browser version follows the same
              rules as the Python benchmark and is checked against it step by step.
            </p>
          </section>
        </aside>
      </main>
    </div>
  );
}

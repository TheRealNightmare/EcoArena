import { useEffect, useRef, useState } from "react";
import { IslandBuilder } from "./IslandBuilder";
import "./loader.css";

/**
 * Full-screen loading screen. `done` is how many of `steps` have really finished. The bar never runs ahead of
 * the work, creeps within the current step so it never looks stuck, and takes at least `minMs` so it never
 * flashes. A click or key skips the minimum time once the work is done.
 *
 * Hand-over, designed so there is no visible cut:
 *   1. settle (600 ms): text and ring fade, the little island glides to the centre of the screen, stops turning
 *   2. reveal: onReveal(frac, ms) starts the real scene far out, where its island looks exactly as big as the
 *      little one (frac), and returns how much it will grow; both cameras then fly in on the same curve while
 *      the dusk sky, then the little island, dissolve
 *   3. finish: onFinished unmounts the loader
 */
export default function Loader({ steps, done, minMs, title, onReveal, onFinished }: {
  steps: string[];
  done: number;
  minMs: number;
  title: string;
  onReveal?: (frac: number, ms: number) => number | undefined;
  onFinished: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const island = useRef<IslandBuilder | null>(null);
  const [p, setP] = useState(0);
  const [phase, setPhase] = useState<"" | "settle" | "reveal">("");
  const stage = useRef<HTMLDivElement>(null);
  const live = useRef({ start: performance.now(), stepStart: performance.now(), lastDone: 0, skip: false, exiting: false });
  const props = useRef({ done, n: steps.length, minMs, onReveal, onFinished });
  props.current = { done, n: steps.length, minMs, onReveal, onFinished };

  useEffect(() => {
    const b = new IslandBuilder(host.current!);
    island.current = b;
    return () => b.dispose();
  }, []);

  useEffect(() => {
    const L = live.current;
    if (done !== L.lastDone) {
      L.lastDone = done;
      L.stepStart = performance.now();
    }
  }, [done]);

  useEffect(() => {
    let raf = 0;
    const L = live.current;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const { done: d, n, minMs: min } = props.current;
      const now = performance.now();
      const creep = d < n ? Math.min(0.85, (now - L.stepStart) / 1600) : 0;
      const work = (d + creep) / n;
      const time = L.skip ? 1 : (now - L.start) / min;
      const v = Math.min(work, time, 1);
      setP(v);
      island.current?.setProgress(v);
      if (d >= n && time >= 1 && !L.exiting) {
        L.exiting = true;
        const FLY = reduce ? 1 : 1700;
        // 1. settle: move the island to the screen centre (where the real island will appear)
        const st = stage.current;
        if (st) {
          const r = st.getBoundingClientRect();
          st.style.setProperty("--dx", `${window.innerWidth / 2 - (r.left + r.width / 2)}px`);
          st.style.setProperty("--dy", `${window.innerHeight / 2 - (r.top + r.height / 2)}px`);
        }
        setPhase("settle");
        setTimeout(() => {
          // 2. reveal: line the real island up with this one and fly both in together
          const r = stage.current?.getBoundingClientRect();
          const canvasFrac = island.current?.apparentRadius() ?? 0.8;
          const frac = r ? (canvasFrac * r.height) / window.innerHeight : 0.35;
          const ratio = props.current.onReveal?.(frac, FLY) ?? 3;
          island.current?.flyIn(FLY, ratio);
          setPhase("reveal");
          setTimeout(() => props.current.onFinished(), reduce ? 250 : 1500);
        }, reduce ? 0 : 620);
      }
    };
    raf = requestAnimationFrame(tick);
    const skip = () => {
      L.skip = true;
    };
    window.addEventListener("pointerdown", skip);
    window.addEventListener("keydown", skip);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointerdown", skip);
      window.removeEventListener("keydown", skip);
    };
  }, []);

  const pct = Math.round(p * 100);
  const R = 46, C = 2 * Math.PI * R;
  const current = Math.min(done, steps.length - 1);
  return (
    <div className={`loader ${phase}`} role="status" aria-live="polite" aria-label={`${title}: ${pct}%`}>
      <div className="loader-sky" aria-hidden="true" />
      <div className="loader-stage" ref={stage}>
        <div className="loader-island" ref={host} aria-hidden="true" />
        <svg className="loader-ring" viewBox="0 0 100 100" aria-hidden="true">
          <circle cx="50" cy="50" r={R} className="track" />
          <circle cx="50" cy="50" r={R} className="fill" strokeDasharray={C} strokeDashoffset={C * (1 - p)} />
        </svg>
      </div>
      <div className="loader-text">
        <p className="loader-title">{title}</p>
        <ol className="loader-steps">
          {steps.map((s, i) => (
            <li key={s} className={i < done ? "done" : i === current ? "now" : "next"}>
              <span className="mark" aria-hidden="true" />
              {s}
            </li>
          ))}
        </ol>
        <p className="loader-pct mono">{pct}%</p>
        {done >= steps.length ? <p className="loader-hint">Click or press any key to skip</p> : null}
      </div>
    </div>
  );
}

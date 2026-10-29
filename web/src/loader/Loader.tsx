import { useEffect, useRef, useState } from "react";
import { IslandBuilder } from "./IslandBuilder";
import "./loader.css";

/**
 * Full-screen loading screen. `done` is how many of `steps` have really finished. The bar never runs ahead of
 * the work, creeps within the current step so it never looks stuck, and takes at least `minMs` so it never
 * flashes. When everything is done: the camera pushes into the little island, `onReveal` fires (start the real
 * scene's fly-in), the screen dissolves, then `onFinished` unmounts it. A click or key skips the minimum time.
 */
export default function Loader({ steps, done, minMs, title, onReveal, onFinished }: {
  steps: string[];
  done: number;
  minMs: number;
  title: string;
  onReveal?: () => void;
  onFinished: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const island = useRef<IslandBuilder | null>(null);
  const [p, setP] = useState(0);
  const [exiting, setExiting] = useState(false);
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
        // let the finished island sit for a beat, then push through it
        setTimeout(() => {
          setExiting(true);
          props.current.onReveal?.();
          const finish = () => setTimeout(() => props.current.onFinished(), reduce ? 200 : 450);
          if (reduce) finish();
          else island.current?.pushThrough(finish);
        }, reduce ? 0 : 380);
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
    <div className={`loader ${exiting ? "exiting" : ""}`} role="status" aria-live="polite" aria-label={`${title}: ${pct}%`}>
      <div className="loader-sky" aria-hidden="true" />
      <div className="loader-stage">
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

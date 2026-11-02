// Population chart on a 2D canvas: time series (prey left axis, predators right axis) or the prey-predator
// phase plane, where Lotka-Volterra cycles show up as loops.

export interface ChartColors {
  fg: string;
  muted: string;
  line: string;
  prey: string;
  predator: string;
}

/** Axis top as 4 equal ticks of a round step (1, 2, 2.5, 3, 4, 5, 6, 8 x 10^k), so every label is a whole number. */
function niceMax(v: number): number {
  const step = Math.max(v, 4) / 4;
  const p = Math.pow(10, Math.floor(Math.log10(step)));
  for (const m of [1, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= step && Number.isInteger(m * p)) return 4 * m * p;
  return 40 * p;
}

export function drawChart(cv: HTMLCanvasElement, prey: number[], pred: number[], t0: number, mode: "time" | "phase", c: ChartColors) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
  }
  const g = cv.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  g.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  g.textBaseline = "middle";
  const n = prey.length;
  if (n < 2) return;
  const pad = { l: 40, r: 40, t: 10, b: 22 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const maxPrey = niceMax(Math.max(...prey));
  const maxPred = niceMax(Math.max(...pred));

  g.strokeStyle = c.line;
  g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const y = pad.t + ih - (k / 4) * ih;
    g.beginPath();
    g.moveTo(pad.l, Math.round(y) + 0.5);
    g.lineTo(pad.l + iw, Math.round(y) + 0.5);
    g.stroke();
  }

  if (mode === "time") {
    const X = (i: number) => pad.l + (i / (n - 1)) * iw;
    const series = (vals: number[], max: number, color: string, fill: boolean) => {
      g.beginPath();
      for (let i = 0; i < n; i++) {
        const y = pad.t + ih - (vals[i] / max) * ih;
        if (i === 0) g.moveTo(X(i), y);
        else g.lineTo(X(i), y);
      }
      g.strokeStyle = color;
      g.lineWidth = 1.6;
      g.stroke();
      if (fill) {
        g.lineTo(X(n - 1), pad.t + ih);
        g.lineTo(X(0), pad.t + ih);
        g.closePath();
        g.globalAlpha = 0.12;
        g.fillStyle = color;
        g.fill();
        g.globalAlpha = 1;
      }
      const yEnd = pad.t + ih - (vals[n - 1] / max) * ih;
      g.fillStyle = color;
      g.beginPath();
      g.arc(X(n - 1), yEnd, 2.6, 0, Math.PI * 2);
      g.fill();
    };
    series(prey, maxPrey, c.prey, true);
    series(pred, maxPred, c.predator, false);
    for (let k = 0; k <= 4; k++) {
      const y = pad.t + ih - (k / 4) * ih;
      g.fillStyle = c.prey;
      g.textAlign = "right";
      g.fillText(String(Math.round((maxPrey * k) / 4)), pad.l - 6, y);
      g.fillStyle = c.predator;
      g.textAlign = "left";
      g.fillText(String(Math.round((maxPred * k) / 4)), pad.l + iw + 6, y);
    }
    g.fillStyle = c.muted;
    g.textAlign = "left";
    g.fillText(`step ${t0.toLocaleString()}`, pad.l, H - 9);
    g.textAlign = "right";
    g.fillText(`${(t0 + n - 1).toLocaleString()}`, pad.l + iw, H - 9);
  } else {
    const X = (v: number) => pad.l + (v / maxPrey) * iw;
    const Y = (v: number) => pad.t + ih - (v / maxPred) * ih;
    const stride = Math.max(1, Math.floor(n / 1500));
    for (let i = stride; i < n; i += stride) {
      g.strokeStyle = c.fg;
      g.globalAlpha = 0.08 + 0.7 * (i / n);
      g.lineWidth = 1.3;
      g.beginPath();
      g.moveTo(X(prey[i - stride]), Y(pred[i - stride]));
      g.lineTo(X(prey[i]), Y(pred[i]));
      g.stroke();
    }
    g.globalAlpha = 1;
    g.fillStyle = c.predator;
    g.beginPath();
    g.arc(X(prey[n - 1]), Y(pred[n - 1]), 3.2, 0, Math.PI * 2);
    g.fill();
    for (let k = 0; k <= 4; k++) {
      g.fillStyle = c.predator;
      g.textAlign = "right";
      g.fillText(String(Math.round((maxPred * k) / 4)), pad.l - 6, pad.t + ih - (k / 4) * ih);
    }
    g.fillStyle = c.prey;
    g.textAlign = "center";
    for (let k = 0; k <= 4; k++) g.fillText(String(Math.round((maxPrey * k) / 4)), pad.l + (k / 4) * iw, H - 9);
  }
}

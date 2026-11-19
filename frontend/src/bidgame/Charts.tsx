import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { signed } from "./format";

/* ------------------------------------------------------------ tooltip */

interface TipState {
  x: number;
  y: number;
  lines: ReactNode[];
}

function TipBox({ tip }: { tip: TipState }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: tip.x + 14, top: tip.y - 60 });

  useLayoutEffect(() => {
    const box = ref.current!.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - box.width - 8, tip.x + 14));
    const top = tip.y - box.height - 12 < 8 ? tip.y + 16 : tip.y - box.height - 12;
    setPos({ left, top });
  }, [tip]);

  return (
    <div ref={ref} className="bg-tip" style={pos} role="status">
      {tip.lines.map((line, i) =>
        i === 0 ? <b key={i}>{line}</b> : <div key={i}>{line}</div>,
      )}
    </div>
  );
}

function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  return {
    show: (event: React.PointerEvent, lines: ReactNode[]) =>
      setTip({ x: event.clientX, y: event.clientY, lines }),
    hide: () => setTip(null),
    node: tip && <TipBox tip={tip} />,
  };
}

export function Swatch({ color, kind = "dot" }: { color?: string; kind?: "dot" | "square" | "dash" }) {
  return (
    <i
      className={`bg-swatch bg-swatch-${kind}`}
      style={kind === "dash" ? undefined : { background: color }}
      aria-hidden="true"
    />
  );
}

/* ------------------------------------------------------------ helpers */

function niceTicks(lo: number, hi: number, count = 4) {
  const raw = (hi - lo || 1) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw)!;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) {
    ticks.push(Math.round(t * 1e6) / 1e6);
  }
  return ticks;
}

/** A bar from the zero line at y0 to y1, rounded at its outer end only. */
function barPath(x: number, w: number, y0: number, y1: number) {
  if (Math.abs(y1 - y0) < 1.5) y1 = y0 + (y1 > y0 ? 1.5 : -1.5);
  const r = Math.min(4, w / 2, Math.abs(y1 - y0));
  const d = y1 < y0 ? r : -r;
  return `M${x},${y0}L${x},${y1 + d}Q${x},${y1} ${x + r},${y1}L${x + w - r},${y1}Q${x + w},${y1} ${x + w},${y1 + d}L${x + w},${y0}Z`;
}

/* ------------------------------------------------------------ bar chart */

export interface BarItem {
  label: string;
  value: number;
  /** Drawn in brass: the best bid, or the trick count that makes the bid. */
  highlight: boolean;
  /** Colours of the players' dots under this bar. */
  marks: string[];
  showValue: boolean;
  text: string;
  tip: ReactNode[];
}

interface BarChartProps {
  items: BarItem[];
  axis: string;
  label: string;
  format: (value: number) => string;
  floor?: number;
  onPick?: (index: number) => void;
}

export function BarChart({ items, axis, label, format, floor, onPick }: BarChartProps) {
  const tip = useTip();
  const [hover, setHover] = useState<number | null>(null);
  const W = 400;
  const H = 240;
  const m = { l: 40, r: 8, t: 20, b: 50 };

  const values = items.map((item) => item.value);
  let lo = floor ?? Math.min(0, ...values);
  let hi = Math.max(0, ...values);
  if (hi === lo) hi = lo + 1;
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0]);
  hi = Math.max(hi, ticks[ticks.length - 1]);
  const pad = (hi - lo) * 0.08;
  if (lo < 0) lo -= pad;
  hi += pad;

  const y = (v: number) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);
  const band = (W - m.l - m.r) / items.length;
  const width = Math.max(4, Math.min(40, band - 4));
  const crowded = items.length > 9 && band < 30;

  return (
    <>
      <svg className="bg-chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="bg-grid" />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end">
              {format(t)}
            </text>
          </g>
        ))}
        {items.map((item, i) => {
          const cx = m.l + band * (i + 0.5);
          const up = item.value >= 0;
          return (
            <g key={item.label}>
              <path
                d={barPath(cx - width / 2, width, y(0), y(item.value))}
                className={item.highlight ? "bg-bar bg-bar-best" : "bg-bar"}
                opacity={hover === i ? 0.8 : 1}
              />
              {item.showValue && (
                <text x={cx} y={y(item.value) + (up ? -5 : 14)} textAnchor="middle" className="strong">
                  {item.text}
                </text>
              )}
              {(!crowded || i % 2 === 0 || item.highlight || item.marks.length > 0) && (
                <text x={cx} y={H - m.b + 16} textAnchor="middle" className={item.highlight ? "strong" : ""}>
                  {item.label}
                </text>
              )}
              {item.marks.map((color, k) => (
                <circle
                  key={k}
                  cx={cx + (k - (item.marks.length - 1) / 2) * 13}
                  cy={H - m.b + 27}
                  r={5}
                  fill={color}
                  className="bg-ring"
                />
              ))}
              <rect
                x={m.l + band * i}
                y={m.t}
                width={band}
                height={H - m.t - 14}
                fill="transparent"
                style={onPick ? { cursor: "pointer" } : undefined}
                onPointerMove={(event) => {
                  setHover(i);
                  tip.show(event, item.tip);
                }}
                onPointerLeave={() => {
                  setHover(null);
                  tip.hide();
                }}
                onClick={onPick ? () => onPick(i) : undefined}
              />
            </g>
          );
        })}
        <line x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} className="bg-baseline" />
        <text x={m.l + (W - m.l - m.r) / 2} y={H - 2} textAnchor="middle">
          {axis}
        </text>
      </svg>
      {tip.node}
    </>
  );
}

/* ------------------------------------------------------------ line chart */

export interface Series {
  name: string;
  color: string;
  dashed?: boolean;
  /** Running total, starting with 0 before the first hand. */
  points: number[];
}

export function LineChart({ series }: { series: Series[] }) {
  const tip = useTip();
  const [hover, setHover] = useState<number | null>(null);
  const W = 780;
  const H = 300;
  const m = { l: 44, r: 130, t: 14, b: 40 };

  const all = series.flatMap((s) => s.points);
  let lo = Math.min(0, ...all);
  let hi = Math.max(0, ...all);
  if (hi === lo) hi = lo + 1;
  const ticks = niceTicks(lo, hi, 5);
  lo = Math.min(lo, ticks[0]);
  hi = Math.max(hi, ticks[ticks.length - 1]);
  const n = series[0].points.length - 1;
  const x = (i: number) => m.l + (i / n) * (W - m.l - m.r);
  const y = (v: number) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);

  // End labels, nudged apart when they would overlap.
  const ends = series
    .map((s) => ({ s, at: y(s.points[n]) }))
    .sort((a, b) => a.at - b.at);
  for (let i = 1; i < ends.length; i++) ends[i].at = Math.max(ends[i].at, ends[i - 1].at + 15);

  const pick = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const i = Math.max(1, Math.min(n, Math.round(((event.clientX - box.left) / box.width) * n)));
    setHover(i);
    tip.show(event, [
      `After hand ${i}`,
      ...series.map((s) => (
        <span className="bg-tip-row" key={s.name}>
          {s.dashed ? <Swatch kind="dash" /> : <Swatch color={s.color} />}
          {s.name} <b>{signed(s.points[i])}</b>
        </span>
      )),
    ]);
  };

  return (
    <>
      <svg className="bg-chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Running score by hand">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className={t === 0 ? "bg-baseline" : "bg-grid"} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        {Array.from({ length: n }, (_, k) => k + 1).map((i) => (
          <text key={i} x={x(i)} y={H - m.b + 16} textAnchor="middle">
            {i}
          </text>
        ))}
        <text x={m.l + (W - m.l - m.r) / 2} y={H - 4} textAnchor="middle">
          Hand
        </text>
        {[...series].reverse().map((s) => (
          <g key={s.name}>
            <polyline
              points={s.points.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeDasharray={s.dashed ? "5 5" : undefined}
            />
            {!s.dashed &&
              s.points.map((v, i) =>
                i ? <circle key={i} cx={x(i)} cy={y(v)} r={4} fill={s.color} className="bg-ring" /> : null,
              )}
          </g>
        ))}
        {ends.map(({ s, at }) => (
          <text key={s.name} x={W - m.r + 12} y={at + 4} className="strong">
            {s.name.length > 10 ? `${s.name.slice(0, 9)}…` : s.name} {signed(s.points[n])}
          </text>
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={m.t} y2={H - m.b} className="bg-baseline" />}
        <rect
          x={m.l}
          y={m.t}
          width={W - m.l - m.r}
          height={H - m.t - m.b}
          fill="transparent"
          onPointerMove={pick}
          onPointerLeave={() => {
            setHover(null);
            tip.hide();
          }}
        />
      </svg>
      {tip.node}
    </>
  );
}

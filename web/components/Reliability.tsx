import type { CalibrationReport } from "@/lib/types";

const SIZE = 300;
const PAD = 30;

function scale(v: number): number {
  return PAD + v * (SIZE - 2 * PAD);
}

export default function Reliability({ report }: { report: CalibrationReport }) {
  const maxCount = Math.max(...report.curve.map((b) => b.count), 1);
  // Area, not radius, encodes bin count (SPEC.md §13) -- equal-sized markers
  // over unequal bins is the standard way this chart misleads.
  const radiusFor = (count: number) => 2 + 10 * Math.sqrt(count / maxCount);

  return (
    <div>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label="Reliability diagram: predicted probability vs. observed frequency"
        style={{ width: "100%", maxWidth: 360, height: "auto" }}
      >
        {/* Perfect calibration reference */}
        <line
          x1={scale(0)}
          y1={scale(1)}
          x2={scale(1)}
          y2={scale(0)}
          stroke="var(--color-border)"
          strokeDasharray="4,4"
        />
        <line x1={PAD} y1={SIZE - PAD} x2={SIZE - PAD} y2={SIZE - PAD} stroke="var(--color-border)" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={SIZE - PAD} stroke="var(--color-border)" />

        {report.curve.map((bin, i) => (
          <circle
            key={i}
            cx={scale(bin.mean_predicted)}
            cy={scale(1 - bin.mean_observed)}
            r={radiusFor(bin.count)}
            fill="var(--color-taken)"
            fillOpacity={0.7}
            stroke="var(--color-taken)"
          />
        ))}

        <text x={SIZE / 2} y={SIZE - 8} fill="var(--color-text-muted)" fontSize={10} textAnchor="middle">
          mean predicted
        </text>
        <text
          x={10}
          y={SIZE / 2}
          fill="var(--color-text-muted)"
          fontSize={10}
          textAnchor="middle"
          transform={`rotate(-90 10 ${SIZE / 2})`}
        >
          mean observed
        </text>
      </svg>

      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, auto)",
          gap: "var(--space-1) var(--space-3)",
          marginTop: "var(--space-3)",
          fontFamily: "var(--font-mono)",
          fontSize: "0.85rem",
        }}
      >
        <dt style={{ color: "var(--color-text-muted)" }}>Brier</dt>
        <dd>{report.brier.toFixed(5)}</dd>
        <dt style={{ color: "var(--color-text-muted)" }}>ECE</dt>
        <dd>{report.ece.toFixed(5)}</dd>
        <dt style={{ color: "var(--color-text-muted)" }}>Reliability</dt>
        <dd>{report.reliability.toFixed(5)}</dd>
        <dt style={{ color: "var(--color-text-muted)" }}>Resolution</dt>
        <dd>{report.resolution.toFixed(5)}</dd>
        <dt style={{ color: "var(--color-text-muted)" }}>n ({report.split})</dt>
        <dd>{report.n.toLocaleString()}</dd>
      </dl>
    </div>
  );
}

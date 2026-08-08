import type { Moment } from "@/lib/types";

// SPEC.md §3: the API emits SPADL coordinates only (105 x 68, bottom-left
// origin, y increases upward). This is the ONE place that flips y into SVG
// space (top-left origin, y increases downward) -- nowhere else in the
// frontend touches coordinates directly.
const PITCH_LENGTH = 105;
const PITCH_WIDTH = 68;

function flipY(y: number): number {
  return PITCH_WIDTH - y;
}

export default function Pitch({ moment }: { moment: Moment }) {
  const chosen = moment.chosen;
  const start = { x: chosen.start_x, y: flipY(chosen.start_y) };
  const end = { x: chosen.end_x, y: flipY(chosen.end_y) };

  return (
    <svg
      viewBox={`0 0 ${PITCH_LENGTH} ${PITCH_WIDTH}`}
      role="img"
      aria-label={`${moment.player_name} ${chosen.type_name} in minute ${moment.minute}`}
      style={{ width: "100%", height: "auto", background: "var(--color-surface)" }}
    >
      <defs>
        <marker id="arrow-taken" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-taken)" />
        </marker>
        <marker id="arrow-best" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-best)" />
        </marker>
      </defs>

      {/* Pitch outline */}
      <rect
        x={0.5}
        y={0.5}
        width={PITCH_LENGTH - 1}
        height={PITCH_WIDTH - 1}
        fill="none"
        stroke="var(--color-border)"
        strokeWidth={0.3}
      />
      <line
        x1={PITCH_LENGTH / 2}
        y1={0}
        x2={PITCH_LENGTH / 2}
        y2={PITCH_WIDTH}
        stroke="var(--color-border)"
        strokeWidth={0.3}
      />
      <circle
        cx={PITCH_LENGTH / 2}
        cy={PITCH_WIDTH / 2}
        r={9.15}
        fill="none"
        stroke="var(--color-border)"
        strokeWidth={0.3}
      />
      {/* Penalty boxes (16.5m deep, 40.3m wide, standard pitch markings) */}
      <rect x={0} y={13.85} width={16.5} height={40.3} fill="none" stroke="var(--color-border)" strokeWidth={0.3} />
      <rect
        x={PITCH_LENGTH - 16.5}
        y={13.85}
        width={16.5}
        height={40.3}
        fill="none"
        stroke="var(--color-border)"
        strokeWidth={0.3}
      />

      {/* Unscored / gated options: dashed outlines, never hidden (SPEC.md §13) */}
      {moment.options
        .filter((o) => !o.scored)
        .map((o, i) => (
          <line
            key={`unscored-${i}`}
            x1={start.x}
            y1={start.y}
            x2={o.end_x}
            y2={flipY(o.end_y)}
            stroke="var(--color-unscored)"
            strokeWidth={0.4}
            strokeDasharray="1.2,1"
          />
        ))}

      {/* Scored options that were not taken */}
      {moment.options
        .filter((o) => o.scored)
        .map((o, i) => (
          <line
            key={`scored-${i}`}
            x1={start.x}
            y1={start.y}
            x2={o.end_x}
            y2={flipY(o.end_y)}
            stroke="var(--color-best)"
            strokeWidth={0.3}
            opacity={0.5}
          />
        ))}

      {/* The best available option, if the counterfactual layer has run */}
      {moment.best && (
        <line
          x1={start.x}
          y1={start.y}
          x2={moment.best.end_x}
          y2={flipY(moment.best.end_y)}
          stroke="var(--color-best)"
          strokeWidth={0.6}
          markerEnd="url(#arrow-best)"
        />
      )}

      {/* The action actually taken */}
      <line
        x1={start.x}
        y1={start.y}
        x2={end.x}
        y2={end.y}
        stroke="var(--color-taken)"
        strokeWidth={0.6}
        markerEnd="url(#arrow-taken)"
      />
      <circle cx={start.x} cy={start.y} r={0.9} fill="var(--color-taken)" />
    </svg>
  );
}

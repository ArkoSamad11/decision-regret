import type { Moment } from "@/lib/types";

function formatValue(value: number | null): string {
  if (value === null) return "unscored";
  return value.toFixed(3);
}

export default function OptionLedger({ moment }: { moment: Moment }) {
  const rows = [
    {
      key: "chosen",
      label: "Taken",
      type: moment.chosen.type_name,
      value: moment.chosen.value as number | null,
      scored: true,
    },
    ...(moment.best
      ? [{ key: "best", label: "Best available", type: moment.best.type_name, value: moment.best.value, scored: true }]
      : []),
    ...moment.options.map((o, i) => ({
      key: `option-${i}`,
      label: "Declined",
      type: o.type_name,
      value: o.value,
      scored: o.scored,
    })),
  ];

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "var(--space-2)", color: "var(--color-text-muted)" }}>
              Role
            </th>
            <th style={{ textAlign: "left", padding: "var(--space-2)", color: "var(--color-text-muted)" }}>
              Action
            </th>
            <th style={{ textAlign: "right", padding: "var(--space-2)", color: "var(--color-text-muted)" }}>
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} style={{ borderTop: "1px solid var(--color-border)" }}>
              <td style={{ padding: "var(--space-2)" }}>{r.label}</td>
              <td style={{ padding: "var(--space-2)" }}>{r.type}</td>
              <td
                style={{
                  padding: "var(--space-2)",
                  textAlign: "right",
                  fontFamily: "var(--font-mono)",
                  color: r.scored ? "var(--color-text)" : "var(--color-text-faint)",
                }}
              >
                {formatValue(r.value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p style={{ marginTop: "var(--space-3)", color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
        Regret:{" "}
        <strong style={{ color: "var(--color-text)" }}>
          {moment.regret === null ? "gap unsupported" : moment.regret.toFixed(3)}
        </strong>
        {moment.regret === null && (
          <>
            {" "}
            -- the counterfactual layer has not scored alternatives for this moment yet
            ({moment.unscored_count} option{moment.unscored_count === 1 ? "" : "s"} gated).
          </>
        )}
      </p>
    </div>
  );
}

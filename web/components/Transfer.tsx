import type { TransferLabelReport, TransferReport } from "@/lib/types";

function Row({ report }: { report: TransferLabelReport }) {
  return (
    <table style={{ fontSize: "0.85rem" }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left", padding: "var(--space-1) var(--space-2)", color: "var(--color-text-muted)" }}></th>
          <th style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)", color: "var(--color-text-muted)" }}>
            Brier
          </th>
          <th style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)", color: "var(--color-text-muted)" }}>
            ECE
          </th>
          <th style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)", color: "var(--color-text-muted)" }}>
            n
          </th>
        </tr>
      </thead>
      <tbody style={{ fontFamily: "var(--font-mono)" }}>
        <tr>
          <td style={{ padding: "var(--space-1) var(--space-2)" }}>Source (in-competition)</td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>{report.source.brier.toFixed(5)}</td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>{report.source.ece.toFixed(5)}</td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>{report.source.n.toLocaleString()}</td>
        </tr>
        <tr>
          <td style={{ padding: "var(--space-1) var(--space-2)" }}>Target, before recalibration</td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>
            {report.target_before.brier.toFixed(5)}
          </td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)", color: "var(--color-taken)" }}>
            {report.target_before.ece.toFixed(5)}
          </td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>
            {report.target_before.n.toLocaleString()}
          </td>
        </tr>
        <tr>
          <td style={{ padding: "var(--space-1) var(--space-2)" }}>Target, after recalibration</td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>
            {report.target_after.brier.toFixed(5)}
          </td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)", color: "var(--color-best)" }}>
            {report.target_after.ece.toFixed(5)}
          </td>
          <td style={{ textAlign: "right", padding: "var(--space-1) var(--space-2)" }}>
            {report.target_after.n.toLocaleString()}
          </td>
        </tr>
      </tbody>
    </table>
  );
}

export default function Transfer({ report }: { report: TransferReport }) {
  const scores = report.label_scores;
  const concedes = report.label_concedes;

  return (
    <div>
      <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginBottom: "var(--space-2)" }}>
        Trained on {scores.source.competitions.join(", ")}, evaluated on{" "}
        {scores.target_before.competitions.join(", ")} -- a competition the model never trained on.
      </p>

      <h3 style={{ fontSize: "0.9rem", marginBottom: "var(--space-1)" }}>label_scores</h3>
      <Row report={scores} />
      {scores.ece_recovered_fraction !== null && (
        <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "var(--space-1)" }}>
          ECE degraded {scores.ece_degradation_ratio?.toFixed(2)}x on transfer; recalibrating on a held-out 20%
          of the target recovered {(scores.ece_recovered_fraction * 100).toFixed(0)}% of that gap.
        </p>
      )}

      <h3 style={{ fontSize: "0.9rem", marginTop: "var(--space-3)", marginBottom: "var(--space-1)" }}>
        label_concedes
      </h3>
      <Row report={concedes} />
      {concedes.ece_recovered_fraction !== null && (
        <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "var(--space-1)" }}>
          ECE degraded {concedes.ece_degradation_ratio?.toFixed(2)}x on transfer; recalibrating on a held-out
          20% of the target recovered {(concedes.ece_recovered_fraction * 100).toFixed(0)}% of that gap.
        </p>
      )}
    </div>
  );
}

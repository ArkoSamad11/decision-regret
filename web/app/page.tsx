import { getCalibration, getTransfer, listMatches, listMoments } from "@/lib/api";
import Pitch from "@/components/Pitch";
import OptionLedger from "@/components/OptionLedger";
import Reliability from "@/components/Reliability";
import MomentPicker from "@/components/MomentPicker";
import Transfer from "@/components/Transfer";

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ match?: string; moment?: string }>;
}) {
  const params = await searchParams;
  const matches = await listMatches().catch(() => []);

  if (matches.length === 0) {
    return (
      <main style={{ padding: "var(--space-5)", maxWidth: 640 }}>
        <h1>xDR</h1>
        <p style={{ color: "var(--color-text-muted)" }}>
          No scored data available yet. Run <code>make reproduce</code> (or at minimum{" "}
          <code>make ingest features train calibrate serve-db</code>) against a running API before
          loading this dashboard.
        </p>
      </main>
    );
  }

  const firstMatch = matches[0];
  if (!firstMatch) {
    return null;
  }
  const selectedMatchId = params.match ?? firstMatch.match_id;
  const moments = await listMoments(selectedMatchId).catch(() => []);
  const selectedMoment =
    moments.find((m) => m.moment_id === params.moment) ??
    [...moments].sort((a, b) => b.chosen.value - a.chosen.value)[0];

  const calibration = await getCalibration("test").catch(() => null);
  const transfer = await getTransfer().catch(() => null);

  return (
    <main style={{ padding: "var(--space-4) var(--space-5)", maxWidth: 1100, margin: "0 auto" }}>
      <header style={{ marginBottom: "var(--space-4)" }}>
        <h1 style={{ marginBottom: "var(--space-1)" }}>xDR</h1>
        <p style={{ color: "var(--color-text-muted)", maxWidth: 640 }}>
          Every on-ball action has alternatives the player declined. This pass ships the
          calibration layer -- match-level Brier decomposition and reliability -- against a real
          LightGBM baseline trained on UEFA Euro 2024, plus a cross-tournament transfer study
          against UEFA Women&apos;s Euro 2025. The counterfactual option fan (declined
          passes/carries/shots and the regret gap) is upcoming work; see docs/DECISIONS.md.
        </p>
      </header>

      <section style={{ marginBottom: "var(--space-4)" }}>
        <h2 style={{ fontSize: "1rem", color: "var(--color-text-muted)", marginBottom: "var(--space-2)" }}>
          Matches (ranked by mean action value -- regret ranking arrives with the counterfactual layer)
        </h2>
        <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
          {matches.map((m) => (
            <a
              key={m.match_id}
              href={`?match=${m.match_id}`}
              style={{
                padding: "var(--space-1) var(--space-2)",
                borderRadius: "var(--radius)",
                border: "1px solid var(--color-border)",
                background: m.match_id === selectedMatchId ? "var(--color-surface-raised)" : "transparent",
                fontSize: "0.85rem",
                textDecoration: "none",
                color: "var(--color-text)",
              }}
            >
              {m.home_team} vs {m.away_team}
            </a>
          ))}
        </div>
      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "260px 1fr 320px",
          gap: "var(--space-4)",
          alignItems: "start",
        }}
      >
        <div style={{ background: "var(--color-surface)", borderRadius: "var(--radius)" }}>
          <MomentPicker moments={moments} />
        </div>

        <div>
          {selectedMoment ? (
            <>
              <Pitch moment={selectedMoment} />
              <div style={{ marginTop: "var(--space-3)" }}>
                <OptionLedger moment={selectedMoment} />
              </div>
            </>
          ) : (
            <p style={{ color: "var(--color-text-muted)" }}>No scored moments for this match.</p>
          )}
        </div>

        <div>
          <h2 style={{ fontSize: "1rem", color: "var(--color-text-muted)", marginBottom: "var(--space-2)" }}>
            Calibration (held-out test split)
          </h2>
          {calibration ? (
            <Reliability report={calibration} />
          ) : (
            <p style={{ color: "var(--color-text-muted)" }}>No calibration report available.</p>
          )}
        </div>
      </section>

      <section style={{ marginTop: "var(--space-5)" }}>
        <h2 style={{ fontSize: "1rem", color: "var(--color-text-muted)", marginBottom: "var(--space-2)" }}>
          Transfer study -- source vs. a competition the model never trained on
        </h2>
        {transfer ? (
          <Transfer report={transfer} />
        ) : (
          <p style={{ color: "var(--color-text-muted)" }}>No transfer study available.</p>
        )}
      </section>

      <footer
        style={{
          marginTop: "var(--space-5)",
          paddingTop: "var(--space-3)",
          borderTop: "1px solid var(--color-border)",
          color: "var(--color-text-faint)",
          fontSize: "0.8rem",
        }}
      >
        Data provided by StatsBomb under their open data terms.
      </footer>
    </main>
  );
}

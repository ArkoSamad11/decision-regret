"use client";

import { useRouter, useSearchParams } from "next/navigation";
import type { Moment } from "@/lib/types";

// The only client component in the tree (SPEC.md §13): it needs interaction
// (click to select) and URL state (?moment=), both of which require running
// in the browser. Everything else stays a server component so the API
// origin never enters the client bundle.
export default function MomentPicker({ moments }: { moments: Moment[] }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selected = searchParams.get("moment");

  const sorted = [...moments].sort((a, b) => {
    if (a.regret !== null && b.regret !== null) return b.regret - a.regret;
    return b.chosen.value - a.chosen.value;
  });

  function select(momentId: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("moment", momentId);
    router.push(`?${params.toString()}`);
  }

  return (
    <ul style={{ listStyle: "none", margin: 0, padding: 0, maxHeight: 480, overflowY: "auto" }}>
      {sorted.map((m) => {
        const isSelected = m.moment_id === selected;
        return (
          <li key={m.moment_id}>
            <button
              onClick={() => select(m.moment_id)}
              aria-current={isSelected}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "var(--space-2) var(--space-3)",
                background: isSelected ? "var(--color-surface-raised)" : "transparent",
                border: "none",
                borderLeft: isSelected ? "3px solid var(--color-taken)" : "3px solid transparent",
                color: "var(--color-text)",
                cursor: "pointer",
                fontFamily: "inherit",
                fontSize: "0.9rem",
              }}
            >
              <div>
                {m.minute}&prime;{String(m.second).padStart(2, "0")} -- {m.player_name}
              </div>
              <div style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
                {m.chosen.type_name}
                {" -- "}
                {m.regret === null ? `value ${m.chosen.value.toFixed(3)}` : `regret ${m.regret.toFixed(3)}`}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

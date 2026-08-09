// Typed fetch wrappers. All calls go to same-origin /api/*, which
// next.config.mjs rewrites to API_ORIGIN server-side -- the backend URL never
// reaches the client bundle (SPEC.md §14).

import type { CalibrationReport, HealthResponse, MatchSummary, Moment, TransferReport } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${baseUrl()}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// Server components run outside the browser, so relative /api/* URLs have no
// origin to resolve against -- use API_ORIGIN directly there. Client
// components keep using the same-origin rewrite.
function baseUrl(): string {
  if (typeof window === "undefined") {
    return process.env.API_ORIGIN ?? "http://localhost:8000";
  }
  return "/api";
}

export function getHealth() {
  return getJson<HealthResponse>("/health");
}

export function listMatches() {
  return getJson<MatchSummary[]>("/matches");
}

export function listMoments(matchId: string, minValue?: number) {
  // Named min_value, not min_regret: this pass has no counterfactual layer,
  // so `regret` is always null and filtering on it would always return
  // nothing. SPEC.md's `min_regret` filter lands with the M8 counterfactual
  // work; see docs/DECISIONS.md.
  const qs = minValue !== undefined ? `?min_value=${minValue}` : "";
  return getJson<Moment[]>(`/matches/${matchId}/moments${qs}`);
}

export function getMoment(momentId: string) {
  return getJson<Moment>(`/moments/${momentId}`);
}

export function getCalibration(split: string) {
  return getJson<CalibrationReport>(`/calibration?split=${split}`);
}

export function getTransfer() {
  return getJson<TransferReport>("/transfer");
}

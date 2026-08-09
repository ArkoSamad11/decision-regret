// Mirrors api/src/xdr/serve/schemas.py. SPEC.md §12: drift between these two
// files is a defect the type checker cannot catch -- change both in one commit.

export interface ScoredAction {
  type_name: string;
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
  p_scores: number;
  p_concedes: number;
  value: number;
}

// Non-null once the counterfactual layer (M8) enumerates alternatives and the
// support gate (M7) decides which are scored. Until then, a candidate that
// was enumerated but gated appears with value: null, scored: false.
export interface Option {
  type_name: string;
  end_x: number;
  end_y: number;
  value: number | null;
  scored: boolean;
}

export interface Moment {
  moment_id: string;
  match_id: string;
  minute: number;
  second: number;
  team: string;
  player_name: string;
  ball_x: number;
  ball_y: number;
  chosen: ScoredAction;
  // null until M8: the counterfactual layer is not built in this pass.
  best: ScoredAction | null;
  regret: number | null;
  options: Option[];
  unscored_count: number;
}

export interface MatchSummary {
  match_id: string;
  competition_name: string;
  home_team: string;
  away_team: string;
  match_date: string;
  action_count: number;
  mean_value: number;
}

export interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  mean_observed: number;
  count: number;
}

export interface CalibrationReport {
  split: string;
  brier: number;
  reliability: number;
  resolution: number;
  uncertainty: number;
  ece: number;
  curve: CalibrationBin[];
  n: number;
}

export interface HealthResponse {
  status: string;
  model_version: string | null;
  run_id: string | null;
}

export interface TransferMetrics {
  competitions: string[];
  n: number;
  brier: number;
  reliability: number;
  resolution: number;
  uncertainty: number;
  ece: number;
  calibration_method: string | null;
  n_recalibration_rows: number | null;
}

export interface TransferLabelReport {
  source: TransferMetrics;
  target_before: TransferMetrics;
  target_after: TransferMetrics;
  ece_degradation_ratio: number | null;
  ece_recovered_fraction: number | null;
}

export interface TransferReport {
  config_name: string;
  run_id: string;
  label_scores: TransferLabelReport;
  label_concedes: TransferLabelReport;
}

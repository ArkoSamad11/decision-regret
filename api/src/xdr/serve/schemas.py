"""Response models (SPEC.md §12). Mirrored by hand in web/lib/types.ts --
changing one without the other is a defect the type checker cannot catch.
"""

from __future__ import annotations

from pydantic import BaseModel


class ScoredAction(BaseModel):
    type_name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    p_scores: float
    p_concedes: float
    value: float


class Option(BaseModel):
    """The counterfactual layer (M8) is not built in this pass -- every moment
    currently has an empty `options` list. The shape is fixed now so the
    frontend contract does not change when M8 lands."""

    type_name: str
    end_x: float
    end_y: float
    value: float | None
    scored: bool


class Moment(BaseModel):
    moment_id: str
    match_id: str
    minute: int
    second: int
    team: str
    player_name: str
    ball_x: float
    ball_y: float
    chosen: ScoredAction
    best: ScoredAction | None = None
    regret: float | None = None
    options: list[Option] = []
    unscored_count: int = 0


class MatchSummary(BaseModel):
    match_id: str
    competition_name: str
    home_team: str
    away_team: str
    match_date: str
    action_count: int
    mean_value: float


class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int


class CalibrationReport(BaseModel):
    split: str
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    curve: list[CalibrationBin]
    n: int


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    run_id: str | None


class TransferMetrics(BaseModel):
    """SPEC.md §11.2 item 2: source vs target vs recalibrated, per label."""

    competitions: list[str]
    n: int
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    ece: float
    calibration_method: str | None = None
    n_recalibration_rows: int | None = None


class TransferLabelReport(BaseModel):
    source: TransferMetrics
    target_before: TransferMetrics
    target_after: TransferMetrics
    ece_degradation_ratio: float | None
    ece_recovered_fraction: float | None


class TransferReport(BaseModel):
    config_name: str
    run_id: str
    label_scores: TransferLabelReport
    label_concedes: TransferLabelReport

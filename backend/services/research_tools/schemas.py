from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ResearchToolBaseRequest(BaseModel):
    expression: str
    universe: str = "hs300"
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    n_groups: int = 5
    holding_period: int = 5
    benchmark: str = "hs300"
    neutralize_industry: bool = True
    neutralize_cap: bool = True


class ValidationRequest(BaseModel):
    expression: str
    mode: str = "local"


class ValidationResponse(BaseModel):
    success: bool
    valid: bool
    mode: str
    message: str
    raw: dict[str, Any] | None = None


class ScoreResponse(BaseModel):
    success: bool
    score: float | None = None
    grade: str | None = None
    summary: str | None = None
    component_scores: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class DiagnosisResponse(BaseModel):
    success: bool
    report: str | None = None
    key_findings: list[str] | None = None
    improvement_suggestions: list[str] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class RobustnessResponse(BaseModel):
    success: bool
    summary: dict[str, Any] | None = None
    details: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class WQBrainAlphaIdsRequest(BaseModel):
    alpha_ids: list[str]
    account: str = "primary"


class WQBrainCandidateSyncRequest(BaseModel):
    factor_ids: list[int]
    account: str = "primary"


class WQBrainSubmitRequest(BaseModel):
    alpha_id: str | None = None
    factor_id: int | None = None
    expression: str | None = None
    account: str = "primary"
    auto_submit: bool = True
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08


class WQBrainBatchSubmitRequest(BaseModel):
    alpha_ids: list[str]
    account: str = "primary"


class WQBrainConfigResponse(BaseModel):
    success: bool
    default_account: str = "primary"
    primary_email: str = ""
    alt_email: str = ""
    has_primary_password: bool = False
    has_alt_password: bool = False


class WQBrainConfigUpdateRequest(BaseModel):
    default_account: str = "primary"
    primary_email: str = ""
    primary_password: str | None = None
    alt_email: str = ""
    alt_password: str | None = None

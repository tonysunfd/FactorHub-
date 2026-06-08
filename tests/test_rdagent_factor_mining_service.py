from __future__ import annotations

from types import SimpleNamespace

from backend.services.expression_schema import FactorEvaluationResult
from backend.services.rdagent_factor_mining_service import (
    RDAgentFactorMiningService,
    RDAgentMiningConfig,
)


def _build_evaluation(expression: str, score: float, rank_ic: float, annual_return: float) -> FactorEvaluationResult:
    return FactorEvaluationResult(
        expression=expression,
        raw_expression=expression,
        engine_type="quantgpt",
        dialect="factorhub_native",
        canonical_expression=expression,
        canonical_ast=None,
        score=score,
        grade="A" if score >= 80 else "B",
        report_metrics={"sharpe": 1.2, "max_drawdown": 0.03, "rank_ic": rank_ic},
        backtest_summary={
            "rank_ic_mean": rank_ic,
            "ic_mean": rank_ic,
            "ic_ir": 0.8,
            "long_short_sharpe": 1.1,
            "long_short_annual": annual_return,
            "turnover": 0.2,
            "wq_fitness": 1.8,
        },
        wq_brain={"wq_rating": "A", "wq_fitness": 1.8, "wq_returns": annual_return},
        component_scores={"total_score": score},
        anti_overfit={"passed": True},
        interpretation={"weaknesses": [], "next_steps": ["继续优化"]},
        diagnostics=[],
        report_url="/api/mining/reports/test-report.html",
        execution_meta={},
    )


class _FakeAutoMiningService:
    def __init__(self) -> None:
        self.select_factors_calls: list[dict] = []
        self.evaluate_expression_calls: list[dict] = []
        self.data_service = SimpleNamespace(
            get_stock_universe=lambda universe, date=None: ["000001.SZ", "000002.SZ", "000004.SZ"]
        )

    def select_factors(self, **kwargs):
        self.select_factors_calls.append(kwargs)
        return {"selected_factors": ["AlphaVolume", "AlphaTrend"]}

    def evaluate_expression(self, **kwargs):
        self.evaluate_expression_calls.append(kwargs)
        expression = kwargs["expression"]
        if "volume" in expression:
            return _build_evaluation(expression, 86.0, 0.08, 0.18)
        return _build_evaluation(expression, 72.0, 0.05, 0.11)


class _FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            content = (
                '{"statement":"量价共振因子可提升综合分数","reason":"上一轮稳定性一般，需要增强量价共振。",'
                '"research_direction":"score","expected_signal":"提升 Score 与 rankIC"}'
            )
        else:
            content = (
                '{"factor_formulations":['
                '"rank(ts_delta(close, 5))",'
                '"rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))"'
                ']}'
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_rdagent_service_runs_independent_executor_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "base_url": "https://example.com/v1", "model": "deepseek-chat"},
    )
    fake_auto_service = _FakeAutoMiningService()
    fake_llm_client = _FakeLLMClient()
    service = RDAgentFactorMiningService(
        auto_mining_service=fake_auto_service,
        llm_client_factory=lambda runtime_config: fake_llm_client,
    )

    progress_events: list[tuple[int, str, dict]] = []
    config = RDAgentMiningConfig(
        task_id="rdagent-test",
        objective="提升综合分数",
        max_iterations=1,
        candidates_per_iteration=2,
        base_factors=[],
        candidate_universe=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe="hs300",
        benchmark="000300.SH",
        acceptance_policy={
            "max_correlation_with_sota": 0.99,
            "min_rank_ic": 0.03,
            "min_annualized_return_delta": 0.0,
            "max_drawdown_regression": 0.05,
            "min_valid_coverage": 0.8,
        },
    )

    result = service.run(
        task_id="rdagent-test",
        config=config,
        on_progress=lambda progress, stage, event: progress_events.append((progress, stage, event)),
    )

    assert fake_auto_service.select_factors_calls, "应复用自动挖掘的因子筛选能力"
    assert len(fake_auto_service.evaluate_expression_calls) == 2
    assert result["rounds"][0]["hypothesis"]["statement"] == "量价共振因子可提升综合分数"
    assert result["rounds"][0]["experiment"]["factor_formulations"][0] == "rank(ts_delta(close, 5))"
    assert result["retained_factors"][0]["status"] == "accepted"
    assert result["continue_mining_request"]["payload"]["continuation_of"] == "rdagent-test"
    assert any(stage == "rdagent_feedback" for _, stage, _ in progress_events)


def test_rdagent_service_falls_back_when_llm_output_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "base_url": "https://example.com/v1", "model": "deepseek-chat"},
    )

    class _BadLLMClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"factor_formulations":["bad func(close)"]}'))]
            )

    fake_auto_service = _FakeAutoMiningService()
    service = RDAgentFactorMiningService(
        auto_mining_service=fake_auto_service,
        llm_client_factory=lambda runtime_config: _BadLLMClient(),
    )

    result = service.run(
        task_id="rdagent-test",
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaVolume"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
        ),
    )

    assert result["rounds"][0]["experiment"]["factor_formulations"], "应回退到内置表达式模板"

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
        self.select_continue_factors_calls: list[dict] = []
        self.evaluate_expression_calls: list[dict] = []
        self.data_service = SimpleNamespace(
            get_stock_universe=lambda universe, date=None: ["000001.SZ", "000002.SZ", "000004.SZ"]
        )

    def select_factors(self, **kwargs):
        self.select_factors_calls.append(kwargs)
        return {"selected_factors": ["AlphaVolume", "AlphaTrend"]}

    def select_continue_factors(self, **kwargs):
        self.select_continue_factors_calls.append(kwargs)
        return {
            "selected_factors": ["AlphaContinue"],
            "continuation_context": {
                "primary_problem": "上一轮区分度不足",
                "recommended_goal": "ls_sharpe",
            },
        }

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


class _RecordingLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        messages = kwargs.get("messages") or []
        user_prompt = messages[-1]["content"] if messages else ""
        self.prompts.append(user_prompt)
        if self.calls % 2 == 1:
            content = (
                '{"statement":"继续优化量价共振","reason":"延续上一轮有效方向。",'
                '"research_direction":"score","expected_signal":"提升 Score"}'
            )
        else:
            content = (
                '{"factor_formulations":['
                '"rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))",'
                '"rank(ts_delta(close, 5))"'
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
    assert len(result["retained_factors"]) == 2
    assert [factor["score"] for factor in result["top_factors"]] == [86.0, 72.0]
    assert result["continue_mining_request"]["payload"]["continuation_of"] == "rdagent-test"
    assert any(stage == "rdagent_feedback" for _, stage, _ in progress_events)


def test_rdagent_continue_request_avoids_reusing_single_single_budget() -> None:
    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())

    continue_request = service._build_continue_request(
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
        final_round={
            "round_index": 1,
            "feedback": {
                "next_hypothesis": "继续围绕 ls_sharpe 优化表达式稳定性与可执行性。",
                "reason": "上一轮预算过小，需要扩大下一轮探索范围。",
            },
            "experiment": {
                "base_factors": ["AlphaVolume"],
            },
        },
        known_expressions=["rank(ts_delta(close, 5))"],
    )

    payload = continue_request["payload"]
    assert payload["max_iterations"] == 2
    assert payload["candidates_per_iteration"] == 2
    assert payload["previous_sota_expressions"] == []


def test_rdagent_service_uses_continue_factor_selection_for_next_round(monkeypatch) -> None:
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

    result = service.run(
        task_id="rdagent-test",
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=2,
            candidates_per_iteration=2,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
        ),
    )

    assert fake_auto_service.select_continue_factors_calls, "应在多轮模式下为下一轮选择基础因子"
    assert result["rounds"][0]["next_base_factors"] == ["AlphaSeed", "AlphaContinue"]
    assert result["rounds"][0]["sota_candidates"], "首轮应沉淀 SOTA 候选，供下一轮复用"


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


def test_rdagent_prompts_include_runtime_constraints() -> None:
    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())
    config = RDAgentMiningConfig(
        task_id="rdagent-test",
        objective="提升收益并控制回撤",
        max_iterations=4,
        candidates_per_iteration=2,
        base_factors=["AlphaVolume"],
        candidate_universe=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe="hs300",
        benchmark="000300.SH",
        n_groups=10,
        holding_period=7,
        direction="report_sharpe",
        neutralize_industry=False,
        neutralize_cap=True,
        continuation_of="parent-task",
        previous_feedback_id="feedback-1",
        acceptance_policy={
            "max_correlation_with_sota": 0.35,
            "min_rank_ic": 0.06,
            "min_annualized_return_delta": 0.12,
            "max_drawdown_regression": 0.03,
            "min_valid_coverage": 0.9,
        },
    )

    hypothesis_prompt = service._build_hypothesis_prompt(
        config=config,
        rounds=[{"feedback": {"observations": "上一轮回撤偏大"}, "evaluation": {"best_score": 81.0}}],
        iteration=2,
        current_base_factors=["AlphaVolume"],
    )
    experiment_prompt = service._build_experiment_prompt(
        config=config,
        hypothesis={"statement": "增强量价协同并抑制回撤"},
        rounds=[],
        iteration=2,
        current_base_factors=["AlphaVolume"],
        known_expressions=["rank(ts_delta(close, 5))"],
        sota_expressions=["rank(ts_mean(volume, 10))"],
    )

    for prompt in (hypothesis_prompt, experiment_prompt):
        assert "总轮数预算：4" in prompt
        assert "回测区间：2024-01-01 至 2024-03-31" in prompt
        assert "股票池：hs300" in prompt
        assert "基准：000300.SH" in prompt
        assert '"min_rank_ic": 0.06' in prompt
        assert '"max_correlation_with_sota": 0.35' in prompt

    assert "候选数量预算：本轮最多生成 2 条候选表达式" in hypothesis_prompt
    assert "需要生成 2 条候选表达式" in experiment_prompt
    assert '当前 SOTA 候选表达式：["rank(ts_mean(volume, 10))"]' in experiment_prompt


def test_rdagent_second_round_prompt_includes_previous_sota_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "base_url": "https://example.com/v1", "model": "deepseek-chat"},
    )

    fake_auto_service = _FakeAutoMiningService()
    recording_llm = _RecordingLLMClient()
    service = RDAgentFactorMiningService(
        auto_mining_service=fake_auto_service,
        llm_client_factory=lambda runtime_config: recording_llm,
    )

    result = service.run(
        task_id="rdagent-test",
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=2,
            candidates_per_iteration=2,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
        ),
    )

    second_round_experiment_prompt = recording_llm.prompts[3]
    first_round_sota = result["rounds"][0]["sota_candidates"]
    assert first_round_sota
    assert "当前 SOTA 候选表达式" in second_round_experiment_prompt
    assert first_round_sota[0]["expression"] in second_round_experiment_prompt


def test_rdagent_continue_request_persists_sota_library() -> None:
    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())

    continue_request = service._build_continue_request(
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=2,
            candidates_per_iteration=2,
            base_factors=["AlphaVolume"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
        ),
        final_round={
            "round_index": 2,
            "feedback": {
                "next_hypothesis": "继续强化优胜表达式附近的搜索。",
            },
            "experiment": {
                "base_factors": ["AlphaVolume"],
            },
            "sota_candidates": [
                {"expression": "rank(ts_mean(volume, 10))"},
                {"expression": "rank(ts_delta(close, 5))"},
            ],
        },
        known_expressions=["rank(ts_delta(close, 5))"],
    )

    assert continue_request["payload"]["previous_sota_expressions"] == [
        "rank(ts_mean(volume, 10))",
        "rank(ts_delta(close, 5))",
    ]


def test_rdagent_service_bootstraps_sota_library_from_previous_request() -> None:
    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())

    result = service.run(
        task_id="rdagent-test",
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            previous_sota_expressions=["rank(ts_mean(close, 5))"],
            acceptance_policy={
                "max_correlation_with_sota": 0.2,
            },
        ),
    )

    sota_candidates = result["sota_candidates"]
    assert any(item["expression"] == "rank(ts_mean(close, 5))" for item in sota_candidates)


def test_rdagent_collects_global_top_factors_across_rounds(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "base_url": "https://example.com/v1", "model": "deepseek-chat"},
    )

    class _MultiRoundAutoMiningService(_FakeAutoMiningService):
        def evaluate_expression(self, **kwargs):
            self.evaluate_expression_calls.append(kwargs)
            expression = kwargs["expression"]
            round_index = len(self.evaluate_expression_calls)
            if "volume" in expression and round_index <= 2:
                return _build_evaluation(expression, 91.0, 0.08, 0.20)
            if "close" in expression and round_index <= 2:
                return _build_evaluation(expression, 85.0, 0.07, 0.16)
            if "volume" in expression:
                return _build_evaluation(expression, 73.0, 0.04, 0.10)
            return _build_evaluation(expression, 69.0, 0.03, 0.08)

    service = RDAgentFactorMiningService(
        auto_mining_service=_MultiRoundAutoMiningService(),
        llm_client_factory=lambda runtime_config: _FakeLLMClient(),
    )

    result = service.run(
        task_id="rdagent-test",
        config=RDAgentMiningConfig(
            task_id="rdagent-test",
            objective="提升综合分数",
            max_iterations=2,
            candidates_per_iteration=2,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
        ),
    )

    top_scores = [factor["score"] for factor in result["top_factors"]]
    assert top_scores == sorted(top_scores, reverse=True)
    assert len(result["top_factors"]) <= 5
    assert result["final_round_result"]["factors"] == result["top_factors"]

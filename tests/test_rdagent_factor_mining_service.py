from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd

from backend.services.expression_schema import FactorEvaluationResult
from backend.services.rdagent_factor_mining_service import (
    RDAgentFactorMiningService,
    RDAgentMiningConfig,
)


def _build_evaluation(
    expression: str,
    score: float,
    rank_ic: float,
    annual_return: float,
    *,
    factor_snapshot: list[dict[str, object]] | None = None,
) -> FactorEvaluationResult:
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
        execution_meta={"factor_snapshot": factor_snapshot or []},
        factor_series=pd.Series(dtype=float),
    )


class _FakeAutoMiningService:
    def __init__(self) -> None:
        self.select_factors_calls: list[dict] = []
        self.select_continue_factors_calls: list[dict] = []
        self.evaluate_expression_calls: list[dict] = []
        self._load_benchmark_returns = lambda benchmark, start_date, end_date: pd.Series(dtype=float)
        self._write_candidate_report = lambda strategy_returns, benchmark_returns, periods_per_year: (
            {"sharpe": 1.2, "max_drawdown": 0.03},
            "/api/mining/reports/test-report.html",
        )
        self.data_service = SimpleNamespace(
            get_stock_universe=lambda universe, date=None: [
                "000001.SZ",
                "000002.SZ",
                "000004.SZ",
                "000005.SZ",
                "000006.SZ",
                "000007.SZ",
                "000008.SZ",
                "000009.SZ",
                "000010.SZ",
                "000011.SZ",
                "000012.SZ",
                "000014.SZ",
            ],
            get_stock_data=lambda stock_code, start_date, end_date: _build_mock_stock_frame(stock_code),
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


def _build_mock_stock_frame(stock_code: str) -> pd.DataFrame:
    stock_offset = (sum(ord(ch) for ch in stock_code) % 7) * 0.03
    rows = []
    for idx, date in enumerate(pd.date_range("2024-01-01", periods=60, freq="D")):
        seasonal = math.sin(idx / 4 + stock_offset)
        momentum = idx * 0.05
        close = 10 + momentum + seasonal * 0.8 + stock_offset
        open_price = close - 0.15 + math.cos(idx / 5 + stock_offset) * 0.05
        rows.append(
            {
                "date": date,
                "open": open_price,
                "high": close + 0.25,
                "low": close - 0.3,
                "close": close,
                "volume": 1000 + idx * 12 + (sum(ord(ch) for ch in stock_code) % 10) * 7 + seasonal * 60,
                "amount": close * (1000 + idx * 12),
                "pct_change": 0.01 * math.sin(idx / 6 + stock_offset),
            }
        )
    return pd.DataFrame(rows).set_index("date")


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
        execution_mode="expression",
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
    assert result["rounds"][0]["pipeline"] == {
        "execution_mode": "expression",
        "proposal": "factorhub_local_hypothesis",
        "coder": "FactorHubRDAgentCoder",
        "runner": "FactorHubRDAgentRunner",
        "feedback": "FactorHubRDAgentFeedback",
        "data_source": "factorhub_v3_local_data_source",
        "evaluation_system": "factorhub_v3_local_evaluation",
    }
    assert result["rounds"][0]["coded_experiment"]["developer_name"] == "FactorHubRDAgentCoder"
    assert result["rounds"][0]["coded_experiment"]["developer_stage"] == "coding"
    assert result["rounds"][0]["feedback"]["developer_name"] == "FactorHubRDAgentFeedback"
    assert result["rounds"][0]["feedback"]["developer_stage"] == "feedback"
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
            execution_mode="expression",
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
            execution_mode="expression",
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
            execution_mode="expression",
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
                execution_mode="expression",
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
            execution_mode="expression",
        ),
    )

    top_scores = [factor["score"] for factor in result["top_factors"]]
    assert top_scores == sorted(top_scores, reverse=True)
    assert len(result["top_factors"]) <= 5
    assert result["final_round_result"]["factors"] == result["top_factors"]


def test_rdagent_estimates_sota_correlation_from_factor_values() -> None:
    candidate = {
        "expression": "rank(ts_delta(close, 5))",
        "execution_meta": {
            "factor_snapshot": [
                {"date": "2024-01-02", "stock_code": "AAA", "factor": 1.0},
                {"date": "2024-01-02", "stock_code": "BBB", "factor": 2.0},
                {"date": "2024-01-03", "stock_code": "AAA", "factor": 3.0},
                {"date": "2024-01-03", "stock_code": "BBB", "factor": 4.0},
            ]
        },
    }
    sota_candidates = [
        {
            "expression": "completely_different_expression(volume)",
            "factor_snapshot": [
                {"date": "2024-01-02", "stock_code": "AAA", "factor": 10.0},
                {"date": "2024-01-02", "stock_code": "BBB", "factor": 20.0},
                {"date": "2024-01-03", "stock_code": "AAA", "factor": 30.0},
                {"date": "2024-01-03", "stock_code": "BBB", "factor": 40.0},
            ],
        }
    ]

    correlation = RDAgentFactorMiningService._estimate_sota_correlation(candidate, sota_candidates)

    assert correlation == 1.0


def test_rdagent_acceptance_policy_uses_real_sota_correlation() -> None:
    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())
    aligned_snapshot = [
        {"date": "2024-01-02", "stock_code": "AAA", "factor": 1.0},
        {"date": "2024-01-02", "stock_code": "BBB", "factor": 2.0},
        {"date": "2024-01-03", "stock_code": "AAA", "factor": 3.0},
        {"date": "2024-01-03", "stock_code": "BBB", "factor": 4.0},
    ]
    candidate = service._format_candidate_payload(
        evaluation=_build_evaluation(
            "rank(ts_delta(close, 5))",
            86.0,
            0.08,
            0.18,
            factor_snapshot=aligned_snapshot,
        ),
        coded_item={"candidate_id": "candidate-1", "raw_expression": "rank(ts_delta(close, 5))"},
        hypothesis={"statement": "测试真实相关性"},
        iteration=1,
        index=0,
        acceptance_policy={"max_correlation_with_sota": 0.5},
    )
    policy = {
        "max_correlation_with_sota": 0.5,
        "_sota_candidates": [
            {
                "expression": "totally_different_formula(volume)",
                "factor_snapshot": aligned_snapshot,
            }
        ],
    }

    service._apply_acceptance_policy(candidate, policy, 0)

    assert candidate["status"] == "watchlist"
    assert candidate["task_details"]["rdagent"]["candidate_score"]["max_correlation_with_sota"] == 1.0
    assert any("max_correlation_with_sota 1.0000 高于阈值 0.5000" in reason for reason in candidate["policy_diagnostics"]["failure_reasons"])


def test_rdagent_result_payload_strips_runtime_factor_frame() -> None:
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
            acceptance_policy={},
            execution_mode="expression",
        ),
    )

    assert "_factor_frame" not in result["top_factors"][0]
    assert "factor_snapshot" not in result["sota_candidates"][0]


def test_rdagent_native_code_mode_uses_factorhub_data_and_local_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "base_url": "https://example.com/v1", "model": "deepseek-chat"},
    )

    class _NativeCodeLLMClient:
        def __init__(self) -> None:
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = (
                    '{"statement":"价格与成交量联动可提升综合分数","reason":"先验证简单价量组合。",'
                    '"research_direction":"score","expected_signal":"提升 Score"}'
                )
            elif self.calls == 2:
                content = (
                    '{"factor_formulations":["placeholder_expression"]}'
                )
            else:
                content = (
                    '{"factor_name":"PriceVolumeSignal",'
                    '"implementation_code":"def calculate_factor(df):\\n'
                    '    close = pd.to_numeric(df[\\\"close\\\"], errors=\\\"coerce\\\")\\n'
                    '    volume = pd.to_numeric(df[\\\"volume\\\"], errors=\\\"coerce\\\")\\n'
                    '    signal = close.pct_change(3).rolling(5, min_periods=1).mean() - volume.pct_change(5).rolling(5, min_periods=1).mean()\\n'
                    '    return pd.Series(signal, index=df.index, dtype=float)",'
                    '"implementation_notes":"simple"}'
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    native_llm_client = _NativeCodeLLMClient()
    service = RDAgentFactorMiningService(
        auto_mining_service=_FakeAutoMiningService(),
        llm_client_factory=lambda runtime_config: native_llm_client,
    )

    result = service.run(
        task_id="rdagent-native",
        config=RDAgentMiningConfig(
            task_id="rdagent-native",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="native_code",
        ),
    )

    factor = result["top_factors"][0]
    assert factor["engine_type"] == "rdagent_native_code"
    assert factor["dialect"] == "python_factor_function"
    assert "implementation_code" in (factor.get("execution_meta") or {})


def test_rdagent_upstream_mode_reuses_reference_proposal_and_local_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": False,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/miniconda3/bin/python3.13",
            "checked_paths": [],
            "importable": False,
            "import_error": "fire missing",
        },
    )

    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "upstream 认为量价共振值得继续验证",
                "reason": "先从简单方向切入。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "UpstreamVolumeFactor",
                    "description": "[Momentum Factor] volume stabilized momentum",
                    "formulation": "rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))",
                    "variables": {"volume": "trading volume"},
                }
            ],
        },
    )

    result = service.run(
        task_id="rdagent-upstream",
        config=RDAgentMiningConfig(
            task_id="rdagent-upstream",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="upstream_rdagent",
        ),
    )

    factor = result["top_factors"][0]
    assert factor["expression"] == "rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))"
    assert factor["engine_type"] == "quantgpt"
    assert factor["task_details"]["rdagent"]["candidate_score"]["score"] >= 80


def test_rdagent_upstream_mode_converts_latex_formulation_before_local_evaluation(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": True,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/Desktop/Factorhub V3/.venv-rdagent/bin/python",
            "checked_paths": [],
        },
    )

    fake_auto_service = _FakeAutoMiningService()
    service = RDAgentFactorMiningService(auto_mining_service=fake_auto_service)
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "短期价格动量值得验证",
                "reason": "先验证简单动量方向。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "ShortTermMomentum",
                    "description": "5 日价格动量",
                    "formulation": r"F_t = \frac{C_t}{C_{t-5}} - 1",
                    "variables": {"C_t": "close price at time t"},
                }
            ],
        },
    )

    result = service.run(
        task_id="rdagent-upstream-latex",
        config=RDAgentMiningConfig(
            task_id="rdagent-upstream-latex",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="upstream_rdagent",
        ),
    )

    assert fake_auto_service.evaluate_expression_calls
    assert fake_auto_service.evaluate_expression_calls[0]["expression"] == "((close) / (ts_shift(close,5))) - 1"
    factor = result["top_factors"][0]
    assert factor["expression"] == "((close) / (ts_shift(close,5))) - 1"
    assert factor["engine_type"] == "quantgpt"


def test_rdagent_upstream_mode_can_fallback_to_native_code_conversion(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": True,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/Desktop/Factorhub V3/.venv-rdagent/bin/python",
            "checked_paths": [],
        },
    )

    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "复杂公式需要函数实现",
                "reason": "先验证函数执行链路。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "FunctionFactor",
                    "description": "通过函数返回滚动信号",
                    "formulation": "non_executable_formula(close, volume)",
                    "variables": {"close": "close", "volume": "volume"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        service,
        "_convert_formulation_to_expression_with_llm",
        lambda formulation, variables: (
            'def calculate_factor(df):\n'
            '    close = pd.to_numeric(df["close"], errors="coerce")\n'
            '    return close.pct_change(3).rolling(5, min_periods=1).mean()'
        ),
    )

    result = service.run(
        task_id="rdagent-upstream-function",
        config=RDAgentMiningConfig(
            task_id="rdagent-upstream-function",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="upstream_rdagent",
        ),
    )

    factor = result["top_factors"][0]
    assert factor["engine_type"] == "rdagent_upstream_native_code"
    assert factor["dialect"] == "python_factor_function"
    assert "implementation_code" in (factor.get("execution_meta") or {})


def test_rdagent_upstream_mode_reports_skip_reason_when_all_candidates_are_unevaluable(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": True,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/Desktop/Factorhub V3/.venv-rdagent/bin/python",
            "checked_paths": [],
        },
    )

    service = RDAgentFactorMiningService(auto_mining_service=_FakeAutoMiningService())
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "复杂公式待验证",
                "reason": "先看是否能转换。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "BadFormula",
                    "description": "无法直接执行的公式",
                    "formulation": "totally_unknown_formula(alpha, beta)",
                    "variables": {"alpha": "alpha", "beta": "beta"},
                }
            ],
        },
    )
    monkeypatch.setattr(
        service,
        "_convert_formulation_to_expression_with_llm",
        lambda formulation, variables: None,
    )

    try:
        service.run(
            task_id="rdagent-upstream-bad",
            config=RDAgentMiningConfig(
                task_id="rdagent-upstream-bad",
                objective="提升综合分数",
                max_iterations=1,
                candidates_per_iteration=1,
                base_factors=["AlphaSeed"],
                candidate_universe=["close", "volume"],
                start_date="2024-01-01",
                end_date="2024-03-31",
                universe="hs300",
                benchmark="000300.SH",
                acceptance_policy={},
                execution_mode="upstream_rdagent",
            ),
        )
    except ValueError as exc:
        message = str(exc)
        assert "execution_mode=upstream_rdagent" in message
        assert "upstream formulation 未能转换为 FactorHub 可执行表达式或 Python 因子函数" in message
        assert "BadFormula" in message
    else:
        raise AssertionError("预期应抛出带跳过原因的 ValueError")


def test_rdagent_upstream_mode_converts_reference_style_momentum_and_sum_formulations(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": True,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/Desktop/Factorhub V3/.venv-rdagent/bin/python",
            "checked_paths": [],
        },
    )

    fake_auto_service = _FakeAutoMiningService()
    service = RDAgentFactorMiningService(auto_mining_service=fake_auto_service)
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "动量和量能值得验证",
                "reason": "使用 reference 风格 formulation。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "Momentum5",
                    "description": "5 日动量",
                    "formulation": r"MOM_{5}(t)=\frac{close_t}{close_{t-5}}-1",
                    "variables": {"close": "close"},
                },
                {
                    "factor_name": "RelativeVolume10",
                    "description": "10 日相对成交量",
                    "formulation": r"RVOL_{10}(t)=\frac{volume_t}{\frac{1}{10}\sum_{i=1}^{10} volume_{t-i}}",
                    "variables": {"volume": "volume"},
                },
            ],
        },
    )

    result = service.run(
        task_id="rdagent-upstream-reference-style",
        config=RDAgentMiningConfig(
            task_id="rdagent-upstream-reference-style",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=2,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="upstream_rdagent",
        ),
    )

    expressions = [call["expression"] for call in fake_auto_service.evaluate_expression_calls]
    assert "((close) / (ts_shift(close,5)))-1" in expressions
    assert "((volume) / (((1) / (10)) * ts_sum(volume,10)))" in expressions
    assert len(result["top_factors"]) == 2


def test_rdagent_upstream_mode_converts_reference_style_log_formulations(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.probe_rdagent_module_import",
        lambda module_name: (True, None),
    )
    monkeypatch.setattr(
        "backend.services.rdagent_factor_mining_service.get_rdagent_runtime_status",
        lambda: {
            "available": True,
            "active_path": "/Users/tonysun/Desktop/reference/RD-Agent",
            "python_path": "/Users/tonysun/Desktop/Factorhub V3/.venv-rdagent/bin/python",
            "checked_paths": [],
        },
    )

    fake_auto_service = _FakeAutoMiningService()
    service = RDAgentFactorMiningService(auto_mining_service=fake_auto_service)
    monkeypatch.setattr(
        service,
        "_generate_upstream_round_plan",
        lambda **kwargs: {
            "hypothesis": {
                "statement": "log 形式的动量值得验证",
                "reason": "使用 reference 风格 ln formulation。",
                "concise_observation": "提升 Score",
            },
            "tasks": [
                {
                    "factor_name": "LogMomentumSpread",
                    "description": "长短窗口 log 动量差",
                    "formulation": r"F_t = \ln\left(\frac{close_t}{close_{t-5}}\right) - \ln\left(\frac{close_t}{close_{t-20}}\right)",
                    "variables": {"close": "close"},
                }
            ],
        },
    )

    result = service.run(
        task_id="rdagent-upstream-log-style",
        config=RDAgentMiningConfig(
            task_id="rdagent-upstream-log-style",
            objective="提升综合分数",
            max_iterations=1,
            candidates_per_iteration=1,
            base_factors=["AlphaSeed"],
            candidate_universe=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-03-31",
            universe="hs300",
            benchmark="000300.SH",
            acceptance_policy={},
            execution_mode="upstream_rdagent",
        ),
    )

    expressions = [call["expression"] for call in fake_auto_service.evaluate_expression_calls]
    assert "log(((close) / (ts_shift(close,5)))) - log(((close) / (ts_shift(close,20))))" in expressions
    assert len(result["top_factors"]) == 1

from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from backend.services.auto_factor_mining_service import AutoFactorMiningService
from backend.services.factor_service import factor_service


def test_build_continuation_context_uses_round_evaluation() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "Sharpe 偏低",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["补充波动率因子"],
        },
        "factors": [
            {
                "score": 72.5,
                "report_metrics": {"sharpe": 0.42, "max_drawdown": 0.28},
                "backtest_summary": {
                    "long_short_sharpe": 0.42,
                    "long_short_annual": 0.08,
                    "rank_ic_mean": 0.019,
                    "turnover": 0.61,
                    "wq_fitness": 0.78,
                },
                "interpretation": {
                    "weaknesses": ["横截面区分度不足"],
                    "next_steps": ["降低换手率"],
                },
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Fallback"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=4,
    )

    assert context["base_factors"] == ["Alpha1", "Alpha2"]
    assert context["primary_problem"] == "Sharpe 偏低"
    assert context["recommended_goal"] == "ls_sharpe"
    assert "补充波动率因子" in context["suggested_actions"]


def test_filter_retained_factors_supports_match_modes() -> None:
    service = AutoFactorMiningService()
    factors = [
        {
            "score": 82,
            "backtest_summary": {"long_short_sharpe": 1.1, "long_short_annual": 0.18},
            "wq_brain": {"wq_rating": "A", "wq_returns": 0.18},
        },
        {
            "score": 58,
            "backtest_summary": {"long_short_sharpe": 0.4, "long_short_annual": 0.03},
            "wq_brain": {"wq_rating": "C", "wq_returns": 0.03},
        },
    ]

    retained_all = service.filter_retained_factors(
        factors,
        {"match_mode": "all", "score_min": 60, "wq_ratings": ["A", "B"], "ls_sharpe_min": 0.8},
    )
    retained_any = service.filter_retained_factors(
        factors,
        {"match_mode": "any", "score_min": 80, "wq_ratings": ["C"]},
    )

    assert retained_all == [factors[0]]
    assert retained_any == factors


def test_run_auto_campaign_returns_real_rounds(monkeypatch) -> None:
    service = AutoFactorMiningService()
    continuation_calls: list[dict[str, object]] = []

    def fake_run_auto_mining(**kwargs):
        base_factors = kwargs["base_factors"]
        round_index = 1 if "ExtraFactor" not in base_factors else 2
        best_score = 60 + round_index * 10
        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
                    "score": best_score,
                    "grade": "B" if round_index == 1 else "A",
                    "report_metrics": {"sharpe": 0.5 + round_index * 0.2, "cagr": 0.05 + round_index * 0.03},
                    "backtest_summary": {
                        "long_short_sharpe": 0.5 + round_index * 0.2,
                        "long_short_annual": 0.05 + round_index * 0.03,
                        "rank_ic_mean": 0.02 + round_index * 0.01,
                        "ic_ir": 0.6 + round_index * 0.1,
                        "turnover": 0.3,
                        "wq_fitness": 0.8 + round_index * 0.1,
                    },
                    "wq_brain": {
                        "wq_rating": "B" if round_index == 1 else "A",
                        "wq_returns": 0.05 + round_index * 0.03,
                        "wq_fitness": 0.8 + round_index * 0.1,
                    },
                    "interpretation": {
                        "weaknesses": [f"第 {round_index} 轮短板"],
                        "next_steps": [f"第 {round_index} 轮建议"],
                    },
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": list(base_factors),
                            "primary_problem": f"第 {round_index} 轮问题",
                            "recommended_goal": "score",
                            "suggested_actions": [f"第 {round_index} 轮动作"],
                            "metric_snapshot": {"score": best_score},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score - 5,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score - 5]},
            "round_evaluation": {
                "base_factors": list(base_factors),
                "primary_problem": f"第 {round_index} 轮问题",
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": best_score},
            },
        }

    def fake_select_continue_factors(**kwargs):
        continuation_calls.append(kwargs)
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补充一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "沿着第 1 轮问题继续优化",
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    snapshots = []
    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
        progress_callback=snapshots.append,
    )

    assert len(result["rounds"]) == 2
    assert result["rounds"][1]["input_base_factors"] == ["Alpha1", "ExtraFactor"]
    assert result["best_score"] == 80
    assert result["retained_factors"][0]["expression"] == "expr_2"
    assert snapshots[-1]["current_round"] == 2
    assert continuation_calls[0]["parent_result"]["round_evaluation"]["primary_problem"] == "第 1 轮问题"
    assert result["rounds"][1]["continuation_hypothesis"]["reason"] == "第 1 轮问题"
    assert result["rounds"][1]["continuation_hypothesis"]["target_goal"] == "score"


def test_run_auto_campaign_progress_uses_current_round_snapshot(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        base_factors = kwargs["base_factors"]
        progress_callback = kwargs["progress_callback"]
        round_index = 1 if "ExtraFactor" not in base_factors else 2
        candidate = {
            "name": f"Auto_{round_index}",
            "expression": f"expr_{round_index}",
            "score": 60 + round_index * 10,
        }
        progress_callback(1, 2, candidate)
        return {
            "factors": [candidate],
            "best_score": candidate["score"],
            "avg_score": candidate["score"] - 5,
            "generations": 1,
            "fitness_history": {"best": [candidate["score"]], "average": [candidate["score"] - 5]},
            "round_evaluation": {
                "base_factors": list(base_factors),
                "primary_problem": f"第 {round_index} 轮问题",
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": candidate["score"]},
            },
        }

    def fake_select_continue_factors(**kwargs):
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补充一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    snapshots = []
    service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=2,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
        progress_callback=snapshots.append,
    )

    in_progress_round_two = next(
        snapshot
        for snapshot in snapshots
        if snapshot["current_round"] == 2 and snapshot["current_generation"] == 1
    )
    latest_round = in_progress_round_two["latest_round"]

    assert latest_round["round_index"] == 2
    assert latest_round["continuation_feedback"] is None
    assert latest_round["input_base_factors"] == ["Alpha1", "ExtraFactor"]
    assert latest_round["all_factors"][0]["expression"] == "expr_2"


def test_select_continue_factors_passes_parent_context_to_llm_selection(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service,
        "select_factors",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or {
            "selected_factors": ["Beta"],
            "selection_rationale": "ok",
            "per_factor_reason": {"Beta": "ok"},
        },
    )

    result = service.select_continue_factors(
        parent_result={
            "round_evaluation": {
                "base_factors": ["Alpha"],
                "primary_problem": "Sharpe 偏低",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["补充波动率因子"],
                "metric_snapshot": {"score": 61},
            },
            "factors": [
                {
                    "score": 61,
                    "report_metrics": {"sharpe": 0.5},
                    "backtest_summary": {"long_short_sharpe": 0.5, "long_short_annual": 0.08},
                    "interpretation": {"weaknesses": ["Sharpe 偏低"], "next_steps": ["补充波动率因子"]},
                }
            ],
        },
        parent_request={
            "base_factors": ["Alpha"],
            "direction": "ls_sharpe",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "universe": "hs300",
            "benchmark": "hs300",
        },
        prompt="提升风险调整后收益",
        direction="ls_sharpe",
        factor_update_mode="append",
        max_factor_count=3,
        candidate_limit=50,
    )

    assert captured["kwargs"]["direction"] == "ls_sharpe"
    assert captured["kwargs"]["start_date"] == "2024-01-01"
    assert captured["kwargs"]["end_date"] == "2024-12-31"
    assert captured["kwargs"]["universe"] == "hs300"
    assert captured["kwargs"]["benchmark"] == "hs300"
    assert "Sharpe 偏低" in captured["kwargs"]["extra_context"]
    assert result["continuation_context"]["recommended_goal"] == "ls_sharpe"


def test_select_factors_uses_real_llm_flow(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {"name": "AlphaClose", "code": "close", "category": "price", "description": "close factor"},
            {"name": "AlphaVolume", "code": "volume", "category": "volume", "description": "volume factor"},
        ],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    def fake_build_prompt(request, candidates):
        captured["request"] = request
        captured["candidates"] = candidates
        return "LLM PROMPT"

    def fake_select_with_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return {
            "selected_factors": ["AlphaVolume", "AlphaClose"],
            "selection_rationale": "LLM 按研究目标选择了量价互补因子。",
            "per_factor_reason": {
                "AlphaVolume": "成交量维度补充趋势信息。",
                "AlphaClose": "价格维度提供主要趋势信息。",
            },
        }

    monkeypatch.setattr("backend.services.auto_factor_mining_service.build_llm_factor_selector_prompt", fake_build_prompt)
    monkeypatch.setattr(service, "_select_factors_with_llm", fake_select_with_llm)

    result = service.select_factors(
        prompt="寻找趋势突破因子",
        direction="ls_sharpe",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="single_stock",
        benchmark="hs300",
        max_factor_count=2,
        candidate_limit=80,
        selection_mode="manual_genetic",
        extra_context="补充控制波动率",
        exclude_factors=["AlphaClose"],
    )

    assert result["selected_factors"] == ["AlphaVolume", "AlphaClose"]
    assert captured["llm_kwargs"]["prompt"] == "LLM PROMPT\n\n补充上下文：\n补充控制波动率"
    assert captured["llm_kwargs"]["max_factor_count"] == 2
    assert captured["llm_kwargs"]["candidates"] == [
        {"name": "AlphaVolume", "code": "volume", "category": "volume", "description": "volume factor"},
    ]
    request = captured["request"]
    assert request.prompt == "寻找趋势突破因子"
    assert request.direction == "ls_sharpe"
    assert request.start_date == "2024-01-01"
    assert request.end_date == "2024-12-31"
    assert request.universe == "single_stock"
    assert request.benchmark == "hs300"
    assert request.selection_mode == "manual_genetic"


def test_select_factors_requires_llm_api_key(monkeypatch) -> None:
    service = AutoFactorMiningService()
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [{"name": "AlphaClose", "code": "close"}],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": ""},
    )

    with pytest.raises(ValueError, match="LLM 未配置 API Key"):
        service.select_factors(
            prompt="寻找趋势突破因子",
            start_date="2024-01-01",
            end_date="2024-12-31",
            universe="single_stock",
            benchmark="hs300",
        )


def test_select_factors_with_llm_filters_unknown_names(monkeypatch) -> None:
    service = AutoFactorMiningService()

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @property
        def chat(self):
            return SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: FakeResponse(
                        '{"selected_factors":["GhostFactor","AlphaClose","AlphaClose"],'
                        '"selection_rationale":"已选择有效因子",'
                        '"per_factor_reason":{"AlphaClose":"价格因子有效"}}'
                    )
                )
            )

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    result = service._select_factors_with_llm(
        prompt="LLM PROMPT",
        max_factor_count=3,
        candidates=[
            {"name": "AlphaClose", "code": "close"},
            {"name": "AlphaVolume", "code": "volume"},
        ],
        llm_config={"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    assert result["selected_factors"] == ["AlphaClose"]
    assert result["per_factor_reason"] == {"AlphaClose": "价格因子有效"}


def test_write_candidate_report_uses_reference_generator(monkeypatch, tmp_path) -> None:
    service = AutoFactorMiningService()
    calls: dict[str, object] = {}

    def fake_generate_report(
        ls_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        title: str = "",
        output_dir: str | None = None,
        periods_per_year: int = 252,
    ) -> dict:
        calls["ls_returns"] = ls_returns
        calls["benchmark_returns"] = benchmark_returns
        calls["title"] = title
        calls["output_dir"] = output_dir
        calls["periods_per_year"] = periods_per_year
        report_path = tmp_path / "backtest_report_20260607_181500.html"
        report_path.write_text("<html>reference report</html>", encoding="utf-8")
        return {
            "report_path": str(report_path),
            "metrics": {"sharpe": 1.23, "cagr": 0.18, "max_drawdown": 0.12},
        }

    monkeypatch.setattr("backend.services.auto_factor_mining_service.generate_report", fake_generate_report)

    strategy_returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        dtype=float,
    )
    benchmark_returns = pd.Series(
        [0.001, 0.0, 0.002],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        dtype=float,
    )

    metrics, report_url = service._write_candidate_report(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=50,
    )

    assert calls["ls_returns"] is strategy_returns
    assert calls["benchmark_returns"] is benchmark_returns


def test_evaluate_expression_prefers_quantgpt_panel_execution(monkeypatch) -> None:
    service = AutoFactorMiningService()

    trade_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    stock_codes = [f"S{i:02d}" for i in range(8)]
    panel_rows = []
    factor_values = []
    for date_index, trade_date in enumerate(trade_dates):
        for stock_index, stock_code in enumerate(stock_codes):
            panel_rows.append(
                {
                    "trade_date": trade_date,
                    "stock_code": stock_code,
                    "close": 10 + stock_index + date_index,
                }
            )
            factor_values.append(float(stock_index))

    panel_df = pd.DataFrame(
        panel_rows
    )
    factor_series = pd.Series(factor_values, index=panel_df.index, dtype=float)

    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: panel_df.copy(),
    )
    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {
            "success": True,
            "valid": True,
            "message": "OK",
            "raw": {
                "input_expression": kwargs["expression"],
                "adapted_expression": "rank(close)",
            },
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "execute_on_panel",
        lambda panel, expression: SimpleNamespace(
            factor_series=factor_series,
            engine_type="quantgpt",
            dialect="quantgpt_local",
            canonical_expression="rank(close)",
            canonical_ast={"operators": ["rank"], "fields": ["close"]},
            diagnostics=[],
            execution_meta={"panel_rows": len(panel)},
            metrics_source="quantgpt_expression_engine",
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_benchmark_returns",
        lambda benchmark, start_date, end_date: pd.Series(dtype=float),
    )
    monkeypatch.setattr(
        service,
        "_write_candidate_report",
        lambda strategy_returns, benchmark_returns, periods_per_year: ({"sharpe": 1.0}, "/reports/mock.html"),
    )

    result = service.evaluate_expression(
        expression="rank(close)",
        prompt="test quantgpt path",
        stock_codes=stock_codes,
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is not None
    assert result.engine_type == "quantgpt"
    assert result.dialect == "quantgpt_local"
    assert result.canonical_expression == "rank(close)"
    assert result.execution_meta["panel_rows"] == len(panel_df)
    assert result.execution_meta["research_tools"]["validation"]["valid"] is True


def test_evaluate_expression_returns_none_when_quantgpt_fails(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {"success": True, "valid": True, "message": "OK", "raw": {}},
    )
    monkeypatch.setattr(
        service,
        "_diagnose_candidate_failure",
        lambda **kwargs: {
            "success": True,
            "report": "表达式需要替换字段",
            "key_findings": ["字段不支持"],
            "improvement_suggestions": ["改用 close"],
            "raw": {},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("quantgpt failed")),
    )

    result = service.evaluate_expression(
        expression="close",
        prompt="test invalid path",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_evaluate_expression_returns_none_when_quantgpt_panel_is_empty(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {"success": True, "valid": True, "message": "OK", "raw": {}},
    )
    monkeypatch.setattr(
        service,
        "_diagnose_candidate_failure",
        lambda **kwargs: {
            "success": True,
            "report": "panel 为空",
            "key_findings": ["无有效 panel"],
            "improvement_suggestions": ["调整 universe"],
            "raw": {},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: pd.DataFrame(),
    )

    result = service.evaluate_expression(
        expression="close",
        prompt="test empty panel invalid",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_evaluate_expression_returns_none_when_quantgpt_validation_fails(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {
            "success": True,
            "valid": False,
            "message": "syntax error",
            "raw": {"adapted_expression": "bad(expr)"},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("validation fail should short-circuit")),
    )

    result = service.evaluate_expression(
        expression="bad(expr)",
        prompt="test validation fail",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_quantgpt_engine_adapts_generator_dialect_before_execution() -> None:
    service = AutoFactorMiningService()

    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "stock_code": ["S001"] * 6,
            "close": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
            "volume": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
        }
    )

    result = service._quantgpt_engine.execute_on_panel(frame, "Ref($close, 1)")

    assert result.factor_series is not None
    assert result.canonical_expression == "ts_shift(close,1)"
    assert float(result.factor_series.iloc[2]) == 11.0


def test_quantgpt_engine_builds_panel_from_unnamed_datetime_index() -> None:
    service = AutoFactorMiningService()

    raw_frame = pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 120.0, 140.0],
            "amount": [1000.0, 1320.0, 1680.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    panel = service._quantgpt_engine.build_panel_data(
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-03",
        expression="rank(close)",
        stock_data_loader=lambda stock_code, start_date, end_date: raw_frame.copy(),
    )

    assert not panel.empty
    assert "trade_date" in panel.columns
    assert panel["trade_date"].notna().all()


def test_filter_supported_expressions_uses_quantgpt_engine_only(monkeypatch) -> None:
    service = AutoFactorMiningService()
    sample_frames = [
        pd.DataFrame(
            {
                "open": [10.0, 10.2, 10.4],
                "high": [10.3, 10.5, 10.7],
                "low": [9.8, 10.0, 10.2],
                "close": [10.1, 10.3, 10.6],
                "volume": [1000, 1100, 1200],
                "amount": [10000, 11330, 12720],
            }
        )
    ]

    monkeypatch.setattr(
        service._quantgpt_engine,
        "can_execute_on_frames",
        lambda expression, frames: expression == "rank(close)",
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: (_ for _ in ()).throw(AssertionError("should not call factorhub calculator")),
    )

    supported = service._filter_supported_expressions(
        ["rank(close)", "bad(expr)"],
        sample_frames=sample_frames,
        limit=2,
    )

    assert supported == ["rank(close)"]


def test_run_auto_mining_raises_when_llm_candidates_are_not_executable(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])
    monkeypatch.setattr(
        service,
        "generate_candidate_expressions",
        lambda **kwargs: ["rank(ts_mean(close, 5))", "ts_rank(correlation(close, volume, 10), 20)"],
    )
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: [],
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    try:
        service.run_auto_mining(
            prompt="提升量价复合因子的稳定性",
            base_factors=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            universe="hs300",
            benchmark="hs300",
            n_groups=5,
            holding_period=5,
            n_candidates=1,
        )
    except ValueError as exc:
        assert str(exc) == "未生成可执行候选表达式"
    else:
        raise AssertionError("expected auto mining to fail when no QuantGPT candidate is executable")


def test_run_auto_mining_keeps_retrying_until_quantgpt_yields_valid_candidate(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["000001.SZ"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])

    generate_calls = {"count": 0}

    def fake_generate_candidate_expressions(**kwargs):
        generate_calls["count"] += 1
        if generate_calls["count"] == 1:
            return ["bad_expr_1"]
        return ["good_expr_2"]

    monkeypatch.setattr(service, "generate_candidate_expressions", fake_generate_candidate_expressions)
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: expressions[:limit],
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: pd.Series(range(len(df)), index=df.index, dtype=float),
    )

    def fake_evaluate_expression(**kwargs):
        expression = kwargs["expression"]
        if expression == "bad_expr_1":
            return None
        return SimpleNamespace(
            expression=expression,
            score=88.0,
            grade="A",
            report_metrics={"sharpe": 1.2, "max_drawdown": 0.1},
            backtest_summary={
                "long_short_sharpe": 1.2,
                "long_short_annual": 0.2,
                "top_group_sharpe": 1.1,
                "monotonicity_score": 0.8,
                "spread": 0.03,
                "group_returns": {"top_minus_bottom_mean": 0.03},
                "rank_ic_mean": 0.05,
                "ic_mean": 0.05,
                "ic_ir": 1.0,
                "ic_win_rate": 0.7,
                "turnover": 0.2,
                "wq_fitness": 1.05,
            },
            wq_brain={
                "wq_rating": "A",
                "wq_fitness": 1.05,
                "wq_sharpe": 1.2,
                "wq_returns": 0.2,
                "wq_turnover": 0.2,
                "submittable": True,
            },
            component_scores={"total_score": 88.0},
            anti_overfit={"score": 80.0, "recommendation": "推荐", "tests": []},
            interpretation={
                "summary": "retry ok",
                "weaknesses": [],
                "next_steps": ["继续迭代"],
                "rating": "A",
                "rating_reason": "推荐",
                "improvement_ideas": ["继续迭代"],
            },
            diagnostics=[],
            report_url="/api/mining/reports/retry.html",
        )

    monkeypatch.setattr(service, "evaluate_expression", fake_evaluate_expression)

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=1,
    )

    assert generate_calls["count"] >= 2
    assert result["best_score"] == 88.0
    assert result["factors"][0]["expression"] == "good_expr_2"


def test_run_auto_mining_accepts_bound_benchmark_and_report_helpers(monkeypatch) -> None:
    service = AutoFactorMiningService()

    trade_dates = pd.date_range("2024-01-01", periods=40, freq="D")
    stock_codes = [f"{i:06d}.SZ" for i in range(1, 9)]
    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: stock_codes,
        get_stock_data=lambda stock_code, start_date, end_date: pd.DataFrame(
            {
                "stock_code": [stock_code] * 40,
                "open": [10.0 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "high": [10.5 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "low": [9.5 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "close": [10.2 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "volume": [1000.0 + i * 10 + (20 * stock_codes.index(stock_code)) for i in range(40)],
                "amount": [10000.0 + i * 100 + (200 * stock_codes.index(stock_code)) for i in range(40)],
            },
            index=trade_dates,
        ),
        get_benchmark_returns=lambda benchmark, start_date, end_date: pd.DataFrame(
            {
                "trade_date": trade_dates,
                "daily_return": [0.001] * 40,
            }
        ),
    )

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close"])
    monkeypatch.setattr(
        service,
        "generate_candidate_expressions",
        lambda **kwargs: ["rank(close)"],
    )
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: expressions[:limit],
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: pd.to_numeric(df["close"], errors="coerce").astype(float),
    )
    monkeypatch.setattr(
        service,
        "_write_candidate_report",
        lambda strategy_returns, benchmark_returns, periods_per_year: (
            {"sharpe": 1.0, "max_drawdown": 0.1},
            "/api/mining/reports/mock.html",
        ),
    )

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close"],
        start_date="2024-01-01",
        end_date="2024-02-29",
        universe="hs300",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        n_candidates=1,
    )

    assert result["best_score"] >= 0
    assert len(result["factors"]) == 1
    assert result["factors"][0]["expression"] == "rank(close)"


def test_factor_calculator_supports_integer_volume_sma() -> None:
    df = pd.DataFrame(
        {
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000 + i * 10 for i in range(40)],
            "amount": [10000 + i * 100 for i in range(40)],
        }
    )

    volume_sma = factor_service.calculator.calculate(df.copy(), "SMA(volume, timeperiod=5)")
    composite = factor_service.calculator.calculate(
        df.copy(),
        "((SMA(close, timeperiod=5) - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)) * (volume / SMA(volume, timeperiod=20))",
    )

    assert int(volume_sma.dropna().shape[0]) > 0
    assert int(composite.dropna().shape[0]) > 0


def test_quantgpt_expression_engine_supports_integer_volume_sma() -> None:
    service = AutoFactorMiningService()
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["AAA"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000 + i * 10 for i in range(40)],
            "amount": [10000 + i * 100 for i in range(40)],
        }
    )

    volume_sma = service._quantgpt_engine.execute_on_panel(df.copy(), "SMA(volume, timeperiod=5)").factor_series
    composite = service._quantgpt_engine.execute_on_panel(
        df.copy(),
        "((SMA(close, timeperiod=5) - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)) * (volume / SMA(volume, timeperiod=20))",
    ).factor_series

    assert volume_sma is not None
    assert composite is not None
    assert int(volume_sma.dropna().shape[0]) > 0
    assert int(composite.dropna().shape[0]) > 0

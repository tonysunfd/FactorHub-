from __future__ import annotations

from typing import Any, Callable


class FactorHubRDAgentCoder:
    """本地 coder：保持 FactorHub V3 数据/执行约束，生成可运行候选。"""

    stage_name = "coding"
    developer_name = "FactorHubRDAgentCoder"

    def __init__(
        self,
        *,
        code_experiment_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self._code_experiment_fn = code_experiment_fn

    def develop(
        self,
        *,
        config: Any,
        experiment: dict[str, Any],
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        coded_experiment = self._code_experiment_fn(
            config=config,
            experiment=experiment,
            hypothesis=hypothesis,
            rounds=rounds,
            iteration=iteration,
        )
        return {
            **coded_experiment,
            "developer_name": self.developer_name,
            "developer_stage": self.stage_name,
        }


class FactorHubRDAgentRunner:
    """本地 runner：保持 FactorHub V3 自有数据源与回测/评分体系。"""

    stage_name = "running"
    developer_name = "FactorHubRDAgentRunner"

    def __init__(
        self,
        *,
        run_experiment_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self._run_experiment_fn = run_experiment_fn

    def develop(
        self,
        *,
        config: Any,
        coded_experiment: dict[str, Any],
        hypothesis: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
        sota_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        run_result = self._run_experiment_fn(
            config=config,
            coded_experiment=coded_experiment,
            hypothesis=hypothesis,
            rounds=rounds,
            iteration=iteration,
            sota_candidates=sota_candidates,
        )
        return {
            **run_result,
            "developer_name": self.developer_name,
            "developer_stage": self.stage_name,
        }


class FactorHubRDAgentFeedback:
    """本地 feedback：保持 FactorHub V3 自有接受策略与总结逻辑。"""

    stage_name = "feedback"
    developer_name = "FactorHubRDAgentFeedback"

    def __init__(
        self,
        *,
        generate_feedback_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self._generate_feedback_fn = generate_feedback_fn

    def generate_feedback(
        self,
        *,
        config: Any,
        hypothesis: dict[str, Any],
        experiment: dict[str, Any],
        run_result: dict[str, Any],
        rounds: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        feedback = self._generate_feedback_fn(
            config=config,
            hypothesis=hypothesis,
            experiment=experiment,
            run_result=run_result,
            rounds=rounds,
            iteration=iteration,
        )
        return {
            **feedback,
            "developer_name": self.developer_name,
            "developer_stage": self.stage_name,
        }


def build_factorhub_rdagent_pipeline_metadata(*, execution_mode: str) -> dict[str, Any]:
    return {
        "execution_mode": execution_mode,
        "proposal": "reference_rdagent_proposal" if execution_mode == "upstream_rdagent" else "factorhub_local_hypothesis",
        "coder": FactorHubRDAgentCoder.developer_name,
        "runner": FactorHubRDAgentRunner.developer_name,
        "feedback": FactorHubRDAgentFeedback.developer_name,
        "data_source": "factorhub_v3_local_data_source",
        "evaluation_system": "factorhub_v3_local_evaluation",
    }

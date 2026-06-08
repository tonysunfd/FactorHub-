from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MAX_RDAGENT_ITERATIONS = 8


class RDAgentTaskCancelled(RuntimeError):
    """Raised when the caller cancels a running RDAgent task."""


@dataclass
class RDAgentMiningConfig:
    task_id: str
    objective: str
    max_iterations: int = 1
    candidates_per_iteration: int = 1
    base_factors: list[str] = field(default_factory=list)
    candidate_universe: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    universe: str = "all"
    benchmark: str = "000300.SH"
    n_groups: int = 5
    holding_period: int = 5
    direction: str | None = None
    neutralize_industry: bool = True
    neutralize_cap: bool = True
    acceptance_policy: Any = None
    continuation_of: str | None = None
    previous_feedback_id: str | None = None
    previous_expressions: list[str] = field(default_factory=list)
    cancel_check: Callable[[], None] | None = None


class RDAgentFactorMiningService:
    """Lightweight orchestration over a router-provided RDAgent backend."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def run(
        self,
        *,
        task_id: str,
        config: RDAgentMiningConfig,
        on_progress: Callable[[int, str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        rounds: list[dict[str, Any]] = []
        retained_factors: list[dict[str, Any]] = []
        watchlist_factors: list[dict[str, Any]] = []
        fitness_history: dict[str, list[float]] = {"best": [], "average": []}

        total_iterations = max(1, min(int(config.max_iterations or 1), MAX_RDAGENT_ITERATIONS))
        total_stages = total_iterations * 5
        stage_count = 0

        def emit(stage: str, iteration: int, payload: dict[str, Any]) -> None:
            nonlocal stage_count
            stage_count += 1
            progress = min(int(stage_count / max(total_stages, 1) * 100), 99)
            if on_progress:
                on_progress(progress, stage, {"iteration": iteration, "payload": payload})

        for iteration in range(1, total_iterations + 1):
            if config.cancel_check:
                config.cancel_check()

            hypothesis = self.backend.propose_hypothesis(config=config, rounds=rounds, iteration=iteration)
            emit("rdagent_hypothesis", iteration, hypothesis)

            if config.cancel_check:
                config.cancel_check()

            experiment = self.backend.hypothesis_to_experiment(
                config=config,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
            )
            emit("rdagent_experiment", iteration, experiment)

            if config.cancel_check:
                config.cancel_check()

            coded_experiment = self.backend.code_experiment(
                config=config,
                experiment=experiment,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
            )
            emit("rdagent_coding", iteration, coded_experiment)

            if config.cancel_check:
                config.cancel_check()

            run_result = self.backend.run_experiment(
                config=config,
                coded_experiment=coded_experiment,
                hypothesis=hypothesis,
                rounds=rounds,
                iteration=iteration,
            )
            candidates = list(run_result.get("candidates") or [])
            evaluation = {
                "iteration": iteration,
                "metrics": run_result.get("metrics") or {},
                "report_ref": run_result.get("report_ref"),
                "backtest_engine": run_result.get("backtest_engine") or "factorhub",
                "best_score": float((run_result.get("metrics") or {}).get("score") or 0.0),
                "avg_score": _average_score(candidates),
            }
            emit(
                "rdagent_running",
                iteration,
                {
                    "candidates": candidates,
                    "evaluation": evaluation,
                    "best_score": evaluation["best_score"],
                    "avg_score": evaluation["avg_score"],
                },
            )

            if config.cancel_check:
                config.cancel_check()

            feedback = self.backend.generate_feedback(
                config=config,
                hypothesis=hypothesis,
                experiment=experiment,
                run_result=run_result,
                rounds=rounds,
                iteration=iteration,
            )
            emit(
                "rdagent_feedback",
                iteration,
                {
                    "feedback": feedback,
                    "candidates": candidates,
                    "best_score": evaluation["best_score"],
                    "avg_score": evaluation["avg_score"],
                },
            )

            round_item = {
                "round_index": iteration,
                "hypothesis": hypothesis,
                "experiment": experiment,
                "coded_experiment": coded_experiment,
                "candidates": candidates,
                "all_factors": candidates,
                "evaluation": evaluation,
                "feedback": feedback,
            }
            rounds.append(round_item)
            retained_factors.extend([item for item in candidates if item.get("status") == "accepted"])
            watchlist_factors.extend([item for item in candidates if item.get("status") == "watchlist"])
            fitness_history["best"].append(evaluation["best_score"])
            fitness_history["average"].append(evaluation["avg_score"])

        final_round = rounds[-1] if rounds else {}
        return {
            "task_id": task_id,
            "objective": config.objective,
            "rounds": rounds,
            "retained_factors": retained_factors,
            "watchlist_factors": watchlist_factors,
            "fitness_history": fitness_history,
            "final_round_result": {
                **(final_round.get("evaluation") or {}),
                "factors": final_round.get("candidates") or [],
            },
        }


def _average_score(candidates: list[dict[str, Any]]) -> float:
    scores = [float(item.get("score") or 0.0) for item in candidates if item.get("score") is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

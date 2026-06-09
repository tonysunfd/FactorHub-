from __future__ import annotations

from typing import Any, Callable


FitnessHistory = dict[str, list[float]]


def _strip_control_chars(value: str) -> str:
    return "".join(
        ch for ch in str(value)
        if ch in ("\t", "\n", "\r") or ord(ch) >= 32
    )


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_control_chars(value)
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_payload(key) if isinstance(key, str) else key: sanitize_payload(item)
            for key, item in value.items()
        }
    return value


def empty_fitness_history() -> FitnessHistory:
    return {"best": [], "average": []}


def normalize_fitness_history(history: Any) -> FitnessHistory:
    if not isinstance(history, dict):
        return empty_fitness_history()
    best = [float(value) for value in history.get("best", [])]
    average = [float(value) for value in history.get("average", [])]
    return {"best": best, "average": average}


def append_fitness_snapshot(task: dict[str, Any], best_fitness: float, avg_fitness: float) -> FitnessHistory:
    history = normalize_fitness_history(task.get("fitness_history"))
    history["best"].append(float(best_fitness))
    history["average"].append(float(avg_fitness))
    task["fitness_history"] = history
    return history


def update_task_progress(
    task: dict[str, Any],
    *,
    generation: int,
    total_generations: int,
    best_fitness: float,
    avg_fitness: float,
    progress: int | None = None,
    candidates: list[dict[str, Any]] | None = None,
    history: FitnessHistory | None = None,
) -> None:
    normalized_total = max(int(total_generations or 0), 1)
    task["progress"] = int(progress if progress is not None else generation / normalized_total * 100)
    task["current_generation"] = int(generation)
    task["total_generations"] = int(total_generations)
    task["best_fitness"] = float(best_fitness)
    task["avg_fitness"] = float(avg_fitness)
    task["fitness_history"] = normalize_fitness_history(history) if history is not None else append_fitness_snapshot(
        task,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
    )
    if candidates is not None:
        task["candidates"] = candidates


def update_task_from_candidates(
    task: dict[str, Any],
    *,
    generation: int,
    total_generations: int,
    candidates: list[dict[str, Any]],
    score_getter: Callable[[dict[str, Any]], float],
    progress: int | None = None,
) -> None:
    scores = [float(score_getter(candidate)) for candidate in candidates]
    best_fitness = max(scores) if scores else 0.0
    avg_fitness = round(sum(scores) / len(scores), 4) if scores else 0.0
    update_task_progress(
        task,
        generation=generation,
        total_generations=total_generations,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
        progress=progress,
        candidates=candidates,
    )


def finalize_task_result(
    task: dict[str, Any],
    result: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    generation_key: str = "generations",
    best_key: str = "best_fitness",
    avg_key: str = "avg_fitness",
    history_key: str = "fitness_history",
    round_evaluation: dict[str, Any] | None = None,
) -> None:
    generations = int(result.get(generation_key, 0) or 0)
    best_fitness = float(result.get(best_key, result.get("best_score", 0.0)) or 0.0)
    avg_fitness = float(result.get(avg_key, result.get("avg_score", 0.0)) or 0.0)
    history = normalize_fitness_history(result.get(history_key))

    task["status"] = "completed"
    task["progress"] = 100
    task["result"] = result
    task["current_generation"] = generations
    task["total_generations"] = generations
    task["best_fitness"] = best_fitness
    task["avg_fitness"] = avg_fitness
    task["fitness_history"] = history
    if candidates is not None:
        task["candidates"] = candidates
    if round_evaluation is not None:
        task["round_evaluation"] = round_evaluation


def build_mining_status_payload(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "error": task.get("error"),
    }

    if task["status"] == "completed" and task.get("result"):
        result = task["result"]
        response_data["current_generation"] = result.get("generations", 0)
        response_data["total_generations"] = result.get("generations", 0)
        response_data["best_fitness"] = result.get("best_fitness", result.get("best_score", 0))
        response_data["avg_fitness"] = result.get("avg_fitness", result.get("avg_score", 0))
        response_data["fitness_history"] = normalize_fitness_history(result.get("fitness_history"))
    else:
        response_data["current_generation"] = task.get("current_generation", 0)
        response_data["total_generations"] = task.get("total_generations", 10)
        response_data["best_fitness"] = task.get("best_fitness", 0.0)
        response_data["avg_fitness"] = task.get("avg_fitness", 0.0)
        response_data["fitness_history"] = normalize_fitness_history(task.get("fitness_history"))

    response_data["candidates"] = task.get("candidates", [])
    response_data["round_evaluation"] = task.get("round_evaluation")
    return sanitize_payload(response_data)


def build_auto_campaign_status(task_id: str, task: dict[str, Any]) -> dict[str, Any]:
    payload = build_mining_status_payload(task_id, task)
    payload["current_round"] = task.get("current_round", 0)
    payload["total_rounds"] = task.get("total_rounds", 0)
    payload["retained_count"] = task.get("retained_count", 0)
    payload["upstream_status"] = task.get("upstream_status")
    payload["rounds"] = task.get("rounds", [])
    payload["latest_round"] = task.get("latest_round")
    return sanitize_payload(payload)

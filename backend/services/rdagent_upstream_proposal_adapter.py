from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from backend.services.llm_config_service import llm_config_service
from backend.services.rdagent_runtime import resolve_rdagent_project_root, resolve_rdagent_python


class RDAgentUpstreamProposalAdapter:
    """复用 reference RD-Agent 的 hypothesis + experiment 生成链路。"""

    def generate_round_plan(
        self,
        *,
        objective: str,
        iteration: int,
        candidate_universe: list[str],
        current_base_factors: list[str],
        rounds: list[dict[str, Any]],
    ) -> dict[str, Any]:
        project_root = resolve_rdagent_project_root()
        python_path = resolve_rdagent_python()
        env = self._build_runtime_env(project_root)
        payload = {
            "objective": objective,
            "iteration": iteration,
            "candidate_universe": list(candidate_universe or []),
            "current_base_factors": list(current_base_factors or []),
            "rounds": self._serialize_rounds(rounds),
        }
        proc = subprocess.run(
            [str(python_path), "-c", self._script()],
            capture_output=True,
            text=True,
            timeout=180,
            env={**env, "FACTORHUB_RDAGENT_PAYLOAD": json.dumps(payload, ensure_ascii=False)},
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or "unknown upstream rdagent error"
            raise ValueError(f"upstream_rdagent proposal 生成失败：{detail}")
        try:
            stdout = (proc.stdout or "").strip()
            if not stdout:
                return {}
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return self._extract_last_json_object(stdout)
        except Exception as exc:
            raise ValueError(f"upstream_rdagent proposal 返回结果解析失败：{exc}") from exc

    def _extract_last_json_object(self, content: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, char in enumerate(content):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
        for candidate in reversed(candidates):
            if "hypothesis" in candidate or "tasks" in candidate:
                return candidate
        if candidates:
            return candidates[-1]
        raise ValueError("stdout 中未找到可解析的 JSON 对象")

    def _build_runtime_env(self, project_root: Path) -> dict[str, str]:
        config = llm_config_service.get_runtime_config()
        api_key = str(config.get("api_key") or "").strip()
        base_url = str(config.get("base_url") or "").strip()
        model = str(config.get("model") or "").strip()
        normalized_model = model if "/" in model or not model else f"openai/{model}"
        chat_temperature = "1" if normalized_model.startswith("openai/gpt-5") else "0.5"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        env["FACTORHUB_RDAGENT_PROJECT_ROOT"] = str(project_root)
        if api_key:
            for key in [
                "OPENAI_API_KEY",
                "LITELLM_OPENAI_API_KEY",
                "CHAT_OPENAI_API_KEY",
                "EMBEDDING_OPENAI_API_KEY",
                "LITELLM_CHAT_OPENAI_API_KEY",
                "LITELLM_EMBEDDING_OPENAI_API_KEY",
            ]:
                env[key] = api_key
        if base_url:
            for key in [
                "OPENAI_BASE_URL",
                "CHAT_OPENAI_BASE_URL",
                "EMBEDDING_OPENAI_BASE_URL",
                "LITELLM_CHAT_OPENAI_BASE_URL",
                "LITELLM_EMBEDDING_OPENAI_BASE_URL",
            ]:
                env[key] = base_url
        if normalized_model:
            env["CHAT_MODEL"] = normalized_model
            env["LITELLM_CHAT_MODEL"] = normalized_model
            env["CHAT_TEMPERATURE"] = chat_temperature
            env["LITELLM_CHAT_TEMPERATURE"] = chat_temperature
        env.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        env.setdefault("LITELLM_EMBEDDING_MODEL", env["EMBEDDING_MODEL"])
        return env

    def _serialize_rounds(self, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for round_item in rounds:
            serialized.append(
                {
                    "hypothesis": round_item.get("hypothesis") or {},
                    "feedback": round_item.get("feedback") or {},
                    "candidates": [
                        {
                            "name": item.get("name"),
                            "expression": item.get("expression"),
                            "description": item.get("description"),
                            "factor_formulation": item.get("factor_formulation"),
                            "variables": item.get("variables") or {},
                        }
                        for item in (round_item.get("candidates") or [])
                    ],
                }
            )
        return serialized

    def _script(self) -> str:
        return r"""
import json
import os
import sys

sys.path.insert(0, os.environ["FACTORHUB_RDAGENT_PROJECT_ROOT"])

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback, Trace
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)
from rdagent.utils.agent.tpl import T


class MinimalQlibScenario(Scenario):
    @property
    def background(self) -> str:
        return (
            "This is a factor research scenario adapted for FactorHub V3. "
            "Use the provided candidate fields and base factors to propose practical alpha factors."
        )

    @property
    def rich_style_description(self) -> str:
        return ""

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None, action=None) -> str:
        background = self.background
        source_data = (
            "Available fields come from FactorHub V3 local data source. "
            "Candidate fields and base factors will be provided in the user instruction."
        )
        factor_output = T("scenarios.qlib.prompts:factor_experiment_output_format").r()
        factor_simulator = (
            "FactorHub V3 will execute candidate expressions or generated implementations "
            "against local market data and evaluate them with local scoring."
        )
        if simple_background:
            return f"Background of the scenario:\n{background}"
        return (
            f"Background of the scenario:\n{background}\n"
            f"The source data you can use:\n{source_data}\n"
            f"The output of your code should be in the format:\n{factor_output}\n"
            f"The simulator user can use to test your factor:\n{factor_simulator}\n"
        )

    def get_runtime_environment(self) -> str:
        return "FactorHub V3 local Python environment with local factor evaluation."

    @property
    def experiment_setting(self) -> str | None:
        return None

payload = json.loads(os.environ["FACTORHUB_RDAGENT_PAYLOAD"])
scen = MinimalQlibScenario()
trace = Trace(scen=scen)

for idx, round_item in enumerate(payload.get("rounds", [])):
    hypo_payload = round_item.get("hypothesis") or {}
    hypothesis = Hypothesis(
        hypothesis=str(hypo_payload.get("statement") or "Historical hypothesis"),
        reason=str(hypo_payload.get("reason") or "Historical reason"),
        concise_reason=str(hypo_payload.get("reason") or "Historical reason"),
        concise_observation=str(hypo_payload.get("expected_signal") or "Historical observation"),
        concise_justification=str(hypo_payload.get("reason") or "Historical justification"),
        concise_knowledge=str(hypo_payload.get("expected_signal") or "Historical knowledge"),
    )
    tasks = []
    for candidate in round_item.get("candidates", []):
        factor_name = str(candidate.get("name") or candidate.get("expression") or f"HistoricalFactor{idx + 1}")
        formulation = str(candidate.get("factor_formulation") or candidate.get("expression") or factor_name)
        tasks.append(
            FactorTask(
                factor_name=factor_name,
                factor_description=str(candidate.get("description") or factor_name),
                factor_formulation=formulation,
                variables=dict(candidate.get("variables") or {}),
            )
        )
    exp = QlibFactorExperiment(tasks, hypothesis=hypothesis)
    feedback_payload = round_item.get("feedback") or {}
    feedback = HypothesisFeedback(
        reason=str(feedback_payload.get("reason") or "Historical feedback"),
        decision=bool(feedback_payload.get("decision")),
        code_change_summary="",
        observations=str(feedback_payload.get("observations") or ""),
        hypothesis_evaluation=str(feedback_payload.get("hypothesis_evaluation") or ""),
        new_hypothesis=str(feedback_payload.get("next_hypothesis") or ""),
        acceptable=bool(feedback_payload.get("acceptable")),
    )
    trace.hist.append((exp, feedback))
    trace.dag_parent.append((idx - 1,) if idx > 0 else ())
    trace.idx2loop_id[idx] = idx

instruction = (
    f"Objective: {payload.get('objective')}\n"
    f"Iteration: {payload.get('iteration')}\n"
    f"Candidate fields: {payload.get('candidate_universe')}\n"
    f"Current base factors: {payload.get('current_base_factors')}\n"
    "Please focus on factor generation and avoid duplicating historical factors."
)
plan = {"user_instruction": instruction}

hypothesis_gen = QlibFactorHypothesisGen(scen)
hypothesis = hypothesis_gen.gen(trace, plan)
experiment_gen = QlibFactorHypothesis2Experiment()
exp = experiment_gen.convert(hypothesis, trace)

tasks = []
for task in getattr(exp, "tasks", []) or getattr(exp, "sub_tasks", []):
    tasks.append(
        {
            "factor_name": getattr(task, "factor_name", ""),
            "description": getattr(task, "factor_description", ""),
            "formulation": getattr(task, "factor_formulation", ""),
            "variables": getattr(task, "variables", {}) or {},
        }
    )

print(
    json.dumps(
        {
            "hypothesis": {
                "statement": getattr(hypothesis, "hypothesis", ""),
                "reason": getattr(hypothesis, "reason", ""),
                "concise_reason": getattr(hypothesis, "concise_reason", ""),
                "concise_observation": getattr(hypothesis, "concise_observation", ""),
                "concise_justification": getattr(hypothesis, "concise_justification", ""),
                "concise_knowledge": getattr(hypothesis, "concise_knowledge", ""),
            },
            "tasks": tasks,
        },
        ensure_ascii=False,
    )
)
"""

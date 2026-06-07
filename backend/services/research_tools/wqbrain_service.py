from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from backend.core.settings import settings
from backend.core.database import get_db_session
from backend.repositories.factor_repository import FactorRepository
from backend.services.factor_service import factor_service

from .quantgpt_client import QuantGPTClient
from .wq_brain_direct_client import (
    SUBMIT_THRESHOLDS,
    WQBrainDirectClient,
    configured_accounts as direct_configured_accounts,
    is_configured as direct_is_configured,
)


class WQBrainService:
    def __init__(self, client: QuantGPTClient | None = None):
        self._client_override = client

    def _client(self) -> QuantGPTClient:
        return self._client_override or QuantGPTClient()

    def _has_client_override(self) -> bool:
        return self._client_override is not None

    def _use_direct(self) -> bool:
        return True

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _classify_alpha_check(self, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("ok"):
            return {
                "final_status": "ERROR",
                "status": None,
                "sc_result": None,
                "sc_value": None,
                "sc_limit": None,
                "fitness": None,
                "sharpe": None,
                "grade": None,
                "error": data.get("error", "unknown"),
            }

        status = (data.get("status") or "").upper()
        is_data = data.get("is", {})
        checks = is_data.get("checks", [])
        sc_check = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)
        sc_result = sc_check.get("result") if sc_check else None

        if status == "ACTIVE":
            final = "ACTIVE"
        elif sc_result == "FAIL":
            final = "SC_FAIL"
        elif status == "UNSUBMITTED":
            final = "UNSUBMITTED"
        elif sc_result == "PENDING" or sc_result is None:
            final = "SC_PENDING"
        else:
            final = "OTHER_FAIL"

        return {
            "final_status": final,
            "status": status,
            "sc_result": sc_result,
            "sc_value": sc_check.get("value") if sc_check else None,
            "sc_limit": sc_check.get("limit") if sc_check else None,
            "fitness": self._safe_float(is_data.get("fitness")),
            "sharpe": self._safe_float(is_data.get("sharpe")),
            "returns": self._safe_float(is_data.get("returns")),
            "turnover": self._safe_float(is_data.get("turnover")),
            "grade": data.get("grade"),
            "dateCreated": data.get("dateCreated"),
        }

    def _env_path(self) -> Path:
        return settings.BASE_DIR / ".env"

    def _safe_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _mask_token(self, token: str) -> str:
        token = token.strip()
        if not token:
            return ""
        if len(token) <= 8:
            return "*" * len(token)
        return f"{token[:4]}***{token[-4:]}"

    def _write_env_values(self, updates: dict[str, str]) -> None:
        env_path = self._env_path()
        lines: list[str] = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        seen: set[str] = set()
        new_lines: list[str] = []
        for line in lines:
            replaced = False
            for key, value in updates.items():
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}")
                    seen.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        for key, value in updates.items():
            if key not in seen:
                new_lines.append(f"{key}={value}")

        env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")

    def _extract_factor_candidate(self, factor: dict[str, Any]) -> dict[str, Any] | None:
        snapshot = factor.get("latest_task_snapshot") or {}
        payload = snapshot.get("payload") or factor.get("task_metadata") or {}
        expression = (payload.get("expression") or factor.get("code") or "").strip()
        if not expression:
            return None

        origin_type = str(factor.get("origin_type") or "").strip().lower()
        snapshot_source = str(snapshot.get("source") or payload.get("source") or factor.get("task_metadata", {}).get("source") or "").strip().lower()
        is_auto_origin = origin_type == "auto_mining"
        is_auto_source = "factorhub_auto" in snapshot_source or "auto_iteration" in snapshot_source
        is_rdagent_origin = origin_type == "rdagent_mining"
        is_rdagent_source = "rdagent" in snapshot_source
        if not is_auto_origin and not is_auto_source and not is_rdagent_origin and not is_rdagent_source:
            return None
        if is_rdagent_origin or is_rdagent_source:
            metadata = factor.get("task_metadata") if isinstance(factor.get("task_metadata"), dict) else {}
            review_status = str(metadata.get("review_status") or payload.get("review_status") or "").strip().lower()
            if review_status != "confirmed":
                return None

        scoring = payload.get("scoring") or {}
        backtest = payload.get("backtest_summary") or {}
        wq_brain = payload.get("wq_brain") or {}
        report_metrics = payload.get("report_metrics") or {}
        interpretation = payload.get("interpretation") or {}

        score = self._safe_float(scoring.get("score") or payload.get("score"))
        ls_sharpe = self._safe_float(backtest.get("long_short_sharpe"))
        ls_return = self._safe_float(backtest.get("long_short_annual"))
        wq_return = self._safe_float(wq_brain.get("wq_returns"))
        latest_rating = (
            wq_brain.get("submission_status")
            or wq_brain.get("final_status")
            or wq_brain.get("platform_status")
            or "UNSUBMITTED"
        )

        if score is None and ls_sharpe is None and wq_return is None and not report_metrics:
            return None

        return {
            "factor_id": factor.get("id"),
            "name": factor.get("name"),
            "category": factor.get("category"),
            "origin_type": factor.get("origin_type"),
            "expression": expression,
            "score": score,
            "grade": scoring.get("grade") or payload.get("grade"),
            "wq_rating": wq_brain.get("wq_rating") or wq_brain.get("rating"),
            "ls_sharpe": ls_sharpe,
            "ls_return": ls_return,
            "wq_return": wq_return,
            "report_sharpe": self._safe_float(report_metrics.get("sharpe")),
            "report_url": payload.get("report_url"),
            "source_task_id": snapshot.get("task_id") or factor.get("task_metadata", {}).get("task_id"),
            "alpha_id": wq_brain.get("alpha_id"),
            "submission_status": latest_rating,
            "platform_status": wq_brain.get("platform_status") or wq_brain.get("status"),
            "sc_result": wq_brain.get("sc_result"),
            "matched_rules": wq_brain.get("matched_rules") or [],
            "snapshot_id": snapshot.get("id"),
            "updated_at": factor.get("updated_at"),
            "report_metrics": report_metrics,
            "backtest_summary": backtest,
            "wq_brain": wq_brain,
            "interpretation": interpretation,
        }

    def _save_factor_wq_state(self, factor_id: int, updates: dict[str, Any]) -> None:
        db = get_db_session()
        repo = FactorRepository(db)
        factor = repo.get_by_id(factor_id)
        if not factor:
            db.close()
            return

        task_metadata = dict(factor.task_metadata or {})
        task_metadata["wq_brain"] = {
            **(task_metadata.get("wq_brain") or {}),
            **updates,
        }
        factor.task_metadata = task_metadata

        repo.update(factor)
        db.close()

    def _load_factor_submission_context(self, factor_id: int) -> dict[str, Any] | None:
        factors = factor_service.get_all_factors()
        factor = next((item for item in factors if item.get("id") == factor_id), None)
        if not factor:
            return None
        return self._extract_factor_candidate(factor)

    def _simulate_and_maybe_submit(self, client: WQBrainDirectClient, payload: dict[str, Any]) -> dict[str, Any]:
        expression = (payload.get("expression") or "").strip()
        if not expression:
            return {"success": False, "mode": "direct", "message": "缺少表达式，无法提交到 WQ BRAIN。"}

        simulated = client.simulate(
            expression=expression,
            region=(payload.get("region") or "USA").strip() or "USA",
            universe=(payload.get("universe") or "TOP3000").strip() or "TOP3000",
            delay=self._safe_int(payload.get("delay")) or 1,
            decay=self._safe_int(payload.get("decay")) or 0,
            neutralization=(payload.get("neutralization") or "SUBINDUSTRY").strip() or "SUBINDUSTRY",
            truncation=self._safe_float(payload.get("truncation")) or 0.08,
        )
        if not simulated.get("ok"):
            return {
                "success": False,
                "mode": "direct",
                "message": simulated.get("error") or "WQ BRAIN 模拟失败",
                "simulation": simulated,
            }

        alpha_id = simulated.get("alpha_id")
        is_data = simulated.get("is") or {}
        summary = {
            "alpha_id": alpha_id,
            "fitness": self._safe_float(is_data.get("fitness")),
            "sharpe": self._safe_float(is_data.get("sharpe")),
            "returns": self._safe_float(is_data.get("returns")),
            "turnover": self._safe_float(is_data.get("turnover")),
        }

        submitted = False
        submission = None
        if payload.get("auto_submit", True) and alpha_id:
            submission = client.submit_alpha(alpha_id)
            submitted = bool(submission.get("ok"))

        return {
            "success": True,
            "mode": "direct",
            "alpha_id": alpha_id,
            "simulated": simulated,
            "submission": submission,
            "submitted": submitted,
            "summary": summary,
        }

    def _normalize_post_submit_status(self, checked: dict[str, Any] | None, submitted: bool) -> str | None:
        if not checked:
            return "SUBMITTED" if submitted else None
        final_status = checked.get("final_status")
        if submitted and final_status == "UNSUBMITTED":
            return "SC_PENDING"
        return final_status

    async def get_config(self) -> dict:
        default_account = (os.getenv("WQ_BRAIN_DEFAULT_ACCOUNT") or settings.WQ_BRAIN_DEFAULT_ACCOUNT or "primary").strip() or "primary"
        primary_email = (os.getenv("WQ_BRAIN_EMAIL") or settings.WQ_BRAIN_EMAIL or "").strip()
        alt_email = (os.getenv("WQ_BRAIN_ALT_EMAIL") or settings.WQ_BRAIN_ALT_EMAIL or "").strip()
        primary_password = (os.getenv("WQ_BRAIN_PASSWORD") or settings.WQ_BRAIN_PASSWORD or "").strip()
        alt_password = (os.getenv("WQ_BRAIN_ALT_PASSWORD") or settings.WQ_BRAIN_ALT_PASSWORD or "").strip()
        return {
            "success": True,
            "default_account": default_account,
            "primary_email": primary_email,
            "alt_email": alt_email,
            "has_primary_password": bool(primary_password),
            "has_alt_password": bool(alt_password),
        }

    async def update_config(self, payload: dict) -> dict:
        default_account = (payload.get("default_account") or "primary").strip() or "primary"
        primary_email = (payload.get("primary_email") or "").strip()
        primary_password = payload.get("primary_password")
        alt_email = (payload.get("alt_email") or "").strip()
        alt_password = payload.get("alt_password")

        updates = {
            "WQ_BRAIN_DEFAULT_ACCOUNT": default_account,
            "WQ_BRAIN_EMAIL": primary_email,
            "WQ_BRAIN_ALT_EMAIL": alt_email,
        }
        if primary_password is not None:
            updates["WQ_BRAIN_PASSWORD"] = primary_password.strip()
        if alt_password is not None:
            updates["WQ_BRAIN_ALT_PASSWORD"] = alt_password.strip()

        self._write_env_values(updates)
        for key, value in updates.items():
            os.environ[key] = value

        return {
            "success": True,
            "message": "WQ BRAIN 配置已保存",
            **(await self.get_config()),
        }

    async def status(self, account: str = "primary") -> dict:
        accounts = direct_configured_accounts()
        configured = direct_is_configured(account)
        connected = False
        if configured:
            client = WQBrainDirectClient.for_account(account)
            try:
                connected = client.authenticate()
            finally:
                client.close()
        return {
            "success": True,
            "connected": connected,
            "account": account,
            "mode": "direct",
            "status": {
                "configured": configured,
                "accounts": accounts,
                "thresholds": SUBMIT_THRESHOLDS,
            },
        }

    async def user_info(self, account: str = "primary") -> dict:
        if not direct_is_configured(account):
            return {
                "success": True,
                "account": account,
                "configured": False,
                "user": {},
                "message": f"WQ BRAIN 账号未配置 (account={account})",
                "mode": "direct",
            }
        client = WQBrainDirectClient.for_account(account)
        try:
            if not client.authenticate():
                raise RuntimeError("WQ BRAIN 认证失败")
            raw = client.get_user_info()
        finally:
            client.close()
        return {
            "success": True,
            "account": account,
            "user": raw,
            "mode": "direct",
        }

    async def platform_alphas(self, account: str = "primary", limit: int = 50) -> dict:
        if self._has_client_override():
            client = self._client()
            try:
                raw = await client.wq_platform_alphas(account, limit)
                return {
                    "success": True,
                    "account": account,
                    "configured": True,
                    "alphas": raw.get("alphas", []),
                    "raw": raw,
                    "mode": "proxy",
                }
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    status = await client.wq_status(account)
                    configured = bool(status.get("configured"))
                    if not configured:
                        return {
                            "success": True,
                            "account": account,
                            "configured": False,
                            "alphas": [],
                            "message": "WQ BRAIN 账号未配置，平台 Alpha 列表不可用。",
                            "mode": "proxy",
                        }
                raise

        if not direct_is_configured(account):
            return {
                "success": True,
                "account": account,
                "configured": False,
                "alphas": [],
                "message": "WQ BRAIN 账号未配置，平台 Alpha 列表不可用。",
                "mode": "direct",
            }
        client = WQBrainDirectClient.for_account(account)
        try:
            if not client.authenticate():
                raise RuntimeError("WQ BRAIN 认证失败")
            raw = client.list_platform_alphas(limit=limit)
        finally:
            client.close()
        return {
            "success": True,
            "account": account,
            "alphas": raw.get("alphas", []),
            "raw": raw,
            "mode": "direct",
        }

    async def candidates(self, limit: int = 100) -> dict:
        factors = factor_service.get_all_factors()
        candidates: list[dict[str, Any]] = []
        for factor in factors:
            candidate = self._extract_factor_candidate(factor)
            if candidate:
                candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item.get("submission_status") in ("ACTIVE", "SC_PENDING"),
                item.get("score") is not None,
                item.get("score") or float("-inf"),
                item.get("ls_sharpe") or float("-inf"),
            ),
            reverse=True,
        )

        return {
            "success": True,
            "total": len(candidates),
            "candidates": candidates[: max(limit, 1)],
        }

    async def submit(self, payload: dict) -> dict:
        account = payload.get("account", "primary")
        if account != "primary":
            return {
                "success": False,
                "mode": "direct",
                "message": "Alpha 提交仅允许 primary 账号。",
            }
        if not direct_is_configured(account):
            return {
                "success": False,
                "mode": "direct",
                "message": f"WQ BRAIN 未配置 (account={account})",
            }
        client = WQBrainDirectClient.for_account(account)
        try:
            if not client.authenticate():
                return {"success": False, "mode": "direct", "message": "WQ BRAIN 认证失败"}
            alpha_id = (payload.get("alpha_id") or "").strip()
            factor_id = self._safe_int(payload.get("factor_id"))
            if alpha_id:
                raw = client.submit_alpha(alpha_id)
                result = {"success": raw.get("ok", False), "submission": raw, "mode": "direct"}
                if factor_id:
                    checked = self._classify_alpha_check(client.check_alpha_status(alpha_id))
                    self._save_factor_wq_state(
                        factor_id,
                        {
                            "alpha_id": alpha_id,
                            "submission_status": self._normalize_post_submit_status(checked, bool(raw.get("ok"))) or ("SUBMITTED" if raw.get("ok") else "ERROR"),
                            "platform_status": checked.get("status"),
                            "sc_result": checked.get("sc_result"),
                            "sc_value": checked.get("sc_value"),
                            "sc_limit": checked.get("sc_limit"),
                            "wq_fitness": checked.get("fitness"),
                            "wq_sharpe": checked.get("sharpe"),
                        },
                    )
                return result

            if not factor_id:
                return {
                    "success": False,
                    "mode": "direct",
                    "message": "当前仅支持从因子库候选因子提交到 WQ BRAIN。",
                }

            factor_context = self._load_factor_submission_context(factor_id)
            if not factor_context:
                return {
                    "success": False,
                    "mode": "direct",
                    "message": f"因子库中未找到可提交的候选因子（factor_id={factor_id}）。",
                }

            result = self._simulate_and_maybe_submit(
                client,
                {
                    **payload,
                    "expression": factor_context.get("expression") or "",
                },
            )
            if result.get("success") and factor_id:
                submission = result.get("submission") or {}
                simulated = result.get("simulated") or {}
                summary = result.get("summary") or {}
                checked = None
                if result.get("alpha_id"):
                    checked = self._classify_alpha_check(client.check_alpha_status(result["alpha_id"]))
                self._save_factor_wq_state(
                    factor_id,
                    {
                        "alpha_id": result.get("alpha_id"),
                        "simulation_id": simulated.get("simulation_id"),
                        "submission_status": self._normalize_post_submit_status(checked, bool(result.get("submitted"))) or ("SUBMITTED" if result.get("submitted") else "SIMULATED"),
                        "platform_status": (checked or {}).get("status"),
                        "sc_result": (checked or {}).get("sc_result"),
                        "sc_value": (checked or {}).get("sc_value"),
                        "sc_limit": (checked or {}).get("sc_limit"),
                        "wq_fitness": summary.get("fitness"),
                        "wq_sharpe": summary.get("sharpe"),
                        "wq_returns": summary.get("returns"),
                        "submit_detail": submission.get("detail") if submission else None,
                    },
                )
            return result
        finally:
            client.close()

    async def batch_submit(self, payload: dict) -> dict:
        account = payload.get("account", "primary")
        alpha_ids = payload.get("alpha_ids") or []
        if not alpha_ids:
            expressions = payload.get("expressions") or []
            return {
                "success": False,
                "mode": "direct",
                "message": f"当前直连批量提交通道仅支持 alpha_ids；收到 expressions={len(expressions)}，表达式批量模拟/提交流程尚未迁完。",
            }
        if account != "primary":
            return {"success": False, "mode": "direct", "message": "Alpha 批量提交仅允许 primary 账号。"}
        if not direct_is_configured(account):
            return {"success": False, "mode": "direct", "message": f"WQ BRAIN 未配置 (account={account})"}
        client = WQBrainDirectClient.for_account(account)
        results: dict[str, Any] = {}
        try:
            if not client.authenticate():
                return {"success": False, "mode": "direct", "message": "WQ BRAIN 认证失败"}
            for alpha_id in alpha_ids:
                results[alpha_id] = client.submit_alpha(alpha_id)
        finally:
            client.close()
        summary = {
            "total": len(alpha_ids),
            "submitted": sum(1 for r in results.values() if r.get("ok")),
            "failed": sum(1 for r in results.values() if not r.get("ok")),
        }
        return {"success": True, "mode": "direct", "summary": summary, "alphas": results}

    async def check_alphas(self, payload: dict) -> dict:
        account = payload.get("account", "primary")
        alpha_ids = payload.get("alpha_ids") or []
        if not direct_is_configured(account):
            return {"success": False, "mode": "direct", "message": f"WQ BRAIN 未配置 (account={account})"}
        client = WQBrainDirectClient.for_account(account)
        results: dict[str, Any] = {}
        try:
            if not client.authenticate():
                return {"success": False, "mode": "direct", "message": "WQ BRAIN 认证失败"}
            for alpha_id in alpha_ids:
                results[alpha_id] = self._classify_alpha_check(client.check_alpha_status(alpha_id))
        finally:
            client.close()
        summary = {
            "total": len(alpha_ids),
            "active": sum(1 for r in results.values() if r.get("final_status") == "ACTIVE"),
            "unsubmitted": sum(1 for r in results.values() if r.get("final_status") == "UNSUBMITTED"),
            "sc_fail": sum(1 for r in results.values() if r.get("final_status") == "SC_FAIL"),
            "sc_pending": sum(1 for r in results.values() if r.get("final_status") == "SC_PENDING"),
            "error": sum(1 for r in results.values() if r.get("final_status") == "ERROR"),
        }
        return {"success": True, "mode": "direct", "summary": summary, "alphas": results}

    async def finalize(self, payload: dict) -> dict:
        # Direct finalize is equivalent to re-checking platform-side alpha statuses.
        checked = await self.check_alphas(payload)
        if not checked.get("success"):
            return checked
        checked["finalized"] = True
        return checked

    async def sync_candidates(self, payload: dict) -> dict:
        factor_ids = payload.get("factor_ids") or []
        account = payload.get("account", "primary")
        factors = factor_service.get_all_factors()
        factor_map = {item.get("id"): item for item in factors}

        target_factors = []
        alpha_ids = []
        for factor_id in factor_ids:
            factor = factor_map.get(factor_id)
            if not factor:
                continue
            candidate = self._extract_factor_candidate(factor)
            if not candidate or not candidate.get("alpha_id"):
                continue
            target_factors.append(candidate)
            alpha_ids.append(candidate["alpha_id"])

        if not alpha_ids:
            return {
                "success": True,
                "mode": "direct",
                "summary": {"total": 0, "active": 0, "sc_fail": 0, "sc_pending": 0, "unsubmitted": 0, "error": 0},
                "factors": [],
            }

        checked = await self.check_alphas({"account": account, "alpha_ids": alpha_ids})
        if not checked.get("success"):
            return checked

        checked_map = checked.get("alphas") or {}
        refreshed: list[dict[str, Any]] = []
        for item in target_factors:
            alpha_id = item["alpha_id"]
            status = checked_map.get(alpha_id) or {}
            self._save_factor_wq_state(
                item["factor_id"],
                {
                    "alpha_id": alpha_id,
                    "submission_status": status.get("final_status"),
                    "platform_status": status.get("status"),
                    "sc_result": status.get("sc_result"),
                    "sc_value": status.get("sc_value"),
                    "sc_limit": status.get("sc_limit"),
                    "wq_fitness": status.get("fitness"),
                    "wq_sharpe": status.get("sharpe"),
                    "wq_returns": status.get("returns"),
                },
            )
            refreshed.append({
                "factor_id": item["factor_id"],
                "alpha_id": alpha_id,
                **status,
            })

        return {
            "success": True,
            "mode": "direct",
            "summary": checked.get("summary"),
            "factors": refreshed,
        }


wqbrain_service = WQBrainService()

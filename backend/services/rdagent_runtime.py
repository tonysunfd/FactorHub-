from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException

from backend.core.settings import settings

DEFAULT_RDAGENT_PROJECT_ROOTS = [
    settings.BASE_DIR.parent / "reference" / "RD-Agent",
    Path("/Users/tonysun/Desktop/reference/RD-Agent"),
]

DEFAULT_RDAGENT_SITE_PACKAGES = [
    settings.BASE_DIR / ".venv-rdagent" / "lib" / "python3.10" / "site-packages",
    settings.BASE_DIR / ".venv-rdagent" / "lib" / "python3.11" / "site-packages",
    settings.BASE_DIR / "backend" / "vendor" / "rdagent_site_packages",
]

DEFAULT_RDAGENT_PYTHONS = [
    settings.BASE_DIR / ".venv-rdagent" / "bin" / "python",
    settings.BASE_DIR / ".venv-rdagent" / "bin" / "python3",
]


def _resolve_project_root_candidates() -> list[Path]:
    configured = (os.environ.get("FACTORHUB_RDAGENT_PROJECT_ROOT") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_PROJECT_ROOTS)
    return candidates


def resolve_rdagent_project_root() -> Path:
    for candidate in _resolve_project_root_candidates():
        resolved = candidate.expanduser().resolve()
        if resolved.exists() and resolved.is_dir() and (resolved / "rdagent").exists():
            return resolved

    raise HTTPException(
        status_code=500,
        detail=(
            "未找到 reference RD-Agent 项目。请配置 FACTORHUB_RDAGENT_PROJECT_ROOT，"
            "或确保 /Users/tonysun/Desktop/reference/RD-Agent 存在。"
        ),
    )


def resolve_rdagent_site_packages() -> Path:
    configured = (os.environ.get("FACTORHUB_RDAGENT_SITE_PACKAGES") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_SITE_PACKAGES)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate

    project_root = resolve_rdagent_project_root()
    return project_root


def resolve_rdagent_python() -> Path:
    configured = (os.environ.get("FACTORHUB_RDAGENT_PYTHON") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_PYTHONS)

    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and expanded.is_file():
            return expanded

    current_python = Path(sys.executable)
    if current_python.exists():
        return current_python

    raise HTTPException(
        status_code=500,
        detail=(
            "未找到 RD-Agent Python 运行时。请配置 FACTORHUB_RDAGENT_PYTHON，"
            "或将 RD-Agent 安装到 .venv-rdagent 下。"
        ),
    )


def _probe_import(project_root: Path, python_path: Path) -> tuple[bool, str | None]:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {project_root.as_posix()!r})\n"
        "from rdagent.app.qlib_rd_loop.factor import FactorRDLoop\n"
        "print(FactorRDLoop.__name__)\n"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, None
    return False, (proc.stderr or proc.stdout or "").strip() or "unknown import error"


def probe_rdagent_module_import(module_name: str) -> tuple[bool, str | None]:
    project_root = resolve_rdagent_project_root()
    python_path = resolve_rdagent_python()
    script = (
        "import importlib\n"
        "import sys\n"
        f"sys.path.insert(0, {project_root.as_posix()!r})\n"
        f"importlib.import_module({module_name!r})\n"
        "print('OK')\n"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, None
    return False, (proc.stderr or proc.stdout or "").strip() or "unknown import error"


def get_rdagent_runtime_status() -> dict:
    checked_roots = []
    for candidate in _resolve_project_root_candidates():
        resolved = candidate.expanduser().resolve()
        exists = resolved.exists() and resolved.is_dir() and (resolved / "rdagent").exists()
        checked_roots.append({"path": str(resolved), "exists": exists})

    try:
        project_root = resolve_rdagent_project_root()
    except HTTPException as exc:
        return {
            "available": False,
            "active_path": None,
            "python_path": None,
            "checked_paths": checked_roots,
            "importable": False,
            "import_error": exc.detail,
        }

    python_path = resolve_rdagent_python()
    loop_importable, loop_import_error = _probe_import(project_root, python_path)
    proposal_importable, proposal_import_error = probe_rdagent_module_import(
        "rdagent.scenarios.qlib.proposal.factor_proposal"
    )
    return {
        "available": proposal_importable,
        "active_path": str(project_root),
        "python_path": str(python_path),
        "checked_paths": checked_roots,
        "importable": proposal_importable,
        "import_error": proposal_import_error,
        "proposal_importable": proposal_importable,
        "proposal_import_error": proposal_import_error,
        "loop_importable": loop_importable,
        "loop_import_error": loop_import_error,
    }

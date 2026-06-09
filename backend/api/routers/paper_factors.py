from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, urlencode, quote_plus
import shutil
import subprocess
import urllib.request
import uuid
import hashlib
import sys
import os
import time
import xml.etree.ElementTree as ET

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.factor_service import factor_service
from backend.services.research_tools.llm_config_service import llm_config_service
from backend.core.settings import settings


router = APIRouter()

WEBDAV_PAPER_ROOT = Path(
    (os.environ.get("FACTORHUB_PAPER_FACTOR_ROOT") or str(settings.DATA_DIR / "paper_factors"))
).expanduser()
ALLOWED_SUFFIXES = {".pdf"}
PAPER_FACTOR_LIBRARY_PATH = settings.DATA_DIR / "paper_factor_library.json"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_RDAGENT_SITE_PACKAGES = [
    settings.BASE_DIR / ".venv-rdagent" / "lib" / "python3.10" / "site-packages",
    settings.BASE_DIR / ".venv-rdagent" / "lib" / "python3.11" / "site-packages",
    settings.BASE_DIR / "backend" / "vendor" / "rdagent_site_packages",
]
DEFAULT_RDAGENT_PYTHONS = [
    settings.BASE_DIR / ".venv-rdagent" / "bin" / "python",
    settings.BASE_DIR / ".venv-rdagent" / "bin" / "python3",
]


def _ensure_root() -> Path:
    if WEBDAV_PAPER_ROOT.exists() and not WEBDAV_PAPER_ROOT.is_dir():
        raise HTTPException(status_code=500, detail="论文因子存储目录无效")
    WEBDAV_PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    return WEBDAV_PAPER_ROOT


def _sanitize_filename(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名无效")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")
    return name


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size": stat.st_size,
        "modified_at": int(stat.st_mtime),
        "source_type": "webdav",
    }


def _load_paper_factor_library() -> list[dict]:
    if not PAPER_FACTOR_LIBRARY_PATH.exists():
        return []
    try:
        data = json.loads(PAPER_FACTOR_LIBRARY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_paper_factor_library(entries: list[dict]) -> None:
    PAPER_FACTOR_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FACTOR_LIBRARY_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _upsert_paper_factor_entry(entry: dict) -> dict:
    entries = _load_paper_factor_library()
    for index, existing in enumerate(entries):
        if existing.get("source_hash") == entry.get("source_hash"):
            merged = {
                **existing,
                **entry,
                "id": existing.get("id", entry["id"]),
                "status": existing.get("status", "draft"),
                "linked_factor_id": existing.get("linked_factor_id"),
                "linked_factor_name": existing.get("linked_factor_name"),
                "converted_at": existing.get("converted_at"),
            }
            entries[index] = merged
            _save_paper_factor_library(entries)
            return merged
    entries.append(entry)
    _save_paper_factor_library(entries)
    return entry


def _replace_paper_factor_entry(updated_entry: dict) -> None:
    entries = _load_paper_factor_library()
    replaced = False
    for index, existing in enumerate(entries):
        if existing.get("id") == updated_entry.get("id"):
            entries[index] = updated_entry
            replaced = True
            break
    if not replaced:
        entries.append(updated_entry)
    _save_paper_factor_library(entries)


def _atom_text(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    child = node.find(path, ATOM_NS)
    if child is None or child.text is None:
        return default
    return str(child.text).strip()


def _parse_atom_entries(blob: bytes) -> list[dict]:
    root = ET.fromstring(blob)
    rows: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _atom_text(entry, "atom:title")
        summary = _atom_text(entry, "atom:summary")
        published = _atom_text(entry, "atom:published")
        updated = _atom_text(entry, "atom:updated")
        entry_id = _atom_text(entry, "atom:id")
        authors = [
            str(name.text).strip()
            for name in entry.findall("atom:author/atom:name", ATOM_NS)
            if name is not None and name.text
        ]
        link = ""
        for link_node in entry.findall("atom:link", ATOM_NS):
            href = str(link_node.attrib.get("href") or "").strip()
            rel = str(link_node.attrib.get("rel") or "").strip().lower()
            if href and (rel in {"alternate", ""}):
                link = href
                break
        rows.append(
            {
                "title": title,
                "abstract": summary,
                "published_at": published or updated,
                "authors": authors,
                "source_url": link or entry_id,
                "external_id": entry_id,
            }
        )
    return rows


class ThirdPartyDownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None


class ExtractPaperFactorsRequest(BaseModel):
    filenames: List[str]
    category: str = "论文因子"


class ConvertPaperFactorsRequest(BaseModel):
    entry_ids: List[str]
    category: str = "论文因子"


class SearchPaperSourcesRequest(BaseModel):
    source: str
    query: str
    limit: int = 10


class RefreshPaperSourcesRequest(BaseModel):
    query: str
    sources: List[str] = ["openalex", "arxiv"]
    limit_per_source: int = 10
    auto_extract: bool = True


class ImportPaperSearchResultsRequest(BaseModel):
    items: List[dict]
    category: str = "论文因子"
    auto_extract: bool = True


def _safe_existing_pdf_path(filename: str) -> Path:
    root = _ensure_root()
    name = Path(filename or "").name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名无效")
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="文件路径非法")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"未找到 PDF 文件: {name}")
    if path.name.startswith("._"):
        raise HTTPException(status_code=400, detail=f"无效的 macOS 资源文件: {name}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"文件不是 PDF: {name}")
    return path


def _download_pdf_to_root(url: str, filename: Optional[str] = None, source_type: str = "third_party_download") -> dict:
    root = _ensure_root()
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="仅支持 http/https 下载地址")

    inferred_name = filename or Path(parsed.path).name or f"paper_{uuid.uuid4().hex[:8]}.pdf"
    if not inferred_name.lower().endswith(".pdf"):
        inferred_name = f"{inferred_name}.pdf"
    safe_name = _sanitize_filename(inferred_name)
    destination = root / safe_name
    if destination.exists():
        destination = root / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'FactorHub paper-import/1.0'})
        with urllib.request.urlopen(req, timeout=90) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"下载 PDF 失败: {exc}") from exc

    record = _file_record(destination)
    record["source_type"] = source_type
    record["source_url"] = url
    return record


def _safe_download_search_result(item: dict) -> tuple[dict | None, dict | None]:
    pdf_url = str(item.get("pdf_url") or "").strip()
    if not pdf_url:
        return None, {
            "title": str(item.get("title") or "").strip() or "paper",
            "filename": None,
            "reason": "搜索结果没有可下载的 PDF",
        }

    title = str(item.get("title") or "paper").strip() or "paper"
    safe_title = _slugify_name(title).replace("-", "_")
    filename = f"{safe_title[:80]}.pdf"
    existing = _ensure_root() / filename
    if existing.exists():
        return None, {
            "title": title,
            "filename": filename,
            "reason": "PDF 已存在于 WebDAV 文献目录",
        }

    try:
        record = _download_pdf_to_root(pdf_url, filename, f"paper_search_{item.get('source', 'external')}")
        return record, None
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return None, {
            "title": title,
            "filename": filename,
            "reason": detail,
        }


def _slugify_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (name or "").strip())
    sanitized = sanitized.strip("_") or "paper_factor"
    return sanitized[:80]


def _build_factor_code_from_formulation(formulation: str, variables: dict) -> str:
    variables_block = "\n".join(f"# {key}: {value}" for key, value in (variables or {}).items())
    formula_comment = f"# RD-Agent extracted LaTeX formulation\n# {formulation}".strip()
    if variables_block:
        formula_comment = f"{formula_comment}\n# Variables\n{variables_block}"
    return (
        "def calculate_factor(df):\n"
        f"    \"\"\"{formula_comment}\"\"\"\n"
        "    # 待进一步实现为可执行表达式；当前先保留论文来源与原始公式。\n"
        "    return df['close'] * 0 + 0\n"
    )


def _sync_rdagent_llm_env() -> None:
    config = llm_config_service._read_env_values()
    api_key = (config.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
    base_url = (config.get("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "").strip()
    model = (config.get("DEEPSEEK_MODEL") or os.getenv("DEEPSEEK_MODEL") or "").strip()
    embedding_model = (os.getenv("FACTORHUB_RDAGENT_EMBEDDING_MODEL") or "openai/text-embedding-3-small").strip()
    normalized_model = model
    if normalized_model and "/" not in normalized_model:
        normalized_model = f"openai/{normalized_model}"
    chat_temperature = "0.5"
    if normalized_model.startswith("openai/gpt-5"):
        chat_temperature = "1"
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["LITELLM_OPENAI_API_KEY"] = api_key
        os.environ["CHAT_OPENAI_API_KEY"] = api_key
        os.environ["EMBEDDING_OPENAI_API_KEY"] = api_key
        os.environ["LITELLM_CHAT_OPENAI_API_KEY"] = api_key
        os.environ["LITELLM_EMBEDDING_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["CHAT_OPENAI_BASE_URL"] = base_url
        os.environ["EMBEDDING_OPENAI_BASE_URL"] = base_url
        os.environ["LITELLM_CHAT_OPENAI_BASE_URL"] = base_url
        os.environ["LITELLM_EMBEDDING_OPENAI_BASE_URL"] = base_url
    if normalized_model:
        os.environ["CHAT_MODEL"] = normalized_model
        os.environ["LITELLM_CHAT_MODEL"] = normalized_model
        os.environ["CHAT_TEMPERATURE"] = chat_temperature
        os.environ["LITELLM_CHAT_TEMPERATURE"] = chat_temperature
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
        os.environ["LITELLM_EMBEDDING_MODEL"] = embedding_model


def _resolve_rdagent_site_packages() -> Path:
    configured = (os.environ.get("FACTORHUB_RDAGENT_SITE_PACKAGES") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_SITE_PACKAGES)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise HTTPException(
        status_code=500,
        detail=(
            "未找到 RD-Agent 运行时。请在 FactorHub 环境中配置 "
            "FACTORHUB_RDAGENT_SITE_PACKAGES，或将 RD-Agent 安装到 "
            "backend/vendor/rdagent_site_packages / .venv-rdagent 下。"
        ),
    )


def _resolve_rdagent_python() -> Path:
    configured = (os.environ.get("FACTORHUB_RDAGENT_PYTHON") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_PYTHONS)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    raise HTTPException(
        status_code=500,
        detail=(
            "未找到 RD-Agent Python 运行时。请在 FactorHub 环境中配置 "
            "FACTORHUB_RDAGENT_PYTHON，或将 RD-Agent 安装到 "
            ".venv-rdagent 下。"
        ),
    )


def _get_rdagent_runtime_status() -> dict:
    configured = (os.environ.get("FACTORHUB_RDAGENT_SITE_PACKAGES") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_RDAGENT_SITE_PACKAGES)

    checked = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        exists = resolved.exists() and resolved.is_dir()
        checked.append({"path": str(resolved), "exists": exists})
        if exists:
            python_path = None
            try:
                python_path = str(_resolve_rdagent_python())
            except HTTPException:
                python_path = None
            return {
                "available": True,
                "active_path": str(resolved),
                "python_path": python_path,
                "checked_paths": checked,
            }

    return {
        "available": False,
        "active_path": None,
        "python_path": None,
        "checked_paths": checked,
    }


def _convert_formulation_to_expression_with_llm(formulation: str, variables: dict) -> Optional[str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        from backend.engines.factor_engine import _get_client, _get_model
    except Exception:
        return None

    prompt = (
        "你是一个量化因子公式转换助手。\n"
        "请把下面的论文 LaTeX 因子公式，转换成 FactorHub 可执行的表达式或 def calculate_factor(df) 函数。\n"
        "要求：\n"
        "1. 如果可以直接写成表达式，就只返回表达式。\n"
        "2. 如果需要多步逻辑，就返回 def calculate_factor(df): ...\n"
        "3. 只能使用 FactorHub 常见字段：open/high/low/close/volume/amount，以及 pandas/NumPy 常见滚动写法。\n"
        "4. 不要输出解释，不要加代码块。\n"
        f"论文公式：{formulation}\n"
        f"变量说明：{variables}\n"
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": "你只返回可执行因子代码。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
            timeout=60,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("python"):
                    content = content[6:]
        return content.strip() or None
    except Exception:
        return None


def _build_paper_factor_code(formulation: str, variables: dict) -> tuple[Optional[str], Optional[str], bool, str]:
    converted = _convert_formulation_to_expression_with_llm(formulation, variables)
    if converted:
        formula_type = "function" if converted.lstrip().startswith("def ") else "expression"
        return converted, formula_type, True, ""
    return None, None, False, "论文公式暂时无法自动转换为可执行因子代码"


def _extract_with_rdagent(pdf_path: Path) -> dict:
    _sync_rdagent_llm_env()
    try:
        rdagent_python = _resolve_rdagent_python()
        rdagent_site_packages = _resolve_rdagent_site_packages()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载 RD-Agent 失败: {exc}") from exc

    script = """
import json
import os
import sys

sys.path.insert(0, os.environ["FACTORHUB_RDAGENT_SITE_PACKAGES"])
from rdagent.app.qlib_rd_loop.factor_from_report import extract_hypothesis_and_exp_from_reports

pdf_path = os.environ["FACTORHUB_PAPER_PDF_PATH"]
exp = extract_hypothesis_and_exp_from_reports(pdf_path)

if exp is None or not getattr(exp, "sub_tasks", None):
    print(json.dumps({"hypothesis": None, "factors": []}, ensure_ascii=False))
    raise SystemExit(0)

hypothesis = getattr(exp, "hypothesis", None)
factor_items = []
for task in exp.sub_tasks:
    factor_items.append(
        {
            "name": getattr(task, "factor_name", ""),
            "description": getattr(task, "factor_description", ""),
            "formulation": getattr(task, "factor_formulation", ""),
            "variables": getattr(task, "variables", {}) or {},
        }
    )

print(json.dumps({
    "hypothesis": {
        "hypothesis": getattr(hypothesis, "hypothesis", "") if hypothesis else "",
        "reason": getattr(hypothesis, "reason", "") if hypothesis else "",
        "concise_reason": getattr(hypothesis, "concise_reason", "") if hypothesis else "",
        "concise_observation": getattr(hypothesis, "concise_observation", "") if hypothesis else "",
        "concise_justification": getattr(hypothesis, "concise_justification", "") if hypothesis else "",
        "concise_knowledge": getattr(hypothesis, "concise_knowledge", "") if hypothesis else "",
    },
    "factors": factor_items,
}, ensure_ascii=False))
"""

    env = os.environ.copy()
    env["FACTORHUB_RDAGENT_SITE_PACKAGES"] = str(rdagent_site_packages)
    env["FACTORHUB_PAPER_PDF_PATH"] = str(pdf_path)

    try:
        proc = subprocess.run(
            [str(rdagent_python), "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RD-Agent 抽取失败: {exc}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=500, detail=f"RD-Agent 抽取失败: {stderr or '未知错误'}")

    try:
        return json.loads((proc.stdout or "").strip() or "{}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RD-Agent 返回结果解析失败: {exc}") from exc


def _search_openalex(query: str, limit: int) -> list[dict]:
    params = urlencode({
        "search": query,
        "per-page": max(1, min(limit, 20)),
    })
    url = f"https://api.openalex.org/works?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "FactorHub paper-search/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = []
    for item in payload.get("results", []):
        primary_location = item.get("primary_location") or {}
        pdf_url = ((primary_location.get("pdf_url") or "").strip() or None)
        landing_url = ((primary_location.get("landing_page_url") or "").strip() or None)
        results.append({
            "source": "openalex",
            "id": item.get("id"),
            "title": item.get("title"),
            "abstract": None,
            "authors": [author.get("author", {}).get("display_name") for author in item.get("authorships", []) if author.get("author", {}).get("display_name")],
            "published_at": item.get("publication_date"),
            "pdf_url": pdf_url,
            "landing_url": landing_url,
            "external_id": item.get("doi") or item.get("id"),
            "can_download_pdf": bool(pdf_url),
        })
    return results


def _search_arxiv(query: str, limit: int) -> list[dict]:
    encoded = quote_plus(query)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}&start=0&max_results={max(1, min(limit, 20))}"
        "&sortBy=relevance&sortOrder=descending"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "FactorHub paper-search/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        blob = response.read()
    rows = _parse_atom_entries(blob)
    results = []
    for row in rows:
        source_url = row.get("source_url") or row.get("external_id") or ""
        pdf_url = None
        if "arxiv.org/abs/" in source_url:
            pdf_url = source_url.replace("/abs/", "/pdf/") + ".pdf"
        results.append({
            "source": "arxiv",
            "id": row.get("external_id"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "authors": row.get("authors") or [],
            "published_at": row.get("published_at"),
            "pdf_url": pdf_url,
            "landing_url": source_url,
            "external_id": row.get("external_id"),
            "can_download_pdf": bool(pdf_url),
        })
    return results


def _search_source(source: str, query: str, limit: int) -> list[dict]:
    normalized = (source or "").strip().lower()
    if normalized == "openalex":
        return _search_openalex(query, limit)
    if normalized == "arxiv":
        return _search_arxiv(query, limit)
    raise HTTPException(status_code=400, detail=f"暂不支持论文源: {source}")


@router.get("/storage")
async def get_paper_storage_info():
    root = _ensure_root()
    return {
        "success": True,
        "data": {
            "root_path": str(root),
            "exists": True,
        },
    }


@router.get("/runtime-status")
async def get_paper_runtime_status():
    return {
        "success": True,
        "data": _get_rdagent_runtime_status(),
    }


@router.get("/files")
async def list_paper_files():
    root = _ensure_root()
    files = sorted(
        [
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES and not path.name.startswith("._")
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return {
        "success": True,
        "data": [_file_record(path) for path in files],
        "total": len(files),
    }


@router.get("/library")
async def list_paper_factor_library():
    entries = _load_paper_factor_library()
    entries.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return {
        "success": True,
        "data": entries,
        "total": len(entries),
    }


@router.post("/search")
async def search_paper_sources(request: SearchPaperSourcesRequest):
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="请输入搜索关键词")

    source = (request.source or "").strip().lower()
    limit = max(1, min(int(request.limit or 10), 20))
    try:
        results = _search_source(source, query, limit)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"论文源搜索失败: {exc}") from exc

    return {
        "success": True,
        "data": results,
        "total": len(results),
    }


@router.post("/refresh")
async def refresh_paper_sources(request: RefreshPaperSourcesRequest):
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="请输入自动更新关键词")

    sources = [str(source).strip().lower() for source in request.sources if str(source).strip()]
    if not sources:
        raise HTTPException(status_code=400, detail="请至少选择一个论文源")

    merged_results: list[dict] = []
    for source in sources:
        merged_results.extend(_search_source(source, query, max(1, min(request.limit_per_source or 10, 20))))

    deduped: list[dict] = []
    seen: set[str] = set()
    for item in merged_results:
        key = str(item.get("external_id") or item.get("id") or item.get("landing_url") or item.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    import_request = ImportPaperSearchResultsRequest(
        items=deduped,
        category="论文因子",
        auto_extract=bool(request.auto_extract),
    )
    import_response = await import_paper_search_results(import_request)
    import_response["data"]["sources"] = sources
    import_response["data"]["query"] = query
    import_response["data"]["matched_results"] = len(deduped)
    import_response["data"]["matched_items"] = deduped
    return import_response


@router.post("/upload")
async def upload_paper_file(file: UploadFile = File(...), source_type: str = Form("upload")):
    root = _ensure_root()
    filename = _sanitize_filename(file.filename or "")
    destination = root / filename
    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix
        destination = root / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    record = _file_record(destination)
    record["source_type"] = source_type or "upload"
    return {
        "success": True,
        "data": record,
        "message": "PDF 已保存到 WebDAV 文献目录",
    }


@router.post("/download")
async def download_third_party_paper(request: ThirdPartyDownloadRequest):
    record = _download_pdf_to_root(request.url, request.filename, "third_party_download")
    return {
        "success": True,
        "data": record,
        "message": "第三方 PDF 已保存到 WebDAV 文献目录",
    }


@router.post("/extract")
async def extract_paper_factors(request: ExtractPaperFactorsRequest):
    if not request.filenames:
        raise HTTPException(status_code=400, detail="请至少选择一个 PDF 文件")

    saved_entries = []
    extraction_results = []

    for filename in request.filenames:
        pdf_path = _safe_existing_pdf_path(filename)
        extraction = _extract_with_rdagent(pdf_path)

        extraction_results.append(
            {
                "file": pdf_path.name,
                "hypothesis": extraction["hypothesis"],
                "factor_count": len(extraction["factors"]),
            }
        )

        for item in extraction["factors"]:
            source_hash = hashlib.md5(f"{pdf_path.name}:{item['name']}:{item['formulation']}".encode("utf-8")).hexdigest()[:8]
            factor_name = f"paper_{_slugify_name(item['name'])}_{source_hash}"
            entry = _upsert_paper_factor_entry(
                {
                    "id": f"paper-{source_hash}",
                    "source_hash": source_hash,
                    "name": factor_name,
                    "display_name": item["name"],
                    "description": item["description"] or f"来源论文：{pdf_path.stem}",
                    "category": request.category or "论文因子",
                    "paper_title": pdf_path.stem,
                    "paper_file": pdf_path.name,
                    "paper_path": str(pdf_path),
                    "source_type": "paper_pdf",
                    "formulation": item["formulation"],
                    "variables": item["variables"],
                    "hypothesis": extraction["hypothesis"],
                    "status": "draft",
                    "created_at": int(time.time()),
                }
            )
            saved_entries.append(
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "display_name": entry["display_name"],
                    "category": entry["category"],
                    "paper_file": pdf_path.name,
                    "description": item["description"],
                    "formulation": item["formulation"],
                    "variables": item["variables"],
                    "status": "draft",
                }
            )

    if not saved_entries:
        raise HTTPException(status_code=400, detail="未从所选 PDF 中抽取到论文因子")

    return {
        "success": True,
        "data": {
            "saved_entries": saved_entries,
            "extractions": extraction_results,
        },
        "message": f"已录入 {len(saved_entries)} 个论文因子草稿，请二次确认后再转为 FactorHub 因子",
    }


@router.post("/convert")
async def convert_paper_factors(request: ConvertPaperFactorsRequest):
    if not request.entry_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个论文因子草稿")

    entries = _load_paper_factor_library()
    entry_map = {entry.get("id"): entry for entry in entries}
    converted_factors = []
    failed_entries = []

    for entry_id in request.entry_ids:
        entry = entry_map.get(entry_id)
        if not entry:
            failed_entries.append({"id": entry_id, "reason": "未找到论文因子草稿"})
            continue

        factor_code, formula_type, converted, failure_reason = _build_paper_factor_code(
            entry.get("formulation", ""),
            entry.get("variables", {}) or {},
        )
        if not converted or not factor_code or not formula_type:
            failed_entries.append(
                {
                    "id": entry_id,
                    "name": entry.get("name"),
                    "reason": failure_reason,
                }
            )
            continue

        task_metadata = {
            "source": "rdagent_fin_factor_report",
            "paper_title": entry.get("paper_title"),
            "paper_file": entry.get("paper_file"),
            "paper_path": entry.get("paper_path"),
            "source_type": entry.get("source_type", "paper_pdf"),
            "source_expression": entry.get("formulation"),
            "paper_factor_status": "converted",
            "converted_code": factor_code,
            "converted_formula_type": formula_type,
            "paper_factor_payload": {
                "description": entry.get("description"),
                "formulation": entry.get("formulation"),
                "variables": entry.get("variables"),
                "hypothesis": entry.get("hypothesis"),
            },
        }
        factor = factor_service.create_factor(
            name=entry.get("name"),
            code=factor_code,
            category=request.category or entry.get("category") or "论文因子",
            description=entry.get("description") or f"来源论文：{entry.get('paper_title', '')}",
            formula_type=formula_type,
            task_metadata=task_metadata,
            scope_type="base",
            origin_type="paper_factor",
        )
        entry["status"] = "converted"
        entry["linked_factor_id"] = factor["id"]
        entry["linked_factor_name"] = factor["name"]
        entry["converted_at"] = int(time.time())
        _replace_paper_factor_entry(entry)

        converted_factors.append(
            {
                "entry_id": entry_id,
                "factor_id": factor["id"],
                "name": factor["name"],
                "formula_type": formula_type,
            }
        )

    if not converted_factors:
        failure_message = failed_entries[0]["reason"] if failed_entries else "没有成功转换任何论文因子"
        raise HTTPException(status_code=400, detail=failure_message)

    return {
        "success": True,
        "data": {
            "converted_factors": converted_factors,
            "failed_entries": failed_entries,
        },
        "message": (
            f"已转换 {len(converted_factors)} 个论文因子"
            if not failed_entries
            else f"已转换 {len(converted_factors)} 个论文因子，{len(failed_entries)} 个草稿未成功转换"
        ),
    }


@router.post("/import-search-results")
async def import_paper_search_results(request: ImportPaperSearchResultsRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="请至少选择一个搜索结果")

    downloaded_files = []
    skipped_items = []
    for item in request.items:
        record, skipped = _safe_download_search_result(item)
        if record:
            downloaded_files.append(record)
        elif skipped:
            skipped_items.append(skipped)

    if not downloaded_files and not skipped_items:
        raise HTTPException(status_code=400, detail="所选搜索结果没有可下载的 PDF")

    response = {
        "success": True,
        "data": {
            "downloaded_files": downloaded_files,
            "skipped_items": skipped_items,
        },
        "message": (
            f"已下载 {len(downloaded_files)} 个 PDF 到 WebDAV 文献目录"
            if not skipped_items
            else f"已下载 {len(downloaded_files)} 个 PDF，跳过 {len(skipped_items)} 个重复文件"
        ),
    }

    if request.auto_extract and downloaded_files:
        extract_request = ExtractPaperFactorsRequest(
            filenames=[item["name"] for item in downloaded_files],
            category=request.category,
        )
        extract_response = await extract_paper_factors(extract_request)
        response["data"]["extract_result"] = extract_response["data"]
        response["message"] = (
            f"已下载 {len(downloaded_files)} 个 PDF，并录入 "
            f"{len(extract_response['data']['saved_entries'])} 个论文因子草稿"
        )
        if skipped_items:
            response["message"] += f"，另跳过 {len(skipped_items)} 个重复文件"

    return response

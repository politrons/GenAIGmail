from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, pipeline

from .mcp_gmail import send_email_via_mcp, sync_gmail_via_mcp
from .rag_retriever import TfidfRagRetriever

try:
    import torch
except Exception:  # pragma: no cover - optional at static analysis time
    torch = None

DEFAULT_PROMPT_ARTIFACT: dict[str, Any] = {
    "instructions": (
        "You are an email intelligence assistant. "
        "Answer only with evidence from retrieved email snippets and knowledge snippets. "
        "If there is not enough evidence, say it explicitly and suggest what query/filter to run next."
    ),
    "default_context": (
        "Focus on concrete email search tasks: status checks, sender/date/subject lookups, and concise operational summaries."
    ),
    "few_shot_demos": [
        {
            "question": "How many messages mention invoice #A-1001 this week?",
            "answer": "State the count from retrieved emails, list the most relevant subjects, and cite sender/date.",
        },
        {
            "question": "Show the latest email from ACME about renewal.",
            "answer": "Return sender, subject, date, and a short evidence snippet from that message.",
        },
    ],
}

DEFAULT_SYSTEM_POLICY = (
    "You are an Email Search Assistant with strict grounding rules.\n"
    "Rules:\n"
    "- Use only retrieved evidence from indexed emails and KB snippets.\n"
    "- Never invent senders, subjects, dates, counts, or timelines.\n"
    "- If question asks for latest/recent emails, focus only on emails that match the topic terms.\n"
    "- If topic evidence is missing, explicitly say there is insufficient evidence and suggest exact next filters.\n"
    "- Prioritize concise operational answers over generic summaries.\n"
    "Output style:\n"
    "- First line: direct answer.\n"
    "- Then: short evidence bullets with sender, subject, and date.\n"
    "- If uncertain: add one actionable next-query suggestion."
)

QUERY_PLANNER_PROMPT = (
    "You are a JSON API for email retrieval planning.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "Required keys and types:\n"
    '{\n'
    '  "intent": "open_qa",\n'
    '  "topic_terms": [],\n'
    '  "must_match_all_terms": true,\n'
    '  "sender_contains": "",\n'
    '  "request_count": false,\n'
    '  "request_latest": false,\n'
    '  "max_results": 6,\n'
    '  "language": "en"\n'
    '}\n'
    "Rules:\n"
    "- intent must be one of: count_latest, latest_list, summary, open_qa.\n"
    "- If user asks count + latest, use intent=count_latest and set both booleans true.\n"
    "- If user asks latest/recent only, use intent=latest_list and request_latest=true.\n"
    "- topic_terms must include only meaningful topic words; use [] when user asks global latest without topic.\n"
    "- If sender is requested, set sender_contains.\n"
    "- max_results must be an integer in [3, 10].\n"
    "- Output must start with '{' and end with '}'.\n"
    "Example valid output:\n"
    '{"intent":"open_qa","topic_terms":[],"must_match_all_terms":true,"sender_contains":"","request_count":false,"request_latest":false,"max_results":6,"language":"en"}\n'
    "- Do not output markdown, comments, options lists, or extra text."
)

CHAT_ACTION_PLANNER_PROMPT = (
    "You are a JSON API for chat action planning.\n"
    "Return exactly one valid JSON object and nothing else.\n"
    "Required keys and types:\n"
    '{\n'
    '  "action": "search",\n'
    '  "recipient_email": "",\n'
    '  "candidate_index": 1,\n'
    '  "email_subject": "",\n'
    '  "email_instruction": ""\n'
    '}\n'
    "Rules:\n"
    "- action must be one of: search, send_last, compose_send, help, none.\n"
    "- Use action=search for normal email lookup/analytics requests.\n"
    "- Use action=send_last to forward/share the previously found email result.\n"
    "- Use action=compose_send to send a new email when user provides recipient and message/body content.\n"
    "- If send/forward requested without recipient_email, use action=help and explain missing recipient.\n"
    "- Commands like 'send email to X', 'mail X', 'forward to X', or 'enviar correo a X' are send requests.\n"
    "- If user includes 'subject ...' or 'with subject ...', map that to email_subject.\n"
    "- If user includes 'body ...' or message content, map that to email_instruction.\n"
    "- If command is explicit compose (to + subject/body), prefer compose_send even when Session has previous search result is false.\n"
    "- If Session has previous search result is false and user asks to forward/share previous result, use action=help.\n"
    "- Never use action=none for clear send or search requests.\n"
    "- candidate_index must be >= 1.\n"
    "- Output must start with '{' and end with '}'.\n"
    "Example valid output:\n"
    '{"action":"search","recipient_email":"","candidate_index":1,"email_subject":"","email_instruction":""}\n'
    "Send examples:\n"
    '- User: send email to politrons@gmail.com with subject "hello_agent" and body "Welcome to the agent world"\n'
    '  Output: {"action":"compose_send","recipient_email":"politrons@gmail.com","candidate_index":1,"email_subject":"hello_agent","email_instruction":"Welcome to the agent world"}\n'
    '- User: enviar correo a politrons@gmail.com con asunto "hola" y cuerpo "mensaje"\n'
    '  Output: {"action":"compose_send","recipient_email":"politrons@gmail.com","candidate_index":1,"email_subject":"hola","email_instruction":"mensaje"}\n'
    '- User: send this email to someone@example.com\n'
    '  Output: {"action":"send_last","recipient_email":"someone@example.com","candidate_index":1,"email_subject":"","email_instruction":""}\n'
    "- Do not output markdown, comments, options lists, or extra text."
)

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

TASK_ALIASES = {
    "text2text-generation": "text-generation",
}

DEFAULT_MCP_SERVER_COMMAND = "npx -y google-workspace-mcp serve"


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return default


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chat-based email assistant mode."""
    _load_env_file(".env")

    parser = argparse.ArgumentParser(description="Chat about Gmail data using a local model and local index")
    parser.add_argument("--context", default="")
    parser.add_argument("--email-chunks", default=os.getenv("GMAIL_CHUNKS_PATH", "data/gmail_chunks.jsonl"))
    parser.add_argument(
        "--knowledge-chunks",
        default=os.getenv("RAG_CHUNKS_PATH", "data/knowledge_base_chunks.jsonl"),
        help="Optional KB chunks JSONL. Use empty value to disable.",
    )
    parser.add_argument(
        "--prompt-artifact",
        default=os.getenv("DSPY_OPTIMIZED_PROMPT_PATH", ""),
        help="Optional path to optimized_prompt.json. If omitted, the tool auto-discovers DSPy artifacts.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=os.getenv("LOCAL_SYSTEM_PROMPT_FILE", "config/email_llm_system_prompt.txt"),
        help="Path to system prompt policy injected before user question.",
    )
    parser.add_argument("--hf-model-id", default=os.getenv("LOCAL_HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--hf-task", default=os.getenv("LOCAL_HF_TASK", "text-generation"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "220")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LOCAL_TEMPERATURE", "0.0")))
    parser.add_argument("--rag-top-k", type=int, default=int(os.getenv("LOCAL_RAG_TOP_K", "6")))
    parser.add_argument("--rag-min-score", type=float, default=float(os.getenv("LOCAL_RAG_MIN_SCORE", "0.01")))
    parser.add_argument("--max-snippet-chars", type=int, default=int(os.getenv("LOCAL_MAX_SNIPPET_CHARS", "420")))
    parser.add_argument("--send-to", default="", help="Optional recipient email to send top result after answering.")
    parser.add_argument("--send-dry-run", action="store_true", help="Do not send. Print what would be sent.")
    parser.add_argument("--mailbox", default=_env_value("GMAIL_MAILBOX", "gmail_mailbox", default="INBOX"))
    parser.add_argument(
        "--search-criterion",
        default=_env_value("GMAIL_SEARCH_CRITERION", "gmail_search_criterion", default="ALL"),
        help='Search expression/query passed to MCP search tool',
    )
    parser.add_argument(
        "--sync-batch-size",
        type=int,
        default=int(_env_value("GMAIL_MAX_EMAILS", "gmail_max_emails", default="250")),
        help="Batch size for each automatic Gmail sync page",
    )
    parser.add_argument(
        "--sync-max-body-chars",
        type=int,
        default=int(_env_value("GMAIL_MAX_BODY_CHARS", "gmail_max_body_chars", default="5000")),
        help="Max email body chars stored during automatic sync",
    )
    parser.add_argument(
        "--mcp-server-command",
        default=_env_value("MCP_SERVER_COMMAND", "mcp_server_command", default=DEFAULT_MCP_SERVER_COMMAND),
        help='Command used to launch MCP server, e.g. \'npx -y google-workspace-mcp serve\'',
    )
    parser.add_argument(
        "--mcp-search-tool",
        default=_env_value("MCP_GMAIL_SEARCH_TOOL", "mcp_gmail_search_tool", default="searchGmail"),
        help="MCP tool name for email search",
    )
    parser.add_argument(
        "--mcp-send-tool",
        default=_env_value("MCP_GMAIL_SEND_TOOL", "mcp_gmail_send_tool", default="sendGmailDraft"),
        help="MCP tool name for email send",
    )
    parser.add_argument(
        "--mcp-account",
        default=_env_value("MCP_GMAIL_ACCOUNT", "mcp_gmail_account", default=""),
        help="Optional account id/name used by MCP Gmail tools that require explicit account parameter",
    )
    parser.add_argument(
        "--mcp-startup-timeout",
        type=int,
        default=int(_env_value("MCP_STARTUP_TIMEOUT", "mcp_startup_timeout", default="20")),
        help="Seconds to wait for MCP server startup and initialize",
    )
    parser.add_argument(
        "--mcp-request-timeout",
        type=int,
        default=int(_env_value("MCP_REQUEST_TIMEOUT", "mcp_request_timeout", default="45")),
        help="Seconds to wait for each MCP request",
    )
    parser.add_argument("--json-output", action="store_true")
    return parser.parse_args()


def _discover_latest_prompt_artifact() -> Path | None:
    candidates: list[Path] = []

    default_path = Path("artifacts/dspy_optimized/optimized_prompt.json")
    if default_path.exists():
        candidates.append(default_path)

    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        for item in artifacts_dir.rglob("optimized_prompt.json"):
            if item.is_file():
                candidates.append(item)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_prompt_artifact_path(raw_path: str | None) -> Path | None:
    candidate = (raw_path or "").strip()
    if candidate:
        path = Path(candidate)
        if path.exists():
            return path
        print(f"[warn] Prompt artifact not found at {candidate}. Falling back to auto-discovery/default prompt.")

    discovered = _discover_latest_prompt_artifact()
    return discovered


def _load_prompt_artifact(path: str | Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    artifact_path = Path(path)
    if not artifact_path.exists():
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    if not isinstance(data, dict):
        return dict(DEFAULT_PROMPT_ARTIFACT), "built-in-default"

    merged = dict(DEFAULT_PROMPT_ARTIFACT)
    merged.update(data)
    return merged, str(artifact_path)


def _load_system_policy(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return DEFAULT_SYSTEM_POLICY

    text = candidate.read_text(encoding="utf-8").strip()
    return text if text else DEFAULT_SYSTEM_POLICY


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path_obj.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _build_retriever_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = [str(Path(args.email_chunks))]
    knowledge_chunks = str(args.knowledge_chunks or "").strip()
    if knowledge_chunks:
        kb_path = Path(knowledge_chunks)
        if kb_path.exists():
            paths.append(str(kb_path))
    return paths


def _sync_email_window(args: argparse.Namespace, *, offset: int, append: bool) -> dict[str, Any]:
    return sync_gmail_via_mcp(
        mcp_server_command=str(args.mcp_server_command),
        search_tool_name=str(args.mcp_search_tool),
        mcp_account=str(args.mcp_account),
        mailbox=str(args.mailbox),
        search_criterion=str(args.search_criterion),
        max_emails=int(args.sync_batch_size),
        max_body_chars=int(args.sync_max_body_chars),
        output_jsonl=str(args.email_chunks),
        offset=offset,
        append=append,
        startup_timeout_s=int(args.mcp_startup_timeout),
        request_timeout_s=int(args.mcp_request_timeout),
    )


def _load_retriever_and_rows(args: argparse.Namespace) -> tuple[TfidfRagRetriever, list[dict[str, Any]]]:
    email_chunks_path = Path(args.email_chunks)
    if not email_chunks_path.exists():
        raise FileNotFoundError(f"Email chunks file not found: {email_chunks_path}")

    rows = _load_rows(email_chunks_path)
    if not rows:
        raise RuntimeError(f"No email rows found in {email_chunks_path}")

    retriever_paths = _build_retriever_paths(args)
    retriever = TfidfRagRetriever.from_jsonl_paths(retriever_paths)
    return retriever, rows


def _email_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sender_counts: Counter[str] = Counter()
    unique_message_ids: set[str] = set()

    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        sender = str(meta.get("from", "")).strip()
        if sender:
            sender_counts[sender] += 1

        message_id = str(meta.get("message_id", "")).strip()
        if message_id:
            unique_message_ids.add(message_id)

    return {
        "indexed_chunks": len(rows),
        "unique_messages": len(unique_message_ids) if unique_message_ids else len(rows),
        "top_senders": sender_counts.most_common(5),
    }


def _parse_email_date(raw: str) -> datetime:
    value = (raw or "").strip()
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _extract_json_object(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _try_parse_json_dict(raw: str) -> dict[str, Any] | None:
    json_blob = _extract_json_object(raw)
    if json_blob is None:
        return None
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _repair_json_with_llm(generator: Any, schema_prompt: str, invalid_output: str, max_new_tokens: int) -> list[str]:
    invalid = str(invalid_output).strip()
    prompts = [
        "\n".join(
            [
                "You are a JSON repair assistant.",
                "Rewrite the invalid output as one valid JSON object that follows the schema and rules.",
                "Return only JSON.",
                "Output must start with '{' and end with '}'.",
                "",
                "Schema and rules:",
                schema_prompt,
                "",
                "Invalid output:",
                invalid,
                "",
                "Fixed JSON:",
            ]
        ),
        "\n".join(
            [
                "Return EXACTLY one minified JSON object.",
                "No prose. No markdown. No assignments with '='. No lists of options.",
                "If a field is unknown, use safe defaults from the schema examples.",
                "Your first character must be '{' and your last character must be '}'.",
                "",
                "Schema and rules:",
                schema_prompt,
                "",
                "Invalid output to fix:",
                invalid,
                "",
                "JSON:",
            ]
        ),
    ]
    out: list[str] = []
    for prompt in prompts:
        out.append(
            _generate_answer(
                generator=generator,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
    return out


def _normalize_query_plan_with_schema_repair(
    parsed: dict[str, Any],
    *,
    generator: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    try:
        return _normalize_llm_plan(parsed)
    except ValueError as exc:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=QUERY_PLANNER_PROMPT,
            invalid_output=f"{json.dumps(parsed, ensure_ascii=True)}\nSchema error: {exc}",
            max_new_tokens=max_new_tokens,
        )
        for repaired in repaired_candidates:
            repaired_parsed = _try_parse_json_dict(repaired)
            if repaired_parsed is None:
                continue
            try:
                return _normalize_llm_plan(repaired_parsed)
            except ValueError:
                continue
        raise RuntimeError(f"LLM planner JSON schema validation failed: {exc}") from exc


def _normalize_chat_action_with_schema_repair(
    parsed: dict[str, Any],
    *,
    generator: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    try:
        return _normalize_chat_action(parsed)
    except ValueError as exc:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=CHAT_ACTION_PLANNER_PROMPT,
            invalid_output=f"{json.dumps(parsed, ensure_ascii=True)}\nSchema error: {exc}",
            max_new_tokens=max_new_tokens,
        )
        for repaired in repaired_candidates:
            repaired_parsed = _try_parse_json_dict(repaired)
            if repaired_parsed is None:
                continue
            try:
                return _normalize_chat_action(repaired_parsed)
            except ValueError:
                continue
        raise RuntimeError(f"LLM chat planner JSON schema validation failed: {exc}") from exc


def _normalize_llm_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a JSON object")

    intent = str(plan.get("intent", "")).strip()
    if intent not in {"count_latest", "latest_list", "summary", "open_qa"}:
        raise ValueError("Invalid or missing plan.intent")

    raw_terms = plan.get("topic_terms")
    if not isinstance(raw_terms, list):
        raise ValueError("Invalid or missing plan.topic_terms")
    terms = [str(item).strip().lower() for item in raw_terms if str(item).strip()]

    must_match_all_raw = plan.get("must_match_all_terms")
    if not isinstance(must_match_all_raw, bool):
        raise ValueError("Invalid or missing plan.must_match_all_terms")
    must_match_all = must_match_all_raw
    sender_contains = str(plan.get("sender_contains", "")).strip().lower()
    request_count_raw = plan.get("request_count")
    if not isinstance(request_count_raw, bool):
        raise ValueError("Invalid or missing plan.request_count")
    request_count = request_count_raw
    request_latest_raw = plan.get("request_latest")
    if not isinstance(request_latest_raw, bool):
        raise ValueError("Invalid or missing plan.request_latest")
    request_latest = request_latest_raw

    max_results_raw = plan.get("max_results")
    try:
        max_results = int(max_results_raw)
    except Exception:
        raise ValueError("Invalid or missing plan.max_results") from None
    if max_results < 3 or max_results > 10:
        raise ValueError("plan.max_results must be in [3,10]")

    language = str(plan.get("language", "")).strip().lower()
    if not language:
        raise ValueError("Invalid or missing plan.language")

    return {
        "intent": intent,
        "topic_terms": terms,
        "must_match_all_terms": must_match_all,
        "sender_contains": sender_contains,
        "request_count": request_count,
        "request_latest": request_latest,
        "max_results": max_results,
        "language": language,
    }


def _plan_query_with_llm(
    question: str,
    context: str,
    system_policy: str,
    generator: Any,
) -> dict[str, Any]:
    """Ask the LLM to build a structured retrieval plan for the current question."""
    plan_prompt = "\n".join(
        [
            QUERY_PLANNER_PROMPT,
            "",
            "System policy:",
            system_policy.strip() or DEFAULT_SYSTEM_POLICY,
            "",
            f"User question: {question}",
            f"Optional context: {context}",
            "",
            "JSON plan:",
        ]
    )
    raw_plan = _generate_answer(
        generator=generator,
        prompt=plan_prompt,
        max_new_tokens=220,
        temperature=0.0,
    )
    parsed = _try_parse_json_dict(raw_plan)
    if parsed is None:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=QUERY_PLANNER_PROMPT,
            invalid_output=raw_plan,
            max_new_tokens=220,
        )
        for repaired in repaired_candidates:
            parsed = _try_parse_json_dict(repaired)
            if parsed is not None:
                break
        if parsed is None:
            preview = str(repaired_candidates[-1]).strip().replace("\n", " ")[:260]
            raise RuntimeError(f"LLM planner did not return valid JSON after repair. Raw output: {preview}")

    return _normalize_query_plan_with_schema_repair(
        parsed,
        generator=generator,
        max_new_tokens=220,
    )


def _unique_sorted_email_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = _metadata_from_row(row)
        key = meta["message_id"] or str(row.get("chunk_id", ""))
        existing = unique.get(key)
        if existing is None:
            unique[key] = row
            continue

        existing_date = _parse_email_date(_metadata_from_row(existing).get("date", ""))
        current_date = _parse_email_date(meta.get("date", ""))
        if current_date > existing_date:
            unique[key] = row

    out = list(unique.values())
    out.sort(key=lambda r: _parse_email_date(_metadata_from_row(r).get("date", "")), reverse=True)
    return out


def _normalize_chat_action(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Chat action must be a JSON object")

    action = str(plan.get("action", "")).strip().lower()
    if action not in {"search", "send_last", "compose_send", "help", "none"}:
        raise ValueError("Invalid or missing action")

    recipient = str(plan.get("recipient_email", "")).strip().lower()
    if recipient and not EMAIL_REGEX.fullmatch(recipient):
        raise ValueError("recipient_email must be a valid email address")

    candidate_raw = plan.get("candidate_index", 1)
    try:
        candidate_index = int(candidate_raw)
    except Exception:
        raise ValueError("candidate_index must be an integer") from None
    if candidate_index < 1:
        raise ValueError("candidate_index must be >= 1")

    email_subject = str(plan.get("email_subject", "")).strip()
    email_instruction = str(plan.get("email_instruction", "")).strip()

    if action == "send_last" and not recipient:
        raise ValueError("action=send_last requires recipient_email")
    if action == "compose_send" and not recipient:
        raise ValueError("action=compose_send requires recipient_email")
    if action == "compose_send" and not email_instruction:
        raise ValueError("action=compose_send requires email_instruction")

    return {
        "action": action,
        "recipient_email": recipient,
        "candidate_index": candidate_index,
        "email_subject": email_subject,
        "email_instruction": email_instruction,
    }


def _plan_chat_action_with_llm(
    user_message: str,
    system_policy: str,
    session_has_last_result: bool,
    last_result_preview: str,
    generator: Any,
) -> dict[str, Any]:
    """Classify a chat turn into an assistant action using LLM JSON planning."""
    prompt = "\n".join(
        [
            CHAT_ACTION_PLANNER_PROMPT,
            "",
            "System policy:",
            system_policy.strip() or DEFAULT_SYSTEM_POLICY,
            "",
            f"Session has previous search result: {session_has_last_result}",
            f"Previous result preview: {last_result_preview}",
            "",
            f"User message: {user_message}",
            "",
            "JSON action:",
        ]
    )
    raw = _generate_answer(
        generator=generator,
        prompt=prompt,
        max_new_tokens=180,
        temperature=0.0,
    )
    parsed = _try_parse_json_dict(raw)
    if parsed is None:
        repaired_candidates = _repair_json_with_llm(
            generator=generator,
            schema_prompt=CHAT_ACTION_PLANNER_PROMPT,
            invalid_output=raw,
            max_new_tokens=180,
        )
        for repaired in repaired_candidates:
            parsed = _try_parse_json_dict(repaired)
            if parsed is not None:
                break
        if parsed is None:
            preview = str(repaired_candidates[-1]).strip().replace("\n", " ")[:260]
            raise RuntimeError(f"LLM chat planner did not return valid JSON after repair. Raw output: {preview}")
    return _normalize_chat_action_with_schema_repair(
        parsed,
        generator=generator,
        max_new_tokens=180,
    )


def _build_email_message_text(
    question: str,
    answer: str,
    candidate: dict[str, Any],
    evidence: list[dict[str, Any]],
    instruction: str = "",
) -> str:
    lines = [
        "Generated by Local Gmail Email Assistant",
        "",
        f"Question: {question}",
        "",
        "Answer:",
        answer,
        "",
        "Forwarded email content:",
        f"- From: {candidate.get('from', '')}",
        f"- Subject: {candidate.get('subject', '')}",
        f"- Date: {candidate.get('date', '')}",
        "",
        str(candidate.get("text", "")).strip(),
    ]
    if instruction:
        lines.extend(["", "Instruction:", instruction])

    if evidence:
        lines.extend(["", "Evidence summary:"])
        for idx, item in enumerate(evidence[:6], start=1):
            lines.append(
                f"{idx}. from={item.get('from', '')} | date={item.get('date', '')} | subject={item.get('subject', '')}"
            )
    return "\n".join(lines).strip()


def _send_email_via_transport(
    *,
    args: argparse.Namespace,
    send_to: str,
    subject: str,
    body: str,
) -> str:
    return send_email_via_mcp(
        mcp_server_command=str(args.mcp_server_command),
        send_tool_name=str(args.mcp_send_tool),
        mcp_account=str(args.mcp_account),
        send_to=send_to,
        subject=subject,
        body=body,
        dry_run=bool(args.send_dry_run),
        startup_timeout_s=int(args.mcp_startup_timeout),
        request_timeout_s=int(args.mcp_request_timeout),
    )


def _print_result(result: dict[str, Any]) -> None:
    print("Answer:")
    print(result.get("answer", ""))
    print("Evidence:")
    evidence = result.get("evidence", [])
    if isinstance(evidence, list):
        for idx, item in enumerate(evidence, start=1):
            print(
                f"{idx}. score={float(item.get('score', 0.0)):.4f} | "
                f"from={item.get('from', '')} | date={item.get('date', '')} | subject={item.get('subject', '')}"
            )
    print("")


def _send_from_result(
    *,
    result: dict[str, Any],
    recipient: str,
    candidate_index: int,
    args: argparse.Namespace,
    instruction: str = "",
    subject_override: str = "",
) -> str:
    """Send one candidate email from the last search result to a recipient."""
    send_candidates = result.get("send_candidates", [])
    if not isinstance(send_candidates, list) or not send_candidates:
        raise RuntimeError("No sendable email candidate found in the last result.")

    idx = max(1, candidate_index) - 1
    if idx >= len(send_candidates):
        idx = 0
    candidate = send_candidates[idx]

    subject = (
        subject_override.strip()
        if subject_override and subject_override.strip()
        else f"FWD: {str(candidate.get('subject', '(no subject)')).strip()}"
    )
    body = _build_email_message_text(
        question=str(result.get("question", "")),
        answer=str(result.get("answer", "")),
        candidate=candidate,
        evidence=result.get("evidence", []) if isinstance(result.get("evidence", []), list) else [],
        instruction=instruction,
    )
    return _send_email_via_transport(
        args=args,
        send_to=recipient,
        subject=subject,
        body=body,
    )


def _last_result_preview(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    answer = str(result.get("answer", "")).strip()
    evidence = result.get("evidence", [])
    if isinstance(evidence, list) and evidence:
        top = evidence[0] if isinstance(evidence[0], dict) else {}
        return (
            f"answer={answer[:120]} | "
            f"top_subject={top.get('subject', '')} | top_from={top.get('from', '')} | top_date={top.get('date', '')}"
        )
    return f"answer={answer[:120]}"


def _metadata_from_row(row: dict[str, Any]) -> dict[str, str]:
    raw_meta = row.get("metadata")
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    return {
        "subject": str(raw_meta.get("subject", "")).strip(),
        "from": str(raw_meta.get("from", "")).strip(),
        "date": str(raw_meta.get("date", "")).strip(),
        "message_id": str(raw_meta.get("message_id", "")).strip(),
        "snippet": str(raw_meta.get("snippet", "")).strip(),
    }


def _format_evidence(chunk: dict[str, Any], max_snippet_chars: int) -> str:
    meta = chunk.get("metadata") if isinstance(chunk, dict) else None
    metadata = meta if isinstance(meta, dict) else {}

    sender = str(metadata.get("from", "")).strip() or "unknown-sender"
    subject = str(metadata.get("subject", "")).strip() or "(no subject)"
    date = str(metadata.get("date", "")).strip() or "unknown-date"
    source = str(chunk.get("source", "email"))

    snippet = str(metadata.get("snippet", "")).strip()
    text = snippet if snippet else str(chunk.get("text", "")).strip()
    if len(text) > max_snippet_chars:
        text = text[:max_snippet_chars].rstrip() + "..."

    score = float(chunk.get("score", 0.0))
    return f"score={score:.4f} | {date} | {sender} | {subject} | {source}\n{text}"


def _build_prompt(
    system_policy: str,
    prompt_artifact: dict[str, Any],
    llm_plan: dict[str, Any],
    question: str,
    context: str,
    retrieved_chunks: list[dict[str, Any]],
    corpus_stats: dict[str, Any],
    query_hit_count: int,
    max_snippet_chars: int,
) -> str:
    instruction = str(prompt_artifact.get("instructions", DEFAULT_PROMPT_ARTIFACT["instructions"]))
    default_context = str(prompt_artifact.get("default_context", DEFAULT_PROMPT_ARTIFACT["default_context"]))
    demos = prompt_artifact.get("few_shot_demos", [])

    lines: list[str] = []
    if system_policy.strip():
        lines.append("System policy:")
        lines.append(system_policy.strip())
    if llm_plan:
        lines.append("LLM query plan:")
        lines.append(json.dumps(llm_plan, ensure_ascii=True))
    lines.append(instruction)
    if default_context:
        lines.append(f"Domain context: {default_context}")
    if context:
        lines.append(f"Request context: {context}")

    lines.append("Indexed corpus stats:")
    lines.append(f"- indexed_chunks: {corpus_stats.get('indexed_chunks', 0)}")
    lines.append(f"- unique_messages: {corpus_stats.get('unique_messages', 0)}")
    lines.append(f"- query_token_hits: {query_hit_count}")

    top_senders = corpus_stats.get("top_senders", [])
    if isinstance(top_senders, list) and top_senders:
        sender_line = ", ".join([f"{sender} ({count})" for sender, count in top_senders[:5]])
        lines.append(f"- top_senders: {sender_line}")

    if isinstance(demos, list) and demos:
        lines.append("Examples:")
        for idx, demo in enumerate(demos[:3], start=1):
            question_demo = str((demo or {}).get("question", "")).strip()
            answer_demo = str((demo or {}).get("answer", "")).strip()
            if question_demo and answer_demo:
                lines.append(f"{idx}. Q: {question_demo}")
                lines.append(f"   A: {answer_demo}")

    lines.append("Retrieved evidence:")
    if retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            lines.append(f"{idx}. {_format_evidence(chunk, max_snippet_chars=max_snippet_chars)}")
    else:
        lines.append("No snippets retrieved for this query.")

    lines.append("Response requirements:")
    lines.append("1) Be concise and concrete.")
    lines.append("2) If evidence exists, cite sender, subject, and date.")
    lines.append("3) If evidence is missing, say exactly what filter/query to run next.")

    lines.append(f"User question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _normalize_task(task: str) -> str:
    requested = (task or "").strip().lower()
    if not requested:
        return "text-generation"
    return TASK_ALIASES.get(requested, requested)


def _safe_model_max_input_tokens(tokenizer: Any, default_max_tokens: int = 2048) -> int:
    model_max = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 100_000:
        return default_max_tokens
    return model_max


def _build_generator(task: str, model_id: str):
    """Create a local text generator for either seq2seq or causal HF models.

    Seq2seq models (e.g. FLAN/T5) are loaded in a manual `generate()` path.
    Causal models use a transformers pipeline with the configured task.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for local generation. Install torch in your environment.")

    pipeline_device = -1
    model_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        pipeline_device = 0
        model_device = "cuda"

    normalized_task = _normalize_task(task)
    config = None
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        config = None

    if config is not None and bool(getattr(config, "is_encoder_decoder", False)):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(model_device)
        model.eval()
        return {
            "engine": "seq2seq-manual",
            "model": model,
            "tokenizer": tokenizer,
            "device": model_device,
        }

    try:
        pipe = pipeline(
            normalized_task,
            model=model_id,
            tokenizer=model_id,
            device=pipeline_device,
        )
        model_obj = getattr(pipe, "model", None)
        generation_config = getattr(model_obj, "generation_config", None)
        if generation_config is not None:
            # Avoid common warnings from instruct defaults when running deterministic generations.
            if hasattr(generation_config, "do_sample"):
                generation_config.do_sample = False
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = 1.0
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = 1.0
            if hasattr(generation_config, "top_k"):
                generation_config.top_k = 50
            if hasattr(generation_config, "max_length"):
                generation_config.max_length = None
        return pipe
    except Exception as exc:
        raise RuntimeError(
            f"Could not build a local generator for task '{task}' and model '{model_id}'. "
            "Use a compatible local model/task combination."
        ) from exc


def _generate_answer(generator: Any, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Generate text from the resolved generator backend using shared parameters."""
    if isinstance(generator, dict) and str(generator.get("engine", "")).endswith("-manual"):
        if torch is None:
            raise RuntimeError("PyTorch is required for manual generation.")
        model = generator.get("model")
        tokenizer = generator.get("tokenizer")
        device = str(generator.get("device", "cpu"))
        if model is None or tokenizer is None:
            raise RuntimeError("Manual generator is missing model/tokenizer.")

        max_input_tokens = _safe_model_max_input_tokens(tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text

    generation_kwargs: dict[str, Any] = {}
    model_obj = getattr(generator, "model", None)
    base_cfg = getattr(model_obj, "generation_config", None)
    if base_cfg is not None:
        cfg = GenerationConfig.from_dict(base_cfg.to_dict())
        cfg.max_new_tokens = max_new_tokens
        cfg.max_length = None
        cfg.do_sample = temperature > 0
        if temperature > 0:
            cfg.temperature = temperature
        else:
            cfg.temperature = 1.0
            cfg.top_p = 1.0
            cfg.top_k = 50
        generation_kwargs["generation_config"] = cfg
    else:
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["do_sample"] = temperature > 0
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

    result = generator(prompt, **generation_kwargs)

    if isinstance(result, list) and result:
        candidate = result[0]
        if isinstance(candidate, dict):
            for key in ("generated_text", "summary_text", "text"):
                if key in candidate:
                    generated = str(candidate[key]).strip()
                    if generated.startswith(prompt):
                        generated = generated[len(prompt) :].strip()
                    return generated
        generated = str(candidate).strip()
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].strip()
        return generated

    generated = str(result).strip()
    if generated.startswith(prompt):
        generated = generated[len(prompt) :].strip()
    return generated


def _run_single_question(
    question: str,
    context: str,
    system_policy: str,
    prompt_artifact: dict[str, Any],
    retriever: TfidfRagRetriever,
    generator: Any,
    email_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute the full retrieval + answer pipeline for one user question.

    Flow: plan query with LLM -> retrieve chunks -> build grounded prompt ->
    generate final answer -> return evidence and send candidates.
    """
    if retriever is None:
        raise RuntimeError("Retriever is not initialized")
    if generator is None:
        raise RuntimeError("Generator is not initialized")

    llm_plan = _plan_query_with_llm(
        question=question,
        context=context,
        system_policy=system_policy,
        generator=generator,
    )

    plan_topic_terms = llm_plan.get("topic_terms", [])
    topic_terms = [str(t).strip() for t in plan_topic_terms if str(t).strip()] if isinstance(plan_topic_terms, list) else []
    sender_filter = str(llm_plan.get("sender_contains", "")).strip()
    selected_k = max(args.rag_top_k, int(llm_plan.get("max_results", args.rag_top_k)))
    query_seed = " ".join(topic_terms) if topic_terms else question
    if sender_filter:
        query_seed = f"{query_seed} sender:{sender_filter}"
    retrieval_query = query_seed if not context else f"{query_seed}\n{context}"
    retrieval_candidates = retriever.retrieve(
        query=retrieval_query,
        top_k=max(args.rag_top_k * 4, selected_k),
        min_score=args.rag_min_score,
    )
    retrieved_chunks = retrieval_candidates[:selected_k]

    corpus_stats = _email_stats(email_rows)
    unique_email_hits: set[str] = set()
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        if metadata.get("kind") != "email":
            continue
        key = str(metadata.get("message_id", "")).strip() or str(chunk.get("chunk_id", "")).strip()
        if key:
            unique_email_hits.add(key)
    hit_count = len(unique_email_hits)

    prompt = _build_prompt(
        system_policy=system_policy,
        prompt_artifact=prompt_artifact,
        llm_plan=llm_plan,
        question=question,
        context=context,
        retrieved_chunks=retrieved_chunks,
        corpus_stats=corpus_stats,
        query_hit_count=hit_count,
        max_snippet_chars=min(args.max_snippet_chars, 180),
    )

    answer = _generate_answer(
        generator,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    evidence: list[dict[str, Any]] = []
    send_candidates: list[dict[str, Any]] = []
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata") if isinstance(chunk, dict) else None
        evidence.append(
            {
                "score": float(chunk.get("score", 0.0)),
                "source": str(chunk.get("source", "")),
                "subject": str((metadata or {}).get("subject", "")),
                "from": str((metadata or {}).get("from", "")),
                "date": str((metadata or {}).get("date", "")),
            }
        )
        if isinstance(metadata, dict) and metadata.get("kind") == "email":
            send_candidates.append(
                {
                    "source": str(chunk.get("source", "")),
                    "subject": str(metadata.get("subject", "")),
                    "from": str(metadata.get("from", "")),
                    "date": str(metadata.get("date", "")),
                    "snippet": str(metadata.get("snippet", "")),
                    "text": str(chunk.get("text", "")),
                }
            )

    return {
        "question": question,
        "answer": answer,
        "query_hit_count": hit_count,
        "evidence": evidence,
        "send_candidates": send_candidates[: max(1, selected_k)],
        "llm_plan": llm_plan,
    }


def main() -> None:
    """Run the assistant entrypoint in interactive chat mode with auto-sync."""
    args = parse_args()

    if int(args.sync_batch_size) <= 0:
        raise ValueError("--sync-batch-size must be greater than 0")
    if not str(args.mcp_server_command).strip():
        raise ValueError("Missing --mcp-server-command (MCP-only runtime).")

    prompt_artifact_path = _resolve_prompt_artifact_path(args.prompt_artifact)
    prompt_artifact, prompt_source = _load_prompt_artifact(prompt_artifact_path)
    if prompt_source == "built-in-default":
        print("[info] Using built-in default prompt artifact (no DSPy optimized prompt found).")
    else:
        print(f"[info] Using DSPy optimized prompt artifact: {prompt_source}")
    system_policy = _load_system_policy(args.system_prompt_file)
    generator: Any = _build_generator(task=args.hf_task, model_id=args.hf_model_id)

    initial_sync_summary: dict[str, Any] | None = None
    email_chunks_path = Path(args.email_chunks)
    existing_rows = _load_rows(email_chunks_path) if email_chunks_path.exists() else []
    if not existing_rows:
        initial_sync_summary = _sync_email_window(args, offset=0, append=False)
        print(
            f"[info] Downloaded initial email batch: "
            f"{initial_sync_summary.get('messages_indexed_this_batch', 0)} messages"
        )

    retriever, email_rows = _load_retriever_and_rows(args)
    loaded_messages = len(_unique_sorted_email_rows(email_rows))
    next_sync_offset = loaded_messages
    has_more_history = True
    if initial_sync_summary is not None:
        has_more_history = bool(initial_sync_summary.get("messages_window_start", 0) > 0)

    print("Chat mode. Ask about emails, then ask to send one (example: 'send that email to user@domain.com').")
    print("Type 'exit' to quit.")
    last_result: dict[str, Any] | None = None

    while True:
        try:
            user_message = input("email-chat> ").strip()
        except EOFError:
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        if not args.json_output:
            print(f"{args.hf_model_id} Thinking.....")

        try:
            action_plan = _plan_chat_action_with_llm(
                user_message=user_message,
                system_policy=system_policy,
                session_has_last_result=last_result is not None,
                last_result_preview=_last_result_preview(last_result),
                generator=generator,
            )
        except Exception as exc:
            planner_error = f"Chat planner failed: {exc}"
            if args.json_output:
                print(
                    json.dumps(
                        {
                            "warning": planner_error,
                            "fallback_action": "search",
                        },
                        indent=2,
                        ensure_ascii=True,
                    )
                )
            else:
                print(f"{planner_error}. Continuing as search.")
                print("")
            action_plan = {
                "action": "search",
                "recipient_email": "",
                "candidate_index": 1,
                "email_subject": "",
                "email_instruction": "",
            }
        action = str(action_plan.get("action", "search")).strip().lower()

        if action == "help":
            help_text = (
                "Ask any search question, then ask to send the last found email.\n"
                "Examples:\n"
                "- find latest email about Supersonic\n"
                "- send this email to someone@example.com\n"
                "- send email to someone@example.com with subject \"hello\" and body \"message\""
            )
            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "message": help_text}, indent=2, ensure_ascii=True))
            else:
                print(help_text)
                print("")
            continue

        if action == "send_last":
            if last_result is None:
                message = "No previous search result available. Ask a search question first."
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": message}, indent=2, ensure_ascii=True))
                else:
                    print(message)
                    print("")
                continue

            recipient = str(action_plan.get("recipient_email", "")).strip() or str(args.send_to or "").strip()
            if not recipient:
                message = "Missing recipient email. Say: send this email to user@domain.com"
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": message}, indent=2, ensure_ascii=True))
                else:
                    print(message)
                    print("")
                continue

            candidate_raw = action_plan.get("candidate_index", 1)
            try:
                candidate_index = int(candidate_raw)
            except Exception:
                candidate_index = 1

            try:
                send_status = _send_from_result(
                    result=last_result,
                    recipient=recipient,
                    candidate_index=candidate_index,
                    args=args,
                    instruction=str(action_plan.get("email_instruction", "")),
                    subject_override=str(action_plan.get("email_subject", "")),
                )
            except Exception as exc:
                error_message = f"Send failed: {exc}"
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": error_message}, indent=2, ensure_ascii=True))
                else:
                    print(error_message)
                    print("")
                continue

            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "send_status": send_status}, indent=2, ensure_ascii=True))
            else:
                print(send_status)
                print("")
            continue

        if action == "compose_send":
            recipient = str(action_plan.get("recipient_email", "")).strip() or str(args.send_to or "").strip()
            if not recipient:
                message = "Missing recipient email. Say: send email to user@domain.com with subject \"...\" and body \"...\""
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": message}, indent=2, ensure_ascii=True))
                else:
                    print(message)
                    print("")
                continue

            body = str(action_plan.get("email_instruction", "")).strip()
            if not body:
                message = "Missing email body. Add body/content in your command."
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": message}, indent=2, ensure_ascii=True))
                else:
                    print(message)
                    print("")
                continue

            subject = str(action_plan.get("email_subject", "")).strip() or "Message from Local Gmail Assistant"
            try:
                send_status = _send_email_via_transport(
                    args=args,
                    send_to=recipient,
                    subject=subject,
                    body=body,
                )
            except Exception as exc:
                error_message = f"Send failed: {exc}"
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "error": error_message}, indent=2, ensure_ascii=True))
                else:
                    print(error_message)
                    print("")
                continue

            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "send_status": send_status}, indent=2, ensure_ascii=True))
            else:
                print(send_status)
                print("")
            continue

        if action == "none":
            message = "No action inferred. Ask a search question or request sending with a recipient email."
            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "message": message}, indent=2, ensure_ascii=True))
            else:
                print(message)
                print("")
            continue

        try:
            result = _run_single_question(
                question=user_message,
                context=str(args.context),
                system_policy=system_policy,
                prompt_artifact=prompt_artifact,
                retriever=retriever,
                generator=generator,
                email_rows=email_rows,
                args=args,
            )
        except Exception as exc:
            query_error = f"Search failed: {exc}"
            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "error": query_error}, indent=2, ensure_ascii=True))
            else:
                print(query_error)
                print("")
            continue

        if int(result.get("query_hit_count", 0)) == 0 and has_more_history:
            try:
                sync_summary = _sync_email_window(args, offset=next_sync_offset, append=True)
                fetched = int(sync_summary.get("messages_indexed_this_batch", 0))
                has_more_history = bool(sync_summary.get("messages_window_start", 0) > 0)
                next_sync_offset += int(args.sync_batch_size)

                if fetched > 0:
                    if args.json_output:
                        print(
                            json.dumps(
                                {
                                    "action_plan": action_plan,
                                    "sync": {
                                        "status": "fetched-next-batch",
                                        "batch_size": fetched,
                                        "offset": sync_summary.get("offset", 0),
                                    },
                                },
                                indent=2,
                                ensure_ascii=True,
                            )
                        )
                    else:
                        print(f"[info] No hits. Downloaded next batch of {fetched} emails and retrying.")
                        print("")

                    retriever, email_rows = _load_retriever_and_rows(args)
                    result = _run_single_question(
                        question=user_message,
                        context=str(args.context),
                        system_policy=system_policy,
                        prompt_artifact=prompt_artifact,
                        retriever=retriever,
                        generator=generator,
                        email_rows=email_rows,
                        args=args,
                    )
            except Exception as exc:
                sync_error = f"Auto-sync failed after zero hits: {exc}"
                if args.json_output:
                    print(json.dumps({"action_plan": action_plan, "warning": sync_error}, indent=2, ensure_ascii=True))
                else:
                    print(sync_error)
                    print("")

        last_result = result

        if args.json_output:
            print(json.dumps({"action_plan": action_plan, "result": result}, indent=2, ensure_ascii=True))
            continue

        _print_result(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

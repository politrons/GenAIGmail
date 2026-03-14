from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import sys
from collections import Counter
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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
    "You are a query planner for email analytics.\n"
    "Given a user question, extract a strict JSON plan for retrieval.\n"
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "intent": "count_latest|latest_list|summary|open_qa",\n'
    '  "topic_terms": ["term1","term2"],\n'
    '  "must_match_all_terms": true,\n'
    '  "sender_contains": "",\n'
    '  "request_count": false,\n'
    '  "request_latest": false,\n'
    '  "max_results": 6,\n'
    '  "language": "en"\n'
    "}\n"
    "Rules:\n"
    "- If the user asks how many + latest, set intent=count_latest, request_count=true, request_latest=true.\n"
    "- Put only meaningful topic terms, no generic words.\n"
    "- If sender is requested, set sender_contains.\n"
    "- Keep max_results in [3, 10]."
)

CHAT_ACTION_PLANNER_PROMPT = (
    "You are a chat action planner for an email assistant.\n"
    "Given the user message and current session context, return ONLY valid JSON.\n"
    "Schema:\n"
    "{\n"
    '  "action": "search|send_last|help|none",\n'
    '  "recipient_email": "",\n'
    '  "candidate_index": 1,\n'
    '  "email_subject": "",\n'
    '  "email_instruction": ""\n'
    "}\n"
    "Rules:\n"
    "- Use action=send_last only if user clearly asks to send/forward and gives a recipient.\n"
    "- candidate_index is 1-based. Default 1.\n"
    "- If recipient missing for send request, set action=help.\n"
    "- For normal analytics/search questions, set action=search."
)

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "about",
    "please",
    "have",
    "has",
    "are",
    "was",
    "were",
    "pero",
    "para",
    "sobre",
    "quiero",
    "como",
    "donde",
    "cuando",
    "que",
    "los",
    "las",
}

PLANNER_NOISE_TERMS = {
    "how",
    "many",
    "what",
    "latest",
    "recent",
    "most",
    "email",
    "emails",
    "mention",
    "mentions",
    "cuantos",
    "cuántos",
    "cuantas",
    "cuántas",
    "correo",
    "correos",
    "menciona",
    "mencionan",
    "reciente",
    "mas",
    "más",
    "cual",
    "cuál",
}


TASK_ALIASES = {
    "text2text-generation": "text-generation",
}


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one-shot and chat-based email assistant modes."""
    _load_env_file(".env")

    parser = argparse.ArgumentParser(description="Ask questions about indexed Gmail data using a local model")
    parser.add_argument("--question", default=None, help="Single question to answer. If omitted, interactive mode starts.")
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
    parser.add_argument("--hf-model-id", default=os.getenv("LOCAL_HF_MODEL_ID", "google/flan-t5-base"))
    parser.add_argument("--hf-task", default=os.getenv("LOCAL_HF_TASK", "text-generation"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCAL_MAX_NEW_TOKENS", "220")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LOCAL_TEMPERATURE", "0.0")))
    parser.add_argument("--rag-top-k", type=int, default=int(os.getenv("LOCAL_RAG_TOP_K", "6")))
    parser.add_argument("--rag-min-score", type=float, default=float(os.getenv("LOCAL_RAG_MIN_SCORE", "0.01")))
    parser.add_argument("--max-snippet-chars", type=int, default=int(os.getenv("LOCAL_MAX_SNIPPET_CHARS", "420")))
    parser.add_argument("--smtp-host", default=os.getenv("GMAIL_SMTP_HOST", "smtp.gmail.com"))
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("GMAIL_SMTP_PORT", "465")))
    parser.add_argument("--smtp-user", default=os.getenv("SMTP_USER", os.getenv("GMAIL_USER", "")))
    parser.add_argument("--smtp-password", default=os.getenv("SMTP_PASSWORD", os.getenv("GMAIL_PASSWORD", "")))
    parser.add_argument("--send-from", default=os.getenv("GMAIL_SEND_FROM", os.getenv("GMAIL_USER", "")))
    parser.add_argument("--send-to", default="", help="Optional recipient email to send top result after answering.")
    parser.add_argument("--send-dry-run", action="store_true", help="Do not send. Print what would be sent.")
    parser.add_argument("--chat", action="store_true", help="Force chat mode even when --question is provided.")
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


def _extract_query_tokens(text: str) -> list[str]:
    tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z0-9_]+", text)]
    out: list[str] = []
    for tok in tokens:
        if len(tok) <= 2:
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


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


def _query_hit_count(rows: list[dict[str, Any]], query: str) -> int:
    tokens = _extract_query_tokens(query)
    if not tokens:
        return 0

    hits: set[str] = set()
    for row in rows:
        text = str(row.get("text", "")).lower()
        if not text:
            continue
        if any(token in text for token in tokens):
            meta = row.get("metadata")
            if isinstance(meta, dict):
                message_id = str(meta.get("message_id", "")).strip()
            else:
                message_id = ""
            hits.add(message_id or str(row.get("chunk_id", "")))
    return len(hits)


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


def _fallback_llm_plan(question: str) -> dict[str, Any]:
    terms = _extract_focus_terms(question)[:5]
    question_lower = question.lower()
    request_count = any(token in question_lower for token in ("how many", "cuantos", "cuántos"))
    request_latest = any(token in question_lower for token in ("latest", "recent", "reciente", "mas reciente", "más reciente"))
    intent = "open_qa"
    if request_count and request_latest:
        intent = "count_latest"
    elif request_latest:
        intent = "latest_list"
    elif any(token in question_lower for token in ("summarize", "resumen", "resume")):
        intent = "summary"

    return {
        "intent": intent,
        "topic_terms": terms,
        "must_match_all_terms": True,
        "sender_contains": "",
        "request_count": request_count,
        "request_latest": request_latest,
        "max_results": 6,
        "language": "en",
    }


def _normalize_llm_plan(plan: dict[str, Any], question: str) -> dict[str, Any]:
    base = _fallback_llm_plan(question)

    if not isinstance(plan, dict):
        return base

    intent = str(plan.get("intent", base["intent"])).strip()
    if intent not in {"count_latest", "latest_list", "summary", "open_qa"}:
        intent = base["intent"]

    raw_terms = plan.get("topic_terms", [])
    terms: list[str] = []
    if isinstance(raw_terms, list):
        for item in raw_terms:
            term = str(item).strip().lower()
            if not term:
                continue
            if term in STOPWORDS:
                continue
            if term in PLANNER_NOISE_TERMS:
                continue
            terms.append(term)
    if not terms:
        terms = _extract_focus_terms(question)[:5]
    if not terms:
        terms = list(base["topic_terms"])

    must_match_all = bool(plan.get("must_match_all_terms", True))
    sender_contains = str(plan.get("sender_contains", "")).strip().lower()
    request_count = bool(plan.get("request_count", False))
    request_latest = bool(plan.get("request_latest", False))

    max_results_raw = plan.get("max_results", 6)
    try:
        max_results = int(max_results_raw)
    except Exception:
        max_results = 6
    max_results = max(3, min(10, max_results))

    language = str(plan.get("language", "en")).strip().lower() or "en"

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
    """Ask the LLM to build a structured retrieval plan for the current question.

    The plan is expected as strict JSON. If parsing fails, a deterministic fallback
    plan is returned to keep runtime resilient.
    """
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
    json_blob = _extract_json_object(raw_plan)
    if json_blob is None:
        return _fallback_llm_plan(question)

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return _fallback_llm_plan(question)

    return _normalize_llm_plan(parsed, question)


def _row_matches_plan(row: dict[str, Any], plan: dict[str, Any]) -> bool:
    text = str(row.get("text", "")).lower()
    if not text:
        return False

    terms = plan.get("topic_terms", [])
    if not isinstance(terms, list):
        terms = []
    topic_terms = [str(t).strip().lower() for t in terms if str(t).strip()]

    if topic_terms:
        if bool(plan.get("must_match_all_terms", True)):
            if not all(term in text for term in topic_terms):
                return False
        else:
            if not any(term in text for term in topic_terms):
                return False

    sender_filter = str(plan.get("sender_contains", "")).strip().lower()
    if sender_filter:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        sender = str((meta or {}).get("from", "")).lower()
        if sender_filter not in sender:
            return False

    return True


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


def _extract_email_address(text: str) -> str:
    match = EMAIL_REGEX.search(text or "")
    return match.group(0) if match else ""


def _fallback_chat_action(user_message: str) -> dict[str, Any]:
    message = (user_message or "").strip()
    lower = message.lower()
    recipient = _extract_email_address(message)
    wants_send = any(token in lower for token in ("send", "forward", "envia", "enviar", "manda", "reenviar"))
    if wants_send:
        if recipient:
            return {
                "action": "send_last",
                "recipient_email": recipient,
                "candidate_index": 1,
                "email_subject": "",
                "email_instruction": "",
            }
        return {
            "action": "help",
            "recipient_email": "",
            "candidate_index": 1,
            "email_subject": "",
            "email_instruction": "Missing recipient email",
        }

    return {
        "action": "search",
        "recipient_email": "",
        "candidate_index": 1,
        "email_subject": "",
        "email_instruction": "",
    }


def _normalize_chat_action(plan: dict[str, Any], user_message: str) -> dict[str, Any]:
    base = _fallback_chat_action(user_message)
    if not isinstance(plan, dict):
        return base

    action = str(plan.get("action", base["action"])).strip().lower()
    if action not in {"search", "send_last", "help", "none"}:
        action = base["action"]

    recipient = str(plan.get("recipient_email", "")).strip()
    if recipient and not EMAIL_REGEX.fullmatch(recipient):
        extracted = _extract_email_address(recipient)
        recipient = extracted

    candidate_raw = plan.get("candidate_index", 1)
    try:
        candidate_index = int(candidate_raw)
    except Exception:
        candidate_index = 1
    candidate_index = max(1, candidate_index)

    email_subject = str(plan.get("email_subject", "")).strip()
    email_instruction = str(plan.get("email_instruction", "")).strip()

    if action == "send_last" and not recipient:
        extracted = _extract_email_address(user_message)
        if extracted:
            recipient = extracted
        else:
            action = "help"
            email_instruction = "Missing recipient email"

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
    """Classify a chat turn into an assistant action using LLM JSON planning.

    Expected actions are `search`, `send_last`, `help`, or `none`. Invalid JSON
    responses are handled via a small fallback classifier.
    """
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
    json_blob = _extract_json_object(raw)
    if json_blob is None:
        return _fallback_chat_action(user_message)
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return _fallback_chat_action(user_message)
    return _normalize_chat_action(parsed, user_message)


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


def _send_email_via_smtp(
    *,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    send_from: str,
    send_to: str,
    subject: str,
    body: str,
    dry_run: bool = False,
) -> str:
    if not EMAIL_REGEX.fullmatch(send_to):
        raise ValueError(f"Invalid recipient email: {send_to}")
    if not smtp_user or not smtp_password:
        raise ValueError("Missing SMTP credentials. Set --smtp-user/--smtp-password or env vars.")

    from_addr = send_from or smtp_user
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = send_to
    msg["Subject"] = subject
    msg.set_content(body)

    if dry_run:
        return f"[dry-run] Email prepared for {send_to} with subject '{subject}'."

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    return f"Email sent to {send_to} with subject '{subject}'."


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
    return _send_email_via_smtp(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=str(args.smtp_user).strip(),
        smtp_password=str(args.smtp_password).strip(),
        send_from=str(args.send_from).strip(),
        send_to=recipient,
        subject=subject,
        body=body,
        dry_run=bool(args.send_dry_run),
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


def _extract_focus_terms(question: str) -> list[str]:
    lower_question = question.lower()

    explicit_terms: list[str] = []
    mention_match = re.search(r"\b(?:menciona|mencionan|mention|mentions)\s+([a-zA-Z0-9_.-]+)", lower_question)
    if mention_match:
        explicit_terms.append(mention_match.group(1).strip().lower())

    quoted_terms: list[str] = []
    for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', question):
        for candidate in match:
            cleaned = candidate.strip().lower()
            if cleaned:
                quoted_terms.append(cleaned)

    generic_intent_tokens = {
        "cuantos",
        "cuantas",
        "correos",
        "correo",
        "emails",
        "email",
        "how",
        "many",
        "what",
        "one",
        "latest",
        "most",
        "recent",
        "mention",
        "mentions",
        "the",
        "and",
        "menciona",
        "mencionan",
        "cual",
        "mas",
        "más",
        "reciente",
    }
    token_terms = [t for t in _extract_query_tokens(question) if t not in generic_intent_tokens]

    # If we already captured explicit topic terms (e.g. after "mention[s]"),
    # avoid diluting matching with generic intent tokens from the whole question.
    if explicit_terms:
        token_terms = []

    ordered: list[str] = []
    seen: set[str] = set()
    for term in [*explicit_terms, *quoted_terms, *token_terms]:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


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


def _safe_model_max_input_tokens(tokenizer: Any, fallback: int = 2048) -> int:
    model_max = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 100_000:
        return fallback
    return model_max


def _build_generator(task: str, model_id: str):
    """Create a local text generator for either seq2seq or causal HF models.

    Seq2seq models (e.g. FLAN/T5) are loaded in a manual `generate()` path.
    Causal models prefer transformers pipeline and fall back to manual loading.
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
        return pipeline(
            normalized_task,
            model=model_id,
            tokenizer=model_id,
            device=pipeline_device,
        )
    except Exception as exc:
        fallback_tasks: list[str] = []
        if normalized_task != "text-generation":
            fallback_tasks.append("text-generation")
        if normalized_task != "text2text-generation":
            fallback_tasks.append("text2text-generation")

        for fallback_task in fallback_tasks:
            try:
                return pipeline(
                    fallback_task,
                    model=model_id,
                    tokenizer=model_id,
                    device=pipeline_device,
                )
            except Exception:
                continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model.to(model_device)
            model.eval()
            return {
                "engine": "causal-manual",
                "model": model,
                "tokenizer": tokenizer,
                "device": model_device,
            }
        except Exception as manual_exc:
            raise RuntimeError(
                f"Could not build a local generator for task '{task}' and model '{model_id}'. "
                "Try another --hf-model-id or a different --hf-task."
            ) from manual_exc


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
        if str(generator.get("engine")) == "causal-manual" and text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
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

    Flow: plan query with LLM -> retrieve chunks -> apply plan filters -> build
    grounded prompt -> generate final answer -> return evidence and send candidates.
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

    topic_terms = llm_plan.get("topic_terms", [])
    if isinstance(topic_terms, list) and topic_terms:
        query_seed = " ".join([str(t) for t in topic_terms if str(t).strip()])
    else:
        query_seed = question
    sender_filter = str(llm_plan.get("sender_contains", "")).strip()
    if sender_filter:
        query_seed = f"{query_seed} sender:{sender_filter}"

    retrieval_query = query_seed if not context else f"{query_seed}\n{context}"
    retrieval_candidates = retriever.retrieve(
        query=retrieval_query,
        top_k=max(args.rag_top_k * 4, args.rag_top_k),
        min_score=args.rag_min_score,
    )

    email_chunks = [c for c in retrieval_candidates if isinstance(c.get("metadata"), dict) and c.get("metadata", {}).get("kind") == "email"]
    filtered_email_chunks = [c for c in email_chunks if _row_matches_plan(c, llm_plan)]
    if filtered_email_chunks:
        email_chunks = filtered_email_chunks

    non_email_chunks = [
        c for c in retrieval_candidates if not (isinstance(c.get("metadata"), dict) and c.get("metadata", {}).get("kind") == "email")
    ]

    selected_k = max(args.rag_top_k, int(llm_plan.get("max_results", args.rag_top_k)))
    retrieved_chunks = (email_chunks + non_email_chunks)[: selected_k]

    corpus_stats = _email_stats(email_rows)
    matched_rows = [row for row in email_rows if _row_matches_plan(row, llm_plan)]
    hit_count = len(_unique_sorted_email_rows(matched_rows))
    if hit_count == 0:
        hit_count = _query_hit_count(email_rows, retrieval_query)

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
    """Run the assistant entrypoint in one-shot mode or interactive chat mode."""
    args = parse_args()

    email_chunks_path = Path(args.email_chunks)
    if not email_chunks_path.exists():
        raise FileNotFoundError(
            f"Email chunks file not found: {email_chunks_path}. Run `python3 main.py sync-gmail -- ...` first."
        )

    retriever_paths: list[str] = [str(email_chunks_path)]
    knowledge_chunks = str(args.knowledge_chunks or "").strip()
    if knowledge_chunks:
        kb_path = Path(knowledge_chunks)
        if kb_path.exists():
            retriever_paths.append(str(kb_path))

    prompt_artifact_path = _resolve_prompt_artifact_path(args.prompt_artifact)
    prompt_artifact, prompt_source = _load_prompt_artifact(prompt_artifact_path)
    if prompt_source == "built-in-default":
        print("[info] Using built-in default prompt artifact (no DSPy optimized prompt found).")
    else:
        print(f"[info] Using DSPy optimized prompt artifact: {prompt_source}")
    system_policy = _load_system_policy(args.system_prompt_file)
    retriever = TfidfRagRetriever.from_jsonl_paths(retriever_paths)
    generator: Any = _build_generator(task=args.hf_task, model_id=args.hf_model_id)

    email_rows = _load_rows(email_chunks_path)

    interactive = bool(args.chat or args.question is None)
    if not interactive:
        user_question = str(args.question or "").strip()
        if not user_question:
            raise ValueError("Missing --question. Provide a question or run with --chat.")

        result = _run_single_question(
            question=user_question,
            context=str(args.context),
            system_policy=system_policy,
            prompt_artifact=prompt_artifact,
            retriever=retriever,
            generator=generator,
            email_rows=email_rows,
            args=args,
        )
        send_status = ""
        recipient = str(args.send_to or "").strip()
        if recipient:
            send_status = _send_from_result(
                result=result,
                recipient=recipient,
                candidate_index=1,
                args=args,
            )

        if args.json_output:
            payload: dict[str, Any] = {"result": result}
            if send_status:
                payload["send_status"] = send_status
            print(json.dumps(payload, indent=2, ensure_ascii=True))
            return

        _print_result(result)
        if send_status:
            print(send_status)
        return

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

        action_plan = _plan_chat_action_with_llm(
            user_message=user_message,
            system_policy=system_policy,
            session_has_last_result=last_result is not None,
            last_result_preview=_last_result_preview(last_result),
            generator=generator,
        )
        action = str(action_plan.get("action", "search")).strip().lower()

        if action == "help":
            help_text = (
                "Ask any search question, then ask to send the last found email.\n"
                "Examples:\n"
                "- find latest email about Supersonic\n"
                "- send this email to someone@example.com"
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

        if action == "none":
            message = "No action inferred. Ask a search question or request sending with a recipient email."
            if args.json_output:
                print(json.dumps({"action_plan": action_plan, "message": message}, indent=2, ensure_ascii=True))
            else:
                print(message)
                print("")
            continue

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

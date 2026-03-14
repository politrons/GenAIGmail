from __future__ import annotations

import argparse
import imaplib
import json
import os
import re
import shlex
from collections import Counter
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


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


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI and environment-backed settings for Gmail IMAP synchronization."""
    _load_env_file(".env")

    parser = argparse.ArgumentParser(description="Sync Gmail messages to a local JSONL index")
    parser.add_argument(
        "--gmail-user",
        default=_env_value("GMAIL_USER", "gmail_user", "GAMAIL_USER", "gamail_user"),
        help="Gmail user/email",
    )
    parser.add_argument(
        "--gmail-password",
        default=_env_value("GMAIL_PASSWORD", "gmail_password", "GAMAIL_PASSWORD", "gamail_password"),
        help="Gmail app password or account password if enabled",
    )
    parser.add_argument("--imap-host", default=_env_value("GMAIL_IMAP_HOST", "gmail_imap_host") or "imap.gmail.com")
    parser.add_argument(
        "--imap-port",
        type=int,
        default=int(_env_value("GMAIL_IMAP_PORT", "gmail_imap_port") or "993"),
    )
    parser.add_argument("--mailbox", default=_env_value("GMAIL_MAILBOX", "gmail_mailbox") or "INBOX")
    parser.add_argument(
        "--search-criterion",
        default=_env_value("GMAIL_SEARCH_CRITERION", "gmail_search_criterion") or "ALL",
        help='IMAP search expression, e.g. "ALL", "UNSEEN", "SINCE 01-Mar-2026"',
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=int(_env_value("GMAIL_MAX_EMAILS", "gmail_max_emails") or "250"),
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=int(_env_value("GMAIL_MAX_BODY_CHARS", "gmail_max_body_chars") or "5000"),
    )
    parser.add_argument(
        "--output-jsonl",
        default=_env_value("GMAIL_CHUNKS_PATH", "gmail_chunks_path") or "data/gmail_chunks.jsonl",
    )
    parser.add_argument("--include-html-fallback", action="store_true")
    return parser.parse_args()


def _decode_header_value(value: str) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return str(value).strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_whitespace(text)


def _extract_text_body(message, include_html_fallback: bool) -> str:
    """Extract a plain-text body from a MIME email, skipping attachments."""
    text_parts: list[str] = []
    html_parts: list[str] = []

    if message.is_multipart():
        for part in message.walk():
            content_disposition = str(part.get_content_disposition() or "").lower()
            if content_disposition == "attachment":
                continue

            content_type = str(part.get_content_type() or "").lower()
            if content_type == "text/plain":
                try:
                    text = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                text = _normalize_whitespace(str(text))
                if text:
                    text_parts.append(text)
            elif include_html_fallback and content_type == "text/html":
                try:
                    html = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                html = _strip_html(str(html))
                if html:
                    html_parts.append(html)
    else:
        content_type = str(message.get_content_type() or "").lower()
        try:
            body = message.get_content()
        except Exception:
            payload = message.get_payload(decode=True) or b""
            charset = message.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

        if content_type == "text/plain":
            cleaned = _normalize_whitespace(str(body))
            if cleaned:
                text_parts.append(cleaned)
        elif include_html_fallback and content_type == "text/html":
            cleaned = _strip_html(str(body))
            if cleaned:
                html_parts.append(cleaned)

    if text_parts:
        return "\n\n".join(text_parts)
    if html_parts:
        return "\n\n".join(html_parts)
    return ""


def _date_to_iso(raw_date: str) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        if dt is None:
            return raw_date
        return dt.isoformat()
    except Exception:
        return raw_date


def _build_text_blob(subject: str, sender: str, recipients: str, date: str, body: str) -> str:
    lines = [
        f"Subject: {subject}",
        f"From: {sender}",
        f"To: {recipients}",
        f"Date: {date}",
        "Body:",
        body,
    ]
    return "\n".join(lines).strip()


def _raise_auth_error(user: str, exc: Exception) -> None:
    """Raise a user-friendly authentication error with actionable Gmail hints."""
    raw = str(exc)
    lower_raw = raw.lower()
    is_workspace = user.lower().endswith(".com") and not user.lower().endswith("@gmail.com")

    hints = [
        "1) Use un App Password de Google (16 chars), no la password normal de la cuenta.",
        "2) Activa verificacion en 2 pasos para poder generar App Password.",
        "3) Verifica IMAP habilitado en Gmail/Workspace.",
    ]
    if is_workspace:
        hints.append("4) Si es Google Workspace, el admin debe permitir IMAP y App Password/OAuth para la cuenta.")

    if "authenticationfailed" in lower_raw or "invalid credentials" in lower_raw:
        raise RuntimeError(
            "Gmail IMAP rechazo autenticacion. "
            f"Cuenta: {user}. "
            + " ".join(hints)
        ) from exc
    raise RuntimeError(f"Error autenticando en Gmail IMAP para {user}: {raw}") from exc


def _load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _row_identity(row: dict[str, object]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        message_id = str(metadata.get("message_id", "")).strip()
        if message_id:
            return message_id
    chunk_id = str(row.get("chunk_id", "")).strip()
    return chunk_id


def _merge_rows(existing: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in existing + new_rows:
        if not isinstance(row, dict):
            continue
        key = _row_identity(row)
        if not key:
            continue
        merged[key] = row

    out: list[dict[str, object]] = list(merged.values())
    out.sort(
        key=lambda row: str(
            ((row.get("metadata") if isinstance(row.get("metadata"), dict) else {}) or {}).get("date", "")
        ),
        reverse=True,
    )
    return out


def sync_gmail_to_jsonl(
    *,
    gmail_user: str,
    gmail_password: str,
    imap_host: str,
    imap_port: int,
    mailbox: str,
    search_criterion: str,
    max_emails: int,
    max_body_chars: int,
    output_jsonl: str | Path,
    include_html_fallback: bool = False,
    offset: int = 0,
    append: bool = False,
) -> dict[str, Any]:
    """Sync one paged window of Gmail emails into JSONL, optionally appending."""
    user = (gmail_user or "").strip()
    password = (gmail_password or "").strip()
    if not user:
        raise ValueError("Missing Gmail user. Set GMAIL_USER or --gmail-user.")
    if not password:
        raise ValueError("Missing Gmail password. Set GMAIL_PASSWORD or --gmail-password.")
    if max_emails <= 0:
        raise ValueError("max_emails must be greater than 0")
    if max_body_chars < 200:
        raise ValueError("max_body_chars must be at least 200")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    search_tokens = shlex.split(search_criterion or "ALL")
    if not search_tokens:
        search_tokens = ["ALL"]

    client = imaplib.IMAP4_SSL(imap_host, imap_port)
    try:
        try:
            login_status, _ = client.login(user, password)
        except imaplib.IMAP4.error as exc:
            _raise_auth_error(user, exc)

        if login_status != "OK":
            raise RuntimeError("Gmail login failed")

        select_status, _ = client.select(mailbox)
        if select_status != "OK":
            raise RuntimeError(f"Could not select mailbox: {mailbox}")

        search_status, search_data = client.search(None, *search_tokens)
        if search_status != "OK":
            raise RuntimeError(f"Search failed for criterion: {search_criterion}")

        uid_list = search_data[0].split() if search_data else []
        total_matched = len(uid_list)
        window_end = max(0, total_matched - offset)
        window_start = max(0, window_end - max_emails)
        selected_uids = uid_list[window_start:window_end]

        rows: list[dict[str, object]] = []
        sender_counts: Counter[str] = Counter()

        for raw_uid in selected_uids:
            uid = raw_uid.decode("utf-8", errors="ignore")
            fetch_status, msg_data = client.fetch(raw_uid, "(RFC822)")
            if fetch_status != "OK" or not msg_data:
                continue

            raw_message = None
            for item in msg_data:
                if isinstance(item, tuple) and len(item) >= 2:
                    raw_message = item[1]
                    break
            if raw_message is None:
                continue

            message = BytesParser(policy=policy.default).parsebytes(raw_message)

            subject = _decode_header_value(str(message.get("Subject", "")))
            sender = _decode_header_value(str(message.get("From", "")))
            recipients = _decode_header_value(str(message.get("To", "")))
            raw_date = _decode_header_value(str(message.get("Date", "")))
            date_iso = _date_to_iso(raw_date)
            message_id = _decode_header_value(str(message.get("Message-ID", ""))) or f"uid-{uid}"

            body = _extract_text_body(message, include_html_fallback=include_html_fallback)
            if len(body) > max_body_chars:
                body = body[: max_body_chars].rstrip() + "..."

            snippet = body[:240]
            sender_counts[sender] += 1

            metadata = {
                "mailbox": mailbox,
                "uid": uid,
                "message_id": message_id,
                "from": sender,
                "to": recipients,
                "date": date_iso,
                "subject": subject,
                "snippet": snippet,
                "kind": "email",
            }

            row = {
                "chunk_id": f"gmail-{uid}",
                "source": f"gmail:{mailbox}",
                "text": _build_text_blob(subject, sender, recipients, date_iso, body),
                "metadata": metadata,
            }
            rows.append(row)

        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if append:
            existing_rows = _load_jsonl_rows(out_path)
            rows_to_write = _merge_rows(existing_rows, rows)
        else:
            rows_to_write = rows

        with out_path.open("w", encoding="utf-8") as f:
            for row in rows_to_write:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        return {
            "gmail_user": user,
            "mailbox": mailbox,
            "search_criterion": search_criterion,
            "messages_matched": total_matched,
            "messages_window_start": window_start,
            "messages_window_end": window_end,
            "messages_indexed_this_batch": len(rows),
            "messages_indexed_total_file": len(rows_to_write),
            "output_jsonl": str(out_path),
            "offset": offset,
            "max_emails": max_emails,
            "append": append,
            "top_senders": [{"sender": sender, "count": count} for sender, count in sender_counts.most_common(5)],
        }
    finally:
        try:
            client.logout()
        except Exception:
            pass


def main() -> None:
    """Sync Gmail messages from IMAP into a local JSONL chunk index."""
    args = parse_args()

    gmail_user = (args.gmail_user or "").strip() or (
        _env_value("GMAIL_USER", "gmail_user", "GAMAIL_USER", "gamail_user") or ""
    )
    gmail_password = (args.gmail_password or "").strip() or (
        _env_value("GMAIL_PASSWORD", "gmail_password", "GAMAIL_PASSWORD", "gamail_password") or ""
    )

    if not gmail_user:
        env_hint = ".env file not found in project root." if not Path(".env").exists() else "Check your .env variable names."
        raise ValueError(
            "Missing Gmail user. Set --gmail-user or GMAIL_USER (also accepted: gmail_user, GAMAIL_USER). "
            + env_hint
        )
    if not gmail_password:
        env_hint = ".env file not found in project root." if not Path(".env").exists() else "Check your .env variable names."
        raise ValueError(
            "Missing Gmail password. Set --gmail-password or GMAIL_PASSWORD (also accepted: gmail_password, GAMAIL_PASSWORD). "
            + env_hint
        )
    summary = sync_gmail_to_jsonl(
        gmail_user=gmail_user,
        gmail_password=gmail_password,
        imap_host=str(args.imap_host),
        imap_port=int(args.imap_port),
        mailbox=str(args.mailbox),
        search_criterion=str(args.search_criterion),
        max_emails=int(args.max_emails),
        max_body_chars=int(args.max_body_chars),
        output_jsonl=str(args.output_jsonl),
        include_html_fallback=bool(args.include_html_fallback),
        offset=0,
        append=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

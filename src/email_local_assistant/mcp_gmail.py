from __future__ import annotations

import json
import queue
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _parse_iso_date(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _read_json_message_lsp(stream) -> dict[str, Any] | None:
    content_length = -1

    while True:
        raw_line = stream.readline()
        if raw_line == b"":
            return None
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            break
        lower = line.lower()
        if lower.startswith("content-length:"):
            _, raw_len = line.split(":", 1)
            try:
                content_length = int(raw_len.strip())
            except Exception:
                content_length = -1

    if content_length <= 0:
        return None

    payload = stream.read(content_length)
    if not payload:
        return None
    try:
        decoded = json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if isinstance(decoded, dict):
        return decoded
    return None


class _StdioMcpClient:
    def __init__(
        self,
        command: str,
        startup_timeout_s: int = 20,
        request_timeout_s: int = 45,
        stdio_protocol: str = "auto",
    ):
        self.command = (command or "").strip()
        self.startup_timeout_s = max(5, int(startup_timeout_s))
        self.request_timeout_s = max(5, int(request_timeout_s))
        protocol = (stdio_protocol or "auto").strip().lower()
        if protocol not in {"auto", "lsp", "ndjson"}:
            protocol = "auto"
        self.stdio_protocol = protocol
        self._active_stdio_protocol = "lsp"
        self._proc: subprocess.Popen[bytes] | None = None
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader_error: Exception | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: list[str] = []
        self._next_id = 1
        self._write_lock = threading.Lock()

    def __enter__(self) -> _StdioMcpClient:
        if not self.command:
            raise ValueError("Missing MCP server command. Set --mcp-server-command or MCP_SERVER_COMMAND.")
        cmd = shlex.split(self.command)
        if not cmd:
            raise ValueError("Invalid MCP server command.")

        protocol_attempts = ["lsp", "ndjson"] if self.stdio_protocol == "auto" else [self.stdio_protocol]
        last_error: Exception | None = None
        for protocol in protocol_attempts:
            self._start_process(cmd, protocol=protocol)
            try:
                self.initialize()
                return self
            except Exception as exc:
                last_error = exc
                self._stop_process()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Could not initialize MCP process.")

    def _start_process(self, cmd: list[str], protocol: str) -> None:
        self._active_stdio_protocol = protocol
        self._messages = queue.Queue()
        self._reader_error = None
        self._stderr_lines = []
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self._stop_process()
            raise RuntimeError("Could not open MCP stdio pipes.")

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_process()

    def _stop_process(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = proc.stderr.readline()
            if line == b"":
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._stderr_lines.append(text)
                if len(self._stderr_lines) > 60:
                    self._stderr_lines = self._stderr_lines[-60:]

    def _reader_loop(self) -> None:
        try:
            proc = self._proc
            if proc is None or proc.stdout is None:
                return
            if self._active_stdio_protocol == "ndjson":
                while True:
                    raw_line = proc.stdout.readline()
                    if raw_line == b"":
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        decoded = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(decoded, dict):
                        self._messages.put(decoded)
            else:
                while True:
                    msg = _read_json_message_lsp(proc.stdout)
                    if msg is None:
                        break
                    self._messages.put(msg)
        except Exception as exc:
            self._reader_error = exc
        finally:
            self._messages.put(None)

    def _send_message(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("MCP process is not available.")
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        with self._write_lock:
            if self._active_stdio_protocol == "ndjson":
                proc.stdin.write(raw + b"\n")
            else:
                header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
                proc.stdin.write(header)
                proc.stdin.write(raw)
            proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send_message(payload)

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_s: int | None = None,
    ) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1

        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send_message(payload)

        timeout_value = self.request_timeout_s if timeout_s is None else max(1, int(timeout_s))
        deadline = time.monotonic() + timeout_value
        while time.monotonic() < deadline:
            if self._reader_error is not None:
                raise RuntimeError(f"MCP reader failed: {self._reader_error}") from self._reader_error

            remaining = max(0.1, deadline - time.monotonic())
            try:
                msg = self._messages.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is None:
                break
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            if str(msg_id) != str(request_id):
                continue

            if "error" in msg and isinstance(msg["error"], dict):
                err = msg["error"]
                code = err.get("code")
                text = str(err.get("message", "Unknown MCP error")).strip()
                stderr_tail = "\n".join(self._stderr_lines[-8:]).strip()
                if stderr_tail:
                    text = f"{text}\nMCP stderr:\n{stderr_tail}"
                raise RuntimeError(f"MCP request '{method}' failed (code={code}): {text}")

            result = msg.get("result")
            if isinstance(result, dict):
                return result
            return {"value": result}

        stderr_tail = "\n".join(self._stderr_lines[-8:]).strip()
        if stderr_tail:
            raise TimeoutError(f"MCP request timeout for '{method}'.\nMCP stderr:\n{stderr_tail}")
        raise TimeoutError(f"MCP request timeout for '{method}'.")

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "genai-gmail-local-assistant", "version": "0.1.0"},
            },
            timeout_s=self.startup_timeout_s,
        )
        self.notify("notifications/initialized", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})


def _extract_json_from_text(text: str) -> Any | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass

    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        try:
            return json.loads(raw[first_obj : last_obj + 1])
        except Exception:
            pass

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        try:
            return json.loads(raw[first_arr : last_arr + 1])
        except Exception:
            pass
    return None


def _extract_records_from_any(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("emails", "messages", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        email_like_keys = {"from", "sender", "subject", "date", "snippet", "body", "to"}
        if email_like_keys.intersection(set(payload.keys())):
            return [payload]
    return []


def _extract_records_from_tool_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    structured = result.get("structuredContent")
    records.extend(_extract_records_from_any(structured))
    if records:
        return records

    content = result.get("content")
    if not isinstance(content, list):
        return records

    for block in content:
        if not isinstance(block, dict):
            continue
        for key in ("json", "data"):
            parsed_records = _extract_records_from_any(block.get(key))
            if parsed_records:
                return parsed_records
        text = str(block.get("text", "")).strip()
        parsed = _extract_json_from_text(text)
        parsed_records = _extract_records_from_any(parsed)
        if parsed_records:
            return parsed_records
    return records


def _field(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            flattened = ", ".join([str(item).strip() for item in value if str(item).strip()])
            if flattened:
                return flattened
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_rows(records: list[dict[str, Any]], mailbox: str, max_body_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        sender = _field(record, ("from", "sender", "from_email", "fromAddress"))
        recipients = _field(record, ("to", "recipients", "to_email", "toAddress"))
        subject = _field(record, ("subject", "title"))
        date_iso = _field(record, ("date", "internalDate", "received_at", "timestamp"))
        message_id = _field(record, ("message_id", "messageId", "id", "threadId", "uid")) or f"mcp-{idx}"
        body = _field(record, ("body", "text", "plainText", "content", "snippet", "preview"))

        if len(body) > max_body_chars:
            body = body[: max_body_chars].rstrip() + "..."
        snippet = body[:240]

        if message_id in seen_ids:
            continue
        seen_ids.add(message_id)

        rows.append(
            {
                "chunk_id": f"mcp-{message_id}",
                "source": f"mcp-gmail:{mailbox}",
                "text": "\n".join(
                    [
                        f"Subject: {subject}",
                        f"From: {sender}",
                        f"To: {recipients}",
                        f"Date: {date_iso}",
                        "Body:",
                        body,
                    ]
                ).strip(),
                "metadata": {
                    "mailbox": mailbox,
                    "uid": message_id,
                    "message_id": message_id,
                    "from": sender,
                    "to": recipients,
                    "date": date_iso,
                    "subject": subject,
                    "snippet": snippet,
                    "kind": "email",
                },
            }
        )

    rows.sort(
        key=lambda row: _parse_iso_date(str((row.get("metadata") or {}).get("date", ""))),
        reverse=True,
    )
    return rows


def _merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing + incoming:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata")
        message_id = ""
        if isinstance(metadata, dict):
            message_id = str(metadata.get("message_id", "")).strip()
        key = message_id or str(row.get("chunk_id", "")).strip()
        if not key:
            continue
        merged[key] = row

    out = list(merged.values())
    out.sort(
        key=lambda row: _parse_iso_date(str(((row.get("metadata") if isinstance(row.get("metadata"), dict) else {}) or {}).get("date", ""))),
        reverse=True,
    )
    return out


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _mcp_query_from_search_criterion(mailbox: str, search_criterion: str) -> str:
    criterion = (search_criterion or "").strip()
    mailbox_normalized = (mailbox or "INBOX").strip().lower()

    if not criterion or criterion.upper() == "ALL":
        return "in:inbox" if mailbox_normalized == "inbox" else f"in:{mailbox_normalized}"
    if criterion.upper() == "UNSEEN":
        prefix = "in:inbox" if mailbox_normalized == "inbox" else f"in:{mailbox_normalized}"
        return f"{prefix} is:unread"
    return criterion


def _call_tool_with_candidates(client: _StdioMcpClient, tool_name: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for arguments in candidates:
        try:
            result = client.call_tool(tool_name, arguments)
            return result, arguments
        except Exception as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError(f"Could not call MCP tool '{tool_name}'. No argument candidates configured.")
    raise RuntimeError(f"Could not call MCP tool '{tool_name}' with available arguments: {last_error}") from last_error


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_accounts_from_any(payload: Any) -> list[str]:
    found: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return

        for list_key in ("accounts", "items", "results", "data"):
            value = node.get(list_key)
            if isinstance(value, list):
                for item in value:
                    _walk(item)

        for key in ("account", "name", "id", "alias"):
            value = node.get(key)
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    found.append(text)

        for value in node.values():
            if isinstance(value, (dict, list)):
                _walk(value)

    _walk(payload)
    return _dedupe_keep_order(found)


def _extract_accounts_from_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    lowered = raw.lower()
    if "no accounts configured" in lowered:
        return []

    matches: list[str] = []
    for pattern in [
        r"(?im)^\s*[-*]\s*(?:\*\*)?`?([A-Za-z0-9._:@-]{2,})`?(?:\*\*)?\s*$",
        r"(?im)^\s*\d+\.\s*(?:\*\*)?`?([A-Za-z0-9._:@-]{2,})`?(?:\*\*)?\s*$",
        r"(?im)\baccount\s*[:=]\s*`?([A-Za-z0-9._:@-]{2,})`?",
        r"(?im)\bname\s*[:=]\s*`?([A-Za-z0-9._:@-]{2,})`?",
        r"(?im)e\.g\.,\s*\"([A-Za-z0-9._:@-]{2,})\"",
    ]:
        for m in re.finditer(pattern, raw):
            token = m.group(1).strip()
            if token and token.lower() not in {"account", "accounts", "name"}:
                matches.append(token)
    return _dedupe_keep_order(matches)


def _extract_accounts_from_tool_result(result: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    structured = result.get("structuredContent")
    candidates.extend(_extract_accounts_from_any(structured))

    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            for key in ("json", "data"):
                candidates.extend(_extract_accounts_from_any(block.get(key)))
            text = str(block.get("text", "")).strip()
            parsed = _extract_json_from_text(text)
            candidates.extend(_extract_accounts_from_any(parsed))
            candidates.extend(_extract_accounts_from_text(text))

    return _dedupe_keep_order(candidates)


def _list_mcp_accounts(client: _StdioMcpClient) -> list[str]:
    try:
        result = client.call_tool("listAccounts", {})
    except Exception:
        return []
    return _extract_accounts_from_tool_result(result)


def _auto_detect_mcp_account(client: _StdioMcpClient) -> str:
    accounts = _list_mcp_accounts(client)
    if len(accounts) == 1:
        return accounts[0]
    return ""


def _with_account(arguments: dict[str, Any], account: str) -> dict[str, Any]:
    account_value = (account or "").strip()
    if not account_value:
        return dict(arguments)
    merged = dict(arguments)
    merged["account"] = account_value
    return merged


def sync_gmail_via_mcp(
    *,
    mcp_server_command: str,
    search_tool_name: str,
    mcp_account: str,
    mailbox: str,
    search_criterion: str,
    max_emails: int,
    max_body_chars: int,
    output_jsonl: str | Path,
    offset: int = 0,
    append: bool = False,
    startup_timeout_s: int = 20,
    request_timeout_s: int = 45,
) -> dict[str, Any]:
    if max_emails <= 0:
        raise ValueError("max_emails must be greater than 0")
    if max_body_chars < 200:
        raise ValueError("max_body_chars must be at least 200")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    mailbox_value = (mailbox or "INBOX").strip() or "INBOX"
    account_value = (mcp_account or "").strip()
    query = _mcp_query_from_search_criterion(mailbox_value, search_criterion)
    result_limit = max_emails + offset

    with _StdioMcpClient(
        command=mcp_server_command,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
    ) as client:
        if not account_value:
            auto_account = _auto_detect_mcp_account(client)
            if auto_account:
                account_value = auto_account

        search_candidates = [
            _with_account({"query": query, "maxResults": result_limit}, account_value),
            _with_account({"query": query, "max_results": result_limit}, account_value),
            _with_account({"query": query, "limit": result_limit}, account_value),
            _with_account({"query": query}, account_value),
        ]
        if not account_value:
            search_candidates.extend(
                [
                    {"query": query, "max_results": result_limit},
                    {"query": query, "limit": result_limit},
                    {"q": query, "limit": result_limit},
                    {"query": query},
                ]
            )

        try:
            result, used_args = _call_tool_with_candidates(
                client,
                (search_tool_name or "").strip() or "searchGmail",
                search_candidates,
            )
        except Exception as sync_exc:
            if not account_value:
                accounts = _list_mcp_accounts(client)
                if len(accounts) == 0:
                    raise RuntimeError(
                        "MCP Gmail search requires an account. "
                        "Run `npx -y google-workspace-mcp setup` and set `MCP_GMAIL_ACCOUNT` "
                        "(or pass `--mcp-account`)."
                    ) from sync_exc
                if len(accounts) > 1:
                    joined = ", ".join(accounts[:8])
                    raise RuntimeError(
                        f"Multiple MCP accounts detected ({joined}). "
                        "Set `MCP_GMAIL_ACCOUNT` or pass `--mcp-account`."
                    ) from sync_exc
            raise

    records = _extract_records_from_tool_result(result)
    normalized_rows = _normalize_rows(records, mailbox=mailbox_value, max_body_chars=max_body_chars)
    total_matched = len(normalized_rows)
    window_rows = normalized_rows[offset : offset + max_emails]

    out_path = Path(output_jsonl)
    if append:
        existing = _load_jsonl_rows(out_path)
        rows_to_write = _merge_rows(existing, window_rows)
    else:
        rows_to_write = window_rows
    _write_jsonl_rows(out_path, rows_to_write)

    messages_window_end = min(total_matched, offset + max_emails)
    messages_window_start = max(0, total_matched - messages_window_end)
    return {
        "transport": "mcp",
        "mailbox": mailbox_value,
        "search_criterion": search_criterion,
        "query_used": query,
        "mcp_search_tool": search_tool_name,
        "mcp_search_args": used_args,
        "mcp_account": account_value,
        "messages_matched": total_matched,
        "messages_window_start": messages_window_start,
        "messages_window_end": messages_window_end,
        "messages_indexed_this_batch": len(window_rows),
        "messages_indexed_total_file": len(rows_to_write),
        "output_jsonl": str(out_path),
        "offset": offset,
        "max_emails": max_emails,
        "append": append,
    }


def _render_tool_result(result: dict[str, Any]) -> str:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("message", "status", "result", "id"):
            value = structured.get(key)
            text = str(value).strip()
            if text:
                return text

    content = result.get("content")
    if isinstance(content, list):
        texts = [str(block.get("text", "")).strip() for block in content if isinstance(block, dict)]
        texts = [text for text in texts if text]
        if texts:
            return " | ".join(texts)
    return json.dumps(result, ensure_ascii=True)


def _find_first_scalar_by_keys(payload: Any, keys: tuple[str, ...]) -> str:
    wanted = {k.lower() for k in keys}

    def _walk(node: Any) -> str:
        if isinstance(node, dict):
            for key in keys:
                if key in node:
                    value = node.get(key)
                    if isinstance(value, (str, int, float)):
                        text = str(value).strip()
                        if text:
                            return text
            for key, value in node.items():
                if str(key).lower() in wanted and isinstance(value, (str, int, float)):
                    text = str(value).strip()
                    if text:
                        return text
                found = _walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found:
                    return found
        return ""

    return _walk(payload)


def _extract_draft_id_from_result(result: dict[str, Any]) -> str:
    def _from_text(raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        patterns = [
            r"(?im)^\s*Draft\s*ID\s*:\s*([A-Za-z0-9._-]+)\s*$",
            r"(?im)\bdraft[\s_-]*id\b\s*[:=]\s*([A-Za-z0-9._-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = str(match.group(1)).strip()
                if candidate:
                    return candidate
        return ""

    structured = result.get("structuredContent")
    draft_id = _find_first_scalar_by_keys(structured, ("draftId", "draft_id"))
    if draft_id:
        return draft_id
    draft_id = _from_text(str(structured) if structured is not None else "")
    if draft_id:
        return draft_id

    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text", "")).strip()
            draft_id = _from_text(text)
            if draft_id:
                return draft_id
            parsed = _extract_json_from_text(text)
            draft_id = _find_first_scalar_by_keys(parsed, ("draftId", "draft_id"))
            if draft_id:
                return draft_id

    # Last fallback for servers that only return a generic id field.
    fallback_id = _find_first_scalar_by_keys(structured, ("id",))
    if fallback_id:
        return fallback_id
    return ""


def send_email_via_mcp(
    *,
    mcp_server_command: str,
    send_tool_name: str,
    mcp_account: str,
    send_to: str,
    subject: str,
    body: str,
    dry_run: bool = False,
    startup_timeout_s: int = 20,
    request_timeout_s: int = 45,
) -> str:
    recipient = (send_to or "").strip().lower()
    account_value = (mcp_account or "").strip()
    tool_name = (send_tool_name or "").strip() or "sendGmailDraft"
    if not EMAIL_REGEX.fullmatch(recipient):
        raise ValueError(f"Invalid recipient email: {send_to}")
    if not body.strip():
        raise ValueError("Missing body for MCP send request.")

    if dry_run:
        return f"[dry-run] MCP email prepared for {recipient} with subject '{subject}' (tool={tool_name})."

    with _StdioMcpClient(
        command=mcp_server_command,
        startup_timeout_s=startup_timeout_s,
        request_timeout_s=request_timeout_s,
    ) as client:
        if not account_value:
            auto_account = _auto_detect_mcp_account(client)
            if auto_account:
                account_value = auto_account

        direct_candidates = [
            _with_account({"to": recipient, "subject": subject, "body": body}, account_value),
            _with_account({"to": [recipient], "subject": subject, "body": body}, account_value),
            _with_account({"recipient": recipient, "subject": subject, "body": body}, account_value),
            _with_account({"email": recipient, "subject": subject, "body": body}, account_value),
            _with_account({"to": recipient, "title": subject, "message": body}, account_value),
        ]
        try:
            result, used_args = _call_tool_with_candidates(client, tool_name, direct_candidates)
            rendered = _render_tool_result(result)
            return (
                f"Email sent via MCP tool '{tool_name}' to {recipient} with subject '{subject}'. "
                f"Result: {rendered} | args={json.dumps(used_args, ensure_ascii=True)}"
            )
        except Exception as send_exc:
            if tool_name != "sendGmailDraft":
                raise
            if not account_value:
                accounts = _list_mcp_accounts(client)
                if len(accounts) == 0:
                    raise RuntimeError(
                        "MCP send tool 'sendGmailDraft' requires an account. "
                        "Run `npx -y google-workspace-mcp setup` and set `MCP_GMAIL_ACCOUNT` "
                        "(or pass `--mcp-account`)."
                    ) from send_exc
                if len(accounts) > 1:
                    joined = ", ".join(accounts[:8])
                    raise RuntimeError(
                        f"Multiple MCP accounts detected ({joined}). "
                        "Set `MCP_GMAIL_ACCOUNT` or pass `--mcp-account`."
                    ) from send_exc
                raise RuntimeError(
                    "MCP send tool 'sendGmailDraft' requires an account. "
                    "Set `MCP_GMAIL_ACCOUNT` (or pass `--mcp-account`)."
                ) from send_exc

            draft_result, draft_args = _call_tool_with_candidates(
                client,
                "createGmailDraft",
                [
                    {"account": account_value, "to": recipient, "subject": subject, "body": body},
                    {"account": account_value, "to": [recipient], "subject": subject, "body": body},
                ],
            )
            draft_id = _extract_draft_id_from_result(draft_result)
            if not draft_id:
                rendered_draft = _render_tool_result(draft_result)
                raise RuntimeError(
                    f"Could not extract draftId from createGmailDraft result: {rendered_draft}"
                ) from send_exc

            sent_result, sent_args = _call_tool_with_candidates(
                client,
                "sendGmailDraft",
                [
                    {"account": account_value, "draftId": draft_id},
                    {"account": account_value, "draft_id": draft_id},
                    {"draftId": draft_id},
                ],
            )
            rendered_sent = _render_tool_result(sent_result)
            return (
                f"Email sent via MCP draft flow to {recipient} with subject '{subject}'. "
                f"draft_id={draft_id} | send_result={rendered_sent} | "
                f"create_args={json.dumps(draft_args, ensure_ascii=True)} | "
                f"send_args={json.dumps(sent_args, ensure_ascii=True)}"
            )

# Important Methods (Developer Guide)

This document explains the key methods in the local Gmail assistant and how they work together.

## 1) Query and chat engine (`src/email_local_assistant/ask_email.py`)

### `parse_args()`
- Loads `.env` automatically and builds CLI options for chat mode, local model settings, send controls, and MCP transport controls.
- Main groups: `--email-chunks`/`--knowledge-chunks`, `--prompt-artifact`/`--system-prompt-file`, `--hf-model-id`/`--hf-task`, send flags, sync flags, and MCP flags (`--mcp-server-command`, `--mcp-search-tool`, `--mcp-send-tool`, `--mcp-account`, timeouts).

### `_plan_query_with_llm(question, context, system_policy, generator)`
- First LLM planning stage.
- Produces a strict JSON retrieval plan (intent, topic terms, sender filter, and retrieval constraints).
- If LLM JSON is invalid, it performs one LLM-based JSON repair pass.
- If still invalid or schema-incompatible after repair, it raises an explicit runtime error.

### `_run_single_question(...)`
- Main search-answer pipeline for one question.
- Steps:
1. Build an LLM query plan.
2. Build retrieval query from plan terms and context.
3. Retrieve candidate chunks using TF-IDF (`TfidfRagRetriever`).
4. Keep top retrieved chunks (no hardcoded business filtering/re-ranking in code).
5. Build final grounded prompt (policy + plan + evidence + corpus stats).
6. Generate final answer with local HF model.
7. Return answer + evidence + send candidates for optional forwarding.

### `_sync_email_window(args, offset, append)` and `_load_retriever_and_rows(args)`
- `_sync_email_window(...)` delegates to MCP Gmail sync (`sync_gmail_via_mcp`).
- Supports pagination via `offset` and batch size.
- `_load_retriever_and_rows(...)` rebuilds the retriever from local JSONL after each sync page.

### `_plan_chat_action_with_llm(user_message, ...)`
- Second LLM planning stage used in interactive chat mode.
- Returns JSON action: `search`, `send_last`, `compose_send`, `help`, or `none`.
- If JSON is invalid, it performs one LLM-based JSON repair pass.
- If still invalid or schema-incompatible after repair, it raises an explicit runtime error.

### `_send_from_result(result, recipient, candidate_index, args, ...)`
- Builds outbound email payload from the last search result.
- Selects candidate email snippet and composes body.
- Sends through MCP (`send_email_via_mcp`) via `_send_email_via_transport(...)`.
- If `--send-dry-run` is enabled, no network send is performed.

### `_build_generator(task, model_id)` and `_generate_answer(...)`
- Creates local generator backend.
- Supports seq2seq models (for example FLAN/T5) via manual `generate()` path, and pipeline execution for causal models.
- Handles truncation and generation params to keep runtime stable.

### `main()`
- Entry point for chat-only mode.
- If no local email index exists, it automatically syncs the first batch from Gmail.
- During chat search, if no query hits are found, it automatically syncs the next batch (for example next 250) and retries once.
- Keeps state (`last_result`) and lets LLM decide per message if it is search or send.

## 2) MCP Gmail transport (`src/email_local_assistant/mcp_gmail.py`)

### `_StdioMcpClient`
- Minimal MCP stdio JSON-RPC client (`initialize`, `notifications/initialized`, `tools/call`).
- Launches MCP server command and auto-negotiates stdio framing:
  - `Content-Length` framed JSON-RPC
  - NDJSON (one JSON message per line)

### `sync_gmail_via_mcp(...)`
- Calls MCP search tool (default `searchGmail`).
- Normalizes returned messages to the same local JSONL email schema used by RAG.
- Supports local paging (`offset`) and append merge behavior.
- Supports optional MCP account injection (`mcp_account`) for servers that require explicit account selection.

### `send_email_via_mcp(...)`
- Calls MCP send tool (default `sendGmailDraft`) for outgoing email.
- If configured tool is `sendGmailDraft`, it performs draft flow:
  - `createGmailDraft` (with recipient/subject/body)
  - `sendGmailDraft` (with extracted `draftId`)

## 3) Local retriever (`src/email_local_assistant/rag_retriever.py`)

### `TfidfRagRetriever.from_jsonl_paths(paths)`
- Loads one or more JSONL chunk files and builds in-memory TF-IDF index.

### `TfidfRagRetriever.retrieve(query, top_k, min_score)`
- Scores chunks with TF-IDF overlap and returns ranked evidence objects.
- Used by `_run_single_question(...)` as first-stage recall.

## 4) Prompt optimization (`src/email_local_assistant/optimize_prompts.py`)

### `_preflight_ollama(args)`
- Validates local Ollama availability when compiler model is `ollama/...`.
- Fails fast with actionable setup instructions.

### `_extract_prompt_artifact(optimized_program, context)`
- Extracts optimized instructions + demos from DSPy program state.
- Writes `optimized_prompt.json` artifact for runtime auto-discovery.

### `main()`
- Loads dataset, builds train/dev splits, runs DSPy optimizer, evaluates baseline vs optimized, and stores artifacts.

## 5) Runtime flow summary

1. `run` starts chat mode and loads policy/model.
2. If local email chunks are missing, it syncs first batch via MCP.
3. LLM creates retrieval plan.
4. TF-IDF retriever returns evidence from local chunks.
5. If there are no hits, app syncs next batch and retries search.
6. LLM produces grounded answer.
7. In chat mode, LLM can route next user turn to `send_last` or `compose_send`.

## 6) Runtime strictness

The query planner and chat action planner are strict LLM-first components:
- They require valid JSON from the model.
- They enforce the expected schema.
- They can run one LLM-only JSON repair pass, then fail fast if still malformed.

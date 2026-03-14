# Important Methods (Developer Guide)

This document explains the key methods in the local Gmail assistant and how they work together.

## 1) Query and chat engine (`src/email_local_assistant/ask_email.py`)

### `parse_args()`
- Loads `.env` automatically and builds CLI options for email inputs, prompt policy, local model settings, and chat/send controls.
- Main groups: `--email-chunks`/`--knowledge-chunks`, `--prompt-artifact`/`--system-prompt-file`, `--hf-model-id`/`--hf-task`, and SMTP/send flags.

### `_plan_query_with_llm(question, context, system_policy, generator)`
- First LLM planning stage.
- Produces a strict JSON retrieval plan (intent, topic terms, sender filter, and retrieval constraints).
- If LLM JSON is invalid, it falls back to a lightweight heuristic plan.

### `_run_single_question(...)`
- Main search-answer pipeline for one question.
- Steps:
1. Build an LLM query plan.
2. Build retrieval query from plan terms and context.
3. Retrieve candidate chunks using TF-IDF (`TfidfRagRetriever`).
4. Filter/re-rank email chunks according to the plan.
5. Build final grounded prompt (policy + plan + evidence + corpus stats).
6. Generate final answer with local HF model.
7. Return answer + evidence + send candidates for optional forwarding.

### `_plan_chat_action_with_llm(user_message, ...)`
- Second LLM planning stage used in interactive chat mode.
- Returns JSON action: `search`, `send_last`, `help`, or `none`.
- If JSON is invalid, it uses a minimal fallback parser.

### `_send_from_result(result, recipient, candidate_index, args, ...)`
- Builds outbound email payload from the last search result.
- Selects candidate email snippet, composes body, and sends via SMTP.
- Uses `_send_email_via_smtp(...)`.
- If `--send-dry-run` is enabled, no network send is performed.

### `_build_generator(task, model_id)` and `_generate_answer(...)`
- Creates local generator backend.
- Supports seq2seq models (for example FLAN/T5) via manual `generate()` path, plus pipeline/fallback paths for causal models.
- Handles truncation and generation params to keep runtime stable.

### `main()`
- Entry point for both one-shot and chat modes.
- One-shot mode (`--question`) optionally supports immediate send via `--send-to`.
- Chat mode keeps state (`last_result`) and lets LLM decide per message if it is search or send.

## 2) Gmail sync (`src/email_local_assistant/gmail_sync.py`)

### `parse_args()`
- Reads CLI + `.env` and supports multiple env aliases for Gmail credentials.

### `_extract_text_body(message, include_html_fallback)`
- Extracts readable text from MIME emails.
- Prefers `text/plain`, optionally falls back to stripped `text/html`.
- Skips attachments.

### `main()`
- Connects to Gmail IMAP.
- Searches with provided IMAP criterion.
- Fetches messages, normalizes metadata and body.
- Writes local JSONL chunks (`data/gmail_chunks.jsonl` by default).

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

1. `sync-gmail` creates local JSONL email chunks.
2. `run` loads chunks, policy, and optimized prompt artifact (if available).
3. LLM creates retrieval plan.
4. TF-IDF retriever returns evidence.
5. LLM produces grounded answer.
6. In chat mode, LLM can route next user turn to `send_last`.

## 6) Why there are fallback methods

Fallback methods exist for reliability when a local model returns malformed JSON.  
Primary behavior remains LLM-first, but fallback logic prevents hard runtime failures in production use.

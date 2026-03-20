# Main Workflow

This document describes the core runtime flow of the local Gmail assistant.

## 1. Entry Point

1. The launcher starts with:
   - `python3 main.py run -- --mcp-server-command "npx -y google-workspace-mcp serve" --mcp-account "<account>"`
2. `main.py` forwards execution to:
   - `src.email_local_assistant.ask_email`

## 2. Runtime Initialization

1. `ask_email.main()` parses CLI + `.env` values.
2. It loads prompt configuration:
   - System policy from `config/email_llm_system_prompt.txt`
   - Prompt artifact (DSPy optimized prompt if available, otherwise built-in defaults)
3. It builds the local LLM generator (`_build_generator`) using the configured HF model/task.

## 3. Email Acquisition (Local Index)

1. If `data/gmail_chunks.jsonl` does not exist or is empty, runtime triggers sync:
   - `_sync_email_window(...)`
2. `_sync_email_window(...)` calls `sync_gmail_via_mcp(...)`.
3. MCP backend writes normalized JSONL chunks locally for retrieval.

## 4. Retriever Build (RAG)

1. `_load_retriever_and_rows(...)` loads local email rows from JSONL.
2. `TfidfRagRetriever.from_jsonl_paths(...)` builds the in-memory TF-IDF index from:
   - Email chunks
   - Optional knowledge chunks

## 5. Chat Turn Routing

For each user message:

1. `_plan_chat_action_with_llm(...)` asks the LLM to return a JSON action:
   - `search`, `send_last`, `compose_send`, `help`, or `none`
2. Runtime validates the JSON schema and executes the selected branch.

## 6. Search Flow (Prompt Injection + Retrieval)

When action is `search`:

1. `_plan_query_with_llm(...)` generates a structured JSON retrieval plan.
2. Runtime derives a retrieval query from that plan.
3. Retriever returns top candidate chunks (`retriever.retrieve(...)`).
4. `_build_prompt(...)` injects into the final generation prompt:
   - System policy
   - DSPy instructions and demos
   - LLM query plan
   - Retrieved evidence snippets
   - Corpus stats
5. `_generate_answer(...)` calls the local LLM to produce the grounded answer.
6. Runtime returns:
   - `answer`
   - `evidence`
   - `send_candidates`
   - `llm_plan`

## 7. Auto-Pagination on Zero Hits

1. If `query_hit_count == 0` and more history may exist:
   - Runtime syncs the next Gmail page (default batch: 250)
2. It rebuilds the retriever and retries the same search once.

## 8. Send Flows

1. `send_last`:
   - Sends based on the last search result candidate.
2. `compose_send`:
   - Sends a newly composed email using recipient + subject + body from action plan.
3. Actual send is executed by MCP tool call.
   - With `google-workspace-mcp` defaults, send flow is `createGmailDraft` -> `sendGmailDraft`.

## 9. Design Principle

Search behavior is LLM-first:

1. LLM decides chat action and query plan.
2. Code executes infrastructure concerns:
   - Email sync
   - Index/retrieval
   - Prompt assembly
   - MCP tool delivery

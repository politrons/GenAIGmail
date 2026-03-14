# Local Gmail Email Assistant

Local assistant to query Gmail data with:
- IMAP sync to a local JSONL index.
- RAG retrieval (TF-IDF) over emails + knowledge base.
- LLM-first reasoning:
  - LLM query planner for retrieval intent, topic and filters.
  - LLM answer generation grounded on retrieved evidence.
  - LLM chat action planner (`search` / `send_last`) in interactive mode.

Primary path for this project:
- No Ollama install required.
- No MLflow serving required.
- Run directly with local Hugging Face model + custom prompt policy file.

## Current stack

- Python launcher: `main.py`
- Gmail sync: `src/email_local_assistant/gmail_sync.py`
- Query engine: `src/email_local_assistant/ask_email.py`
- KB indexing: `src/email_local_assistant/build_rag_index.py`
- Retriever: `src/email_local_assistant/rag_retriever.py`
- DSPy optimization (optional): `src/email_local_assistant/optimize_prompts.py`

## Developer documentation

- Important methods guide: `docs/IMPORTANT_METHODS.md`

## Data status

The project no longer uses legacy SaaS datasets/prompts.
`data/` is email-focused:
- `data/gmail_chunks.jsonl` (output from Gmail sync)
- `data/email_assistant_qa.jsonl` (email DSPy dataset)
- `data/knowledge_base.md` (email playbook)
- `data/knowledge_base_chunks.jsonl` (KB chunks)

## Requirements

- Python 3.10+
- Gmail account with IMAP access
- If you use 2FA: Google App Password

Install:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create `.env` from template:

```bash
cp .env.example .env
```

2. Edit `.env` and set at least:

```bash
GMAIL_USER=your_account@gmail.com
GMAIL_PASSWORD=your_app_password_16_chars
```

Useful `.env` variables:
- `GMAIL_MAILBOX` (default `INBOX`)
- `GMAIL_SEARCH_CRITERION` (for example: `ALL`, `UNSEEN`, `SINCE 01-Mar-2026`)
- `GMAIL_MAX_EMAILS`
- `LOCAL_HF_MODEL_ID`
- `LOCAL_HF_TASK`
- `LOCAL_SYSTEM_PROMPT_FILE` (default: `config/email_llm_system_prompt.txt`)
- `SMTP_USER` / `SMTP_PASSWORD` (for outgoing email)
- `GMAIL_SMTP_HOST` / `GMAIL_SMTP_PORT` (default Gmail SMTP)
- `GMAIL_SEND_FROM` (optional sender override)

## Recommended flow

## 1) Sync Gmail to local index

```bash
python3 main.py sync-gmail -- \
  --mailbox "INBOX" \
  --search-criterion "ALL" \
  --max-emails 300 \
  --output-jsonl data/gmail_chunks.jsonl
```

Alternative with direct credentials:

```bash
python3 main.py sync-gmail -- \
  --gmail-user "your_account@gmail.com" \
  --gmail-password "your_app_password_16_chars" \
  --mailbox "INBOX" \
  --search-criterion "ALL" \
  --max-emails 300 \
  --output-jsonl data/gmail_chunks.jsonl
```

## 2) (Optional) Reindex knowledge base

```bash
python3 main.py index-kb -- \
  --input-paths data/knowledge_base.md \
  --output-jsonl data/knowledge_base_chunks.jsonl \
  --max-chars 900 \
  --overlap-chars 120
```

## 3) Run questions (single daily command)

The query engine injects a system prompt policy from:
`config/email_llm_system_prompt.txt`

Edit this file to define your custom behavior (topic filtering, latest-email rules, evidence style).

Prompt logic is automatic:
- If a DSPy optimized prompt exists (for example `artifacts/**/optimized_prompt.json`), it is used.
- If not, the built-in default prompt is used.
- No extra flag is required.

Search pattern logic is also LLM-first:
- The model builds a JSON retrieval plan (intent, topic terms, sender filters, latest/count flags).
- The retrieval and answer flow follows that plan.
- In chat mode, another LLM planner decides if the message is a search request or a send action.

Single query:

```bash
python3 main.py run -- \
  --question "How many emails mention Supersonic and what is the latest one?" \
  --email-chunks data/gmail_chunks.jsonl \
  --knowledge-chunks data/knowledge_base_chunks.jsonl \
  --hf-model-id "google/flan-t5-base" \
  --rag-top-k 6
```

This is the primary daily command:
- `python3 main.py run -- ...`
- It auto-loads DSPy optimized prompts if present; otherwise it uses the built-in prompt.

Interactive mode:

```bash
python3 main.py run -- \
  --chat \
  --email-chunks data/gmail_chunks.jsonl \
  --knowledge-chunks data/knowledge_base_chunks.jsonl
```

Chat send flow example:

1. Ask:
`find the latest email about Supersonic`

2. Then ask:
`send this email to someone@example.com`

You can test without sending:

```bash
python3 main.py run -- \
  --chat \
  --send-dry-run \
  --email-chunks data/gmail_chunks.jsonl
```

Single-shot search + send top candidate:

```bash
python3 main.py run -- \
  --question "Find latest email about Supersonic" \
  --email-chunks data/gmail_chunks.jsonl \
  --send-to "someone@example.com" \
  --send-dry-run
```

JSON output (useful for integrations):

```bash
python3 main.py run -- \
  --question "Summarize the 3 most recent Supersonic emails" \
  --email-chunks data/gmail_chunks.jsonl \
  --json-output
```

## Answering behavior

- The LLM first produces a structured plan for the query.
- The assistant uses retrieval + LLM with evidence (sender, subject, date).
- In interactive mode, action selection (`search` or `send_last`) is also LLM-driven.
- If evidence is insufficient, it should state that explicitly.

## Custom prompt policy (recommended)

File:
`config/email_llm_system_prompt.txt`

Example rule you can keep:
- "If the question asks for latest emails, return only latest emails that match the asked topic."

This lets you improve behavior without running DSPy optimization.

## DSPy (optional)

Use only if you want to refine prompt artifacts:

If you use `--compiler-model "ollama_chat/llama3.1:8b"`, Ollama must be installed and running:

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

```bash
python3 main.py optimize -- \
  --dataset data/email_assistant_qa.jsonl \
  --domain-context-file config/email_domain_context.txt \
  --output-dir artifacts/dspy_optimized \
  --compiler-model "ollama_chat/llama3.1:8b" \
  --auto light
```

`run` works even if you skip this step.

## Quick troubleshooting

- `Missing Gmail user`: `.env` is missing or `GMAIL_USER` is not set.
- `AUTHENTICATIONFAILED`: use App Password, not your normal account password.
- HF warning (`HF_TOKEN`): non-blocking, only affects download speed/rate limits.
- Slow model download on first run: expected; model is cached locally.

## Security

- Do not commit `.env`.
- Use App Password and rotate credentials when needed.
- `data/gmail_chunks.jsonl` contains sensitive email data.
- Use `--send-dry-run` before enabling real outgoing email.

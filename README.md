# Local Gmail Email Assistant (MCP Only)

Local chat assistant to search, read, and send Gmail emails using MCP + local LLM.

## Setup (one-time)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create your local config:

```bash
cp .env.example .env
```

3. Configure Google Workspace MCP account:

```bash
npx -y google-workspace-mcp setup
```

4. In `.env`, set at least:

```bash
MCP_GMAIL_ACCOUNT="work"
```

To see available account names:

```bash
npx -y google-workspace-mcp accounts list
```

## Run Chat

Use this single Python command:

```bash
python3 main.py run --
```

It uses `npx -y google-workspace-mcp serve` by default.

Inside chat you can do everything:
- Search latest emails by topic/pattern.
- Read matching emails from local indexed chunks.
- Ask follow-up searches.
- Send emails (MCP send flow).

Example chat prompts:
- `give me the latest email about invoices`
- `find emails from ariana about ASO`
- `send an email to politrons@gmail.com with subject hello_agent and body Welcome to the agent world`
- `send this email to politrons@gmail.com`

## Runtime behavior

- If local email chunks do not exist, the first batch is downloaded automatically.
- Searches run on local chunks (RAG).
- If no matches are found, it downloads the next batch and retries once.
- Transport is MCP-only for both search and send.

## DSPy Prompt Optimization (Optional)

If you want to generate optimized prompts with DSPy:

1. Install and run Ollama:

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

2. Run optimizer:

```bash
python3 -m src.email_local_assistant.optimize_prompts \
  --dataset data/email_assistant_qa.jsonl \
  --domain-context-file config/email_domain_context.txt \
  --output-dir artifacts/dspy_optimized \
  --compiler-model ollama_chat/llama3.1:8b \
  --auto light
```

At runtime, chat auto-loads `artifacts/dspy_optimized/optimized_prompt.json` if present.

## Security

- Do not commit `.env`.
- `data/gmail_chunks.jsonl` contains sensitive email content.

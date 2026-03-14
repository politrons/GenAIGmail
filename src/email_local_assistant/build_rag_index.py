from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TEXT_EXTENSIONS = {".md", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL chunk index for local RAG retrieval")
    parser.add_argument(
        "--input-paths",
        nargs="+",
        default=["data/knowledge_base.md"],
        help="Files or directories to index (.md/.txt)",
    )
    parser.add_argument("--output-jsonl", default="data/knowledge_base_chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap-chars", type=int, default=120)
    return parser.parse_args()


def _collect_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in _TEXT_EXTENSIONS:
            out.append(p)
            continue

        if p.is_dir():
            for item in sorted(p.rglob("*")):
                if item.is_file() and item.suffix.lower() in _TEXT_EXTENSIONS:
                    out.append(item)

    if not out:
        raise FileNotFoundError("No .md/.txt files found in --input-paths")
    return out


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _chunk_document(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        chunks.extend(_split_long_text(paragraph, max_chars=max_chars, overlap_chars=overlap_chars))

    if current:
        chunks.append(current)

    return chunks


def main() -> None:
    args = parse_args()

    if args.max_chars < 200:
        raise ValueError("--max-chars should be at least 200")
    if args.overlap_chars < 0:
        raise ValueError("--overlap-chars must be non-negative")
    if args.overlap_chars >= args.max_chars:
        raise ValueError("--overlap-chars must be smaller than --max-chars")

    files = _collect_files(args.input_paths)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        doc_chunks = _chunk_document(text, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
        for idx, chunk in enumerate(doc_chunks, start=1):
            rows.append(
                {
                    "chunk_id": f"{file.stem}-{idx:04d}",
                    "source": str(file),
                    "text": chunk,
                    "metadata": {
                        "kind": "knowledge_base",
                    },
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "files_indexed": len(files),
        "chunks_created": len(rows),
        "output_jsonl": str(out_path),
        "max_chars": args.max_chars,
        "overlap_chars": args.overlap_chars,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Helper to query Perch (via the CLI) for each test question in the Input
The Output contains Perch's responses in the JSON format needed for scoring

Usage:
  python app/test/eval/run_perch_cli_suite.py \
    --input app/test/eval/input/12_question_eval_set_4_26_26.json \
    --output app/test/eval/output/eval_input_5_08_26.json
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def _extract_answer(cli_stdout: str) -> str:
    """
    Parse answer text from perch.py stdout.
    """
    # Remove spinner carriage-return lines
    cleaned = cli_stdout.replace("\r", "\n")

    # Capture from "Perch:" up to sources block or next prompt/end.
    match = re.search(
        r"Perch:\s*(.*?)\n(?:--- SOURCES ---|You:|$)",
        cleaned,
        flags=re.DOTALL,
    )
    if not match:
        return ""
    return match.group(1).strip()


def _ask_perch(question: str, repo_root: Path) -> str:
    """
    Run perch CLI once with a single question and parse response.
    """
    cmd = [".venv/bin/python", "perch.py"]
    payload = f"{question}\nquit\n"

    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(repo_root),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"perch.py failed (code={proc.returncode}):\n{proc.stderr.strip()}"
        )

    answer = _extract_answer(proc.stdout)
    if not answer:
        raise RuntimeError("Could not parse Perch response from CLI output.")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Perch CLI eval suite.")
    parser.add_argument("--input", required=False, help="Questions JSON file (list[str]).")
    parser.add_argument(
        "--output",
        required=False,
        help="Output JSON file (list[{query, response}]).",
    )
    parser.add_argument(
        "--eval-id",
        required=False,
        help="Optional eval identifier (typically date, e.g., 5-10-26). If --output is omitted, writes to app/test/eval/input/eval_input_<eval-id>.json.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    if args.input:
        input_path = (repo_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    else:
        parser.error("Provide  --input")

    eval_id = args.eval_id

    if args.output:
      output_path = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    elif eval_id:
      output_path = repo_root / "app" / "test" / "eval" / "input" / f"eval_input_{eval_id}.json"
    else:
      parser.error("Provide either --output or --eval-id.")

    questions = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Input file must be a JSON list of question strings.")

    rows = []
    total = len(questions)
    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, str) or not q.strip():
            continue
        print(f"[{idx}/{total}] Querying Perch CLI...")
        answer = _ask_perch(q.strip(), repo_root)
        rows.append({"query": q.strip(), "response": answer})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Wrote {len(rows)} responses to {output_path}")


if __name__ == "__main__":
    main()

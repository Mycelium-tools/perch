# score.py
#
# Uses LLM as judge to evaluate Perch's response quality using a 4-dimension rubric,
#   evaluating: Source Specificity, Actionability, Advocacy Context, and Evidence Base.
#
# Example Usage
# 1. Run live scoring against the benchmark dataset
#       python score.py --score --input-eval-id 5-08-26
# 2. Parse the LLM justifications into a structured CSV
#       python score.py --parse --input-eval-id 5-08-26
# 4. Score AND parse results into CSV:
#       python score.py --score --parse --input-eval-id 5-08-26

import datetime
import os
import json
import re
import pandas as pd
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Configuration
DEFAULT_EVAL_ID = datetime.datetime.now().strftime("%-m-%d-%y_%H%M")
INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'

RUBRIC_PROMPT = """
   You are evaluating responses from an AI system designed to help animal advocacy organizations.

    For each response, score it on 4 dimensions using this rubric:

    RUBRIC:

    1. Source Specificity (0-3)
    - 0: No sources mentioned
    - 1: Vague reference ("research shows") but no specific document
    - 2: Names a source/study but lacks detail
    - 3: Cites specific source with detail (author, year, document name)

    2. Actionability (0-3)
    - 0: Only general advice; no specifics
    - 1: One specific example/step mentioned, but vague
    - 2: 2-3 concrete examples/steps with some detail
    - 3: 3+ concrete, ready-to-use examples or steps with measurable targets/timelines

    3. Advocacy Context (0-3)
    - 0: Advice that could apply to any cause
    - 1: References advocacy but generic
    - 2: Specific to animal/food advocacy but not tailored to constraints
    - 3: Acknowledges real barriers and addresses them

    4. Evidence Base (0-3)
    - 0: Contradicts known evidence or unsupported claims
    - 1: General claims without evidence
    - 2: References research but incompletely
    - 3: Grounds claims in cited evidence; acknowledges uncertainty

    TASK:

    Query: {QUERY}

    Response: {RESPONSE}

    Score this response on each dimension (0-3). Then provide:
    1. Individual scores for each dimension
    2. Mean score (average of 4 dimensions)
    3. Justification (1-2 sentences per dimension explaining the score)

    Format your response as:
    Source Specificity: [0-3]
    Justification: [sentence]

    Actionability: [0-3]
    Justification: [sentence]

    Advocacy Context: [0-3]
    Justification: [sentence]

    Evidence Base: [0-3]
    Justification: [sentence]

    Mean Score: [X.XX]
"""

def get_judge():
    return ChatOpenAI(model_name="gpt-5-mini", temperature=0.0)

def build_paths(eval_id: str):
    input_file = INPUT_DIR + f'eval_input_{eval_id}.json'
    eval_results_file = os.path.join(OUTPUT_DIR, f'eval_results_{eval_id}.json')
    parsed_csv_file = os.path.join(OUTPUT_DIR, f'parsed_results_{eval_id}.csv')
    return input_file, eval_results_file, parsed_csv_file

def run_scoring(input_file: str, eval_results_file: str):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    model = get_judge()
    results = []

    for case in benchmark_data:
        prompt = RUBRIC_PROMPT.format(QUERY=case["query"], RESPONSE=case["response"])
        response = model.invoke([{"role": "user", "content": prompt}])
        
        results.append({"query": case["query"], "scores": response.content})
        mean_match = re.search(r"Mean Score:\s*([\d.]+)", response.content or "")
        mean_display = mean_match.group(1) if mean_match else "n/a"
        print(f"Scored: {case['query'][:50]}... (mean={mean_display})")

    with open(eval_results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Raw results saved to {eval_results_file}")

def run_parsing(eval_results_file: str, parsed_csv_file: str):
    if not os.path.exists(eval_results_file):
        print(f"Error: {eval_results_file} not found. Run --score first.")
        return

    with open(eval_results_file, 'r') as f:
        data = json.load(f)

    rows = []
    for entry in data:
        q = entry.get("query", "No Query")
        s = entry.get("scores", "")
        
        # Metrics extraction
        pattern = r"(.*?): (\d+)\s*Justification: (.*?)(?=\n\n|\n[A-Z]|$)"
        matches = re.findall(pattern, s, re.DOTALL)
        for m in matches:
            rows.append({
                "query": q, "metric": m[0].strip(), 
                "score": float(m[1]), "justification": m[2].strip()
            })
            
        # Mean score extraction
        mean_match = re.search(r"Mean Score:\s*([\d.]+)", s)
        if mean_match:
            rows.append({
                "query": q, "metric": "OVERALL MEAN",
                "score": float(mean_match.group(1)),
                "justification": "Aggregated average"
            })

    df = pd.DataFrame(rows)
    df.to_csv(parsed_csv_file, index=False)
    print(f"Done. CSV saved to {parsed_csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge CLI")
    parser.add_argument("--score", action="store_true", help="Run the LLM judge on input queries")
    parser.add_argument("--parse", action="store_true", help="Parse raw JSON scores into CSV")
    parser.add_argument(
        "--eval-id",
        default=DEFAULT_EVAL_ID,
        help=f"Eval identifier used for output/result filenames (default: {DEFAULT_EVAL_ID})",
    )
    parser.add_argument(
        "--input-eval-id",
        default=None,
        help="Optional separate eval id for the input file. If omitted, uses --eval-id.",
    )
    
    args = parser.parse_args()
    input_eval_id = args.input_eval_id or args.eval_id
    input_file, _, _ = build_paths(input_eval_id)
    _, eval_results_file, parsed_csv_file = build_paths(args.eval_id)

    if args.score:
        run_scoring(input_file, eval_results_file)
    if args.parse:
        run_parsing(eval_results_file, parsed_csv_file)
    if not (args.score or args.parse):
        parser.print_help()

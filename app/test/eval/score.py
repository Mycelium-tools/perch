# score.py
#
# Uses LLM as judge to evaluate Perch's response quality using a 4-dimension rubric,
#   evaluating: Source Specificity, Actionability, Advocacy Context, and Evidence Base.
#
# Example Usage
# 1. Run live scoring against the benchmark dataset
#       python score.py --score
# 2. Parse the LLM justifications into a structured CSV
#       python score.py --parse
# 4. Score AND parse results into CSV:
#       python score.py --score --parse

import os
import json
import re
import pandas as pd
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Configuration
EVAL_ID = "4-28-26" # use eval date as its ID
INPUT_DIR = 'input/'
INPUT_FILE = INPUT_DIR + f'eval_input_{EVAL_ID}.json'
OUTPUT_DIR = 'output/'
EVAL_RESULTS_FILE = os.path.join(OUTPUT_DIR, f'eval_results_{EVAL_ID}.json')
PARSED_CSV_FILE = os.path.join(OUTPUT_DIR, f'parsed_results_{EVAL_ID}.csv')

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

def run_scoring():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    model = get_judge()
    results = []

    for case in benchmark_data:
        prompt = RUBRIC_PROMPT.format(QUERY=case["query"], RESPONSE=case["response"])
        response = model.invoke([{"role": "user", "content": prompt}])
        
        results.append({"query": case["query"], "scores": response.content})
        print(f"Scored: {case['query'][:50]}...")

    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Raw results saved to {EVAL_RESULTS_FILE}")

def run_parsing():
    if not os.path.exists(EVAL_RESULTS_FILE):
        print(f"Error: {EVAL_RESULTS_FILE} not found. Run --score first.")
        return

    with open(EVAL_RESULTS_FILE, 'r') as f:
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
    df.to_csv(PARSED_CSV_FILE, index=False)
    print(f"Done. CSV saved to {PARSED_CSV_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge CLI")
    parser.add_argument("--score", action="store_true", help="Run the LLM judge on input queries")
    parser.add_argument("--parse", action="store_true", help="Parse raw JSON scores into CSV")
    
    args = parser.parse_args()

    if args.score:
        run_scoring()
    if args.parse:
        run_parsing()
    if not (args.score or args.parse):
        parser.print_help()
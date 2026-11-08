"""Offline retrieval smoke evaluation; optional model and judge checks incur API calls."""
import argparse
import json
import os
from pathlib import Path
from simple_rag import RAGAgent, TfidfRetriever, load_documents
from simple_rag.models import OpenAIModel
from judge import LLMJudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Generate answers with this model")
    parser.add_argument("--judge-model", help="Also grade answers with this model; requires --model")
    args = parser.parse_args()
    if args.judge_model and not args.model:
        parser.error("--judge-model requires --model")
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evaluation/cases.json").read_text())
    retriever = TfidfRetriever(load_documents(root / "examples/documents", ["handbook.md"]))
    base_url = os.getenv("OPENAI_BASE_URL")
    agent = RAGAgent(retriever, OpenAIModel(args.model, base_url)) if args.model else None
    judge = LLMJudge(args.judge_model, base_url) if args.judge_model else None
    retrieval_pass = answer_pass = answer_total = judge_pass = judge_total = 0
    for case in cases:
        chunks = retriever.search(case["question"])
        expected = case["expected_source"]
        ok = any(c.source == expected for c in chunks) if expected else not chunks
        retrieval_pass += ok
        print(f"retrieval={'PASS' if ok else 'FAIL'}: {case['question']}")
        if agent:
            result = agent.ask(case["question"])
            if case["answer_contains"]:
                answer_total += 1
                answer_ok = case["answer_contains"].lower() in result.text.lower()
                answer_pass += answer_ok
                print(f"  answer substring={'PASS' if answer_ok else 'FAIL'}")
            if judge:
                rating = judge.score(case["question"], case["reference_answer"], result.text, result.sources)
                judge_total += 1
                judge_pass += rating["correctness"] and rating["groundedness"]
                print(f"  judge correctness={rating['correctness']} groundedness={rating['groundedness']}: {rating['reason']}")
    print(f"Retrieval: {retrieval_pass}/{len(cases)}")
    if agent:
        print(f"Answer substring: {answer_pass}/{answer_total}")
    if judge:
        print(f"Judge (both criteria): {judge_pass}/{judge_total}")
    if retrieval_pass != len(cases) or answer_pass != answer_total or judge_pass != judge_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

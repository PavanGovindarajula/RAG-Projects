import argparse
import os
from pathlib import Path
from .agent import RAGAgent
from .documents import load_documents
from .models import OpenAIModel
from .retrieval import TfidfRetriever


def main():
    parser = argparse.ArgumentParser(description="Ask a question about explicitly configured documents")
    parser.add_argument("question")
    parser.add_argument("--root", type=Path, required=True, help="Trusted document directory")
    parser.add_argument("--file", action="append", required=True, help="Relative document path; repeat for more")
    parser.add_argument("--model", default=os.getenv("RAG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    docs = load_documents(args.root, args.file)
    agent = RAGAgent(TfidfRetriever(docs), OpenAIModel(args.model, os.getenv("OPENAI_BASE_URL")), args.top_k)
    answer = agent.ask(args.question)
    print(answer.text)
    for chunk in answer.sources:
        print(f"  [{chunk.source}, chunk {chunk.index}, score {chunk.score:.3f}]")


if __name__ == "__main__":
    main()

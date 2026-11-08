"""Core interfaces and orchestration. No network calls until a model is invoked."""
from dataclasses import dataclass
from typing import Protocol
from .retrieval import Chunk


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[Chunk]: ...


class AnswerModel(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> str: ...


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Chunk]


class RAGAgent:
    def __init__(self, retriever: Retriever, model: AnswerModel, top_k: int = 3):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever, self.model, self.top_k = retriever, model, top_k

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question or len(question) > 4000:
            raise ValueError("Question must contain 1 to 4000 characters")
        chunks = self.retriever.search(question, self.top_k)
        if not chunks:
            return Answer("I don't know based on the configured documents.", [])
        return Answer(self.model.answer(question, chunks), chunks)

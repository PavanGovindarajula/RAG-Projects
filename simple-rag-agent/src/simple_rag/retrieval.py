"""Deterministic TF-IDF retrieval with overlapping word chunks."""
from collections import Counter
from dataclasses import dataclass
import math
import re
from .documents import Document

TOKEN = re.compile(r"\b\w+\b", re.UNICODE)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str
    score: float = 0.0


class TfidfRetriever:
    def __init__(self, documents: list[Document], chunk_words: int = 180, overlap: int = 40):
        if chunk_words <= 0 or not 0 <= overlap < chunk_words:
            raise ValueError("Require chunk_words > 0 and 0 <= overlap < chunk_words")
        self.chunks: list[Chunk] = []
        for doc in documents:
            words = doc.text.split()
            for start in range(0, len(words), chunk_words - overlap):
                self.chunks.append(Chunk(doc.source, len([c for c in self.chunks if c.source == doc.source]), " ".join(words[start:start + chunk_words])))
                if start + chunk_words >= len(words):
                    break
        counts = [Counter(tokens(c.text)) for c in self.chunks]
        df = Counter(word for count in counts for word in count)
        n = len(counts)
        self.idf = {word: math.log((n + 1) / (freq + 1)) + 1 for word, freq in df.items()}
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, count: Counter) -> dict[str, float]:
        weighted = {w: (1 + math.log(n)) * self.idf[w] for w, n in count.items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in weighted.values()))
        return {w: v / norm for w, v in weighted.items()} if norm else {}

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        q = self._vector(Counter(tokens(query)))
        scored = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(weight * vector.get(word, 0.0) for word, weight in q.items())
            if score > 0:
                scored.append(Chunk(chunk.source, chunk.index, chunk.text, score))
        return sorted(scored, key=lambda c: (-c.score, c.source, c.index))[:top_k]

import hashlib
import re
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(directory):
    chunks = []
    for path in sorted(Path(directory).glob('*.md')):
        # Each heading is a small, independently retrievable support article.
        for part in re.split(r'(?=^## )', path.read_text(encoding='utf-8'), flags=re.M):
            text = part.strip()
            if text and text.startswith('## '):
                identifier = hashlib.sha256((path.name + '\0' + text).encode()).hexdigest()
                chunks.append({'id': identifier, 'source': path.name, 'text': text})
    return chunks


def fuse(keyword, semantic, limit=20, k=60):
    """Reciprocal rank fusion; each list contains IDs in descending relevance."""
    scores = {}
    for ranking in (keyword, semantic):
        for rank, identifier in enumerate(dict.fromkeys(ranking), 1):
            scores[identifier] = scores.get(identifier, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda x: (-scores[x], x))[:limit]


class HybridRetriever:
    def __init__(self, directory, database, embedding_model='sentence-transformers/all-MiniLM-L6-v2', reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.chunks = load_chunks(directory)
        if not self.chunks:
            raise ValueError('No Markdown sections found in document directory')
        self.by_id = {c['id']: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokens(c['text']) for c in self.chunks])
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.client = chromadb.PersistentClient(path=str(database))
        self.collection = self.client.get_or_create_collection('support', metadata={'hnsw:space': 'cosine'})
        self.embedding_model = embedding_model

    def rebuild(self):
        # Recreate the collection so removed and changed documents cannot remain searchable.
        self.client.delete_collection('support')
        self.collection = self.client.create_collection('support', metadata={'hnsw:space': 'cosine', 'embedding_model': self.embedding_model})
        embeddings = self.embedder.encode([c['text'] for c in self.chunks], normalize_embeddings=True).tolist()
        self.collection.add(ids=[c['id'] for c in self.chunks], embeddings=embeddings, documents=[c['text'] for c in self.chunks], metadatas=[{'source': c['source']} for c in self.chunks])

    def search(self, query, top_k=3, candidates=20):
        if not query.strip() or top_k < 1 or candidates < 1:
            return []
        indexed = set(self.collection.get(include=[])['ids'])
        if indexed != set(self.by_id) or (self.collection.metadata or {}).get('embedding_model') != self.embedding_model:
            raise ValueError('Index missing or stale; run the index command first')
        scores = self.bm25.get_scores(tokens(query))
        keyword = [self.chunks[i]['id'] for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if scores[i] > 0][:candidates]
        embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        semantic = self.collection.query(query_embeddings=embedding, n_results=min(candidates, len(self.chunks)))['ids'][0]
        ids = fuse(keyword, semantic, candidates)
        rerank_scores = self.reranker.predict([(query, self.by_id[i]['text']) for i in ids])
        ranked = sorted(zip(ids, rerank_scores), key=lambda pair: -float(pair[1]))[:top_k]
        return [{**self.by_id[i], 'score': float(score)} for i, score in ranked]

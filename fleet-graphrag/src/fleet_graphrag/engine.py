import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r'[a-z0-9]+')
ALIASES = {'redis pool exhaustion': 'cache exhaustion', 'cache pool saturation': 'cache exhaustion', 'expired certificate': 'certificate expiration'}
STOP = {'the','a','an','and','of','in','to','for','what','which','is','are','across','service','services','show','me','did','how','why','about','with','incidents','incident','failures','failure','themes','fleet'}

def words(text):
    return [x for x in TOKEN.findall(text.lower()) if x not in STOP]

def canonical(value):
    value = ' '.join(value.lower().strip().split())
    return ALIASES.get(value, value)

def vector(text, dimensions=256):
    # Deterministic hashed bag-of-words: a local lexical vector, not a semantic embedding.
    counts = Counter(words(text))
    out = [0.0] * dimensions
    for word, count in counts.items():
        slot = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'big') % dimensions
        out[slot] += count
    norm = math.sqrt(sum(x*x for x in out)) or 1
    return [x / norm for x in out]

def cosine(a, b):
    return sum(x*y for x, y in zip(a, b))

def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db

def build_index(source, target, extractor=None, embedding_model=None):
    """Build a complete replacement in a temporary file; atomic replace on success."""
    records = json.loads(Path(source).read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Dataset must be a nonempty list')
    if extractor is not None:
        records = [{**r, 'facts': extractor(r['text'])} for r in records]
    ids = set()
    for record in records:
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) and record[k].strip() for k in ('id', 'access', 'text')) or record['id'] in ids:
            raise ValueError('Each source needs a unique id, text, and access tag')
        ids.add(record['id'])
        if not isinstance(record.get('facts'), list) or any(not isinstance(f, list) or len(f) != 3 or not all(isinstance(s, str) and s.strip() for s in f) for f in record['facts']):
            raise ValueError('facts must be a list of nonempty string triples')
    embedder = None
    if embedding_model:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding_model)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.fleet-index-', suffix='.sqlite', dir=target.parent)
    os.close(fd)
    try:
        with connect(tmp) as db:
            db.executescript('''CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE chunks (id TEXT PRIMARY KEY, source_id TEXT, access TEXT, text TEXT, vector TEXT);
            CREATE TABLE edges (subject TEXT, relation TEXT, object TEXT, chunk_id TEXT, access TEXT);
            CREATE INDEX edges_subject ON edges(subject); CREATE INDEX edges_object ON edges(object);''')
            db.execute('INSERT INTO metadata VALUES (?,?)', ('embedding_model', embedding_model or 'hashed-lexical-v1'))
            for record in records:
                # One short incident per chunk, preserving the source identifier.
                chunk_id = record['id'] + '#0'
                db.execute('INSERT INTO chunks VALUES (?,?,?,?,?)', (chunk_id, record['id'], record['access'], record['text'], json.dumps(embedder.encode(record['text'], normalize_embeddings=True).tolist() if embedder else vector(record['text']))))
                for subject, relation, obj in record['facts']:
                    db.execute('INSERT INTO edges VALUES (?,?,?,?,?)', (canonical(subject), canonical(relation), canonical(obj), chunk_id, record['access']))
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_authorized(path, tags):
    if not Path(path).is_file(): raise FileNotFoundError('Index not found; run index first')
    if not tags: return [], [], 'hashed-lexical-v1'
    placeholders = ','.join('?' for _ in tags)
    with connect(path) as db:
        model = db.execute("SELECT value FROM metadata WHERE key='embedding_model'").fetchone()['value']
        chunks = [dict(x) for x in db.execute(f'SELECT * FROM chunks WHERE access IN ({placeholders})', tuple(tags))]
        edges = [dict(x) for x in db.execute(f'SELECT * FROM edges WHERE access IN ({placeholders})', tuple(tags))]
    return chunks, edges, model

def route(question):
    q = question.lower()
    return 'global' if any(x in q for x in ('theme', 'systemic', 'across', 'fleet', 'overall', 'common', 'trend')) else 'local'

def search(path, question, tags, mode='auto', limit=5):
    start = time.perf_counter()
    if not question.strip() or limit < 1: raise ValueError('Question and positive limit required')
    if mode not in ('auto', 'global', 'local'): raise ValueError('Invalid search mode')
    chunks, edges, embedding_model = load_authorized(path, tags)
    mode = route(question) if mode == 'auto' else mode
    by_id = {c['id']: c for c in chunks}
    if mode == 'global':
        # Communities are grouped by failure theme, restricted to the authorized subgraph.
        grouped = defaultdict(list)
        for e in edges:
            if e['relation'] == 'affected_by': grouped[e['object']].append(e)
        themes = []
        for theme, group in grouped.items():
            services = sorted({e['subject'] for e in group})
            sources = sorted({by_id[e['chunk_id']]['source_id'] for e in group if e['chunk_id'] in by_id})
            if len(services) >= 2 and len(sources) >= 2:
                themes.append({'theme': theme, 'services': services, 'sources': sources})
        themes.sort(key=lambda x: (-len(x['services']), -len(x['sources']), x['theme']))
        themes = themes[:limit]
        answer = '\n'.join(f"{t['theme']}: {', '.join(t['services'])} [{', '.join(t['sources'])}]" for t in themes)
        evidence = sorted({s for t in themes for s in t['sources']})
        details = themes
    else:
        # Resolve entities only within authorized edges; retrieve neighboring source chunks.
        normalized_question = canonical(question)
        entities = {e[k] for e in edges for k in ('subject', 'object') if re.search(r'(?<!\w)' + re.escape(e[k]) + r'(?!\w)', normalized_question.lower())}
        linked = [e for e in edges if e['subject'] in entities or e['object'] in entities]
        # Explicit entity lookup must not bring in unrelated text-similar sources.
        candidates = {e['chunk_id'] for e in linked} if entities else set(by_id)
        qw = set(words(question))
        if embedding_model == 'hashed-lexical-v1':
            qv = vector(question)
        else:
            from sentence_transformers import SentenceTransformer
            qv = SentenceTransformer(embedding_model).encode(question, normalize_embeddings=True).tolist()
        ranked = []
        for cid in candidates:
            c = by_id.get(cid)
            if not c: continue
            overlap = len(qw & set(words(c['text'])))
            score = overlap + cosine(qv, json.loads(c['vector']))
            if entities or overlap or (embedding_model != 'hashed-lexical-v1' and cosine(qv, json.loads(c['vector'])) >= 0.4): ranked.append((score, cid))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        chosen = [by_id[cid] for _, cid in ranked[:limit]]
        answer = '\n'.join(f"{c['text']} [{c['source_id']}]" for c in chosen)
        evidence = [c['source_id'] for c in chosen]
        details = [{'source': c['source_id'], 'text': c['text']} for c in chosen]
    return {'mode': mode, 'answer': answer or 'Insufficient accessible evidence to answer.', 'abstained': not bool(answer), 'sources': evidence, 'details': details, 'trace': {'authorized_chunks': len(chunks), 'elapsed_ms': round((time.perf_counter()-start)*1000, 2), 'model_calls': 0, 'estimated_model_cost_usd': 0.0}}

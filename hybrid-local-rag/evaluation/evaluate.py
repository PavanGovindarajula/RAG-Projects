import argparse
import json
from pathlib import Path
from hybrid_rag.retrieval import HybridRetriever

p = argparse.ArgumentParser()
p.add_argument('--documents', default='data')
p.add_argument('--db', default='.chroma')
p.add_argument('--k', type=int, default=3)
a = p.parse_args()
r = HybridRetriever(a.documents, a.db)
cases = json.loads((Path(__file__).parent / 'cases.json').read_text())
hits = 0
for case in cases:
    results = r.search(case['question'], top_k=a.k)
    hit = any(case['expected'] in result['text'].splitlines()[0] for result in results)
    hits += hit
    print(f"{'PASS' if hit else 'FAIL'}: {case['question']}")
print(f'Recall@{a.k}: {hits}/{len(cases)} = {hits / len(cases):.0%}')

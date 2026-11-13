import json
import tempfile
from pathlib import Path
from fleet_graphrag.engine import build_index, search

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / 'evaluation/cases.json').read_text())
with tempfile.TemporaryDirectory() as directory:
    db = Path(directory) / 'eval.sqlite'
    build_index(root / 'data/incidents.json', db)
    hits = 0
    for c in cases:
        r = search(db, c['question'], c['tags'])
        themes = {t['theme'] for t in r['details']} if r['mode'] == 'global' else set()
        passed = r['mode'] == c['mode'] and set(c.get('expected_sources', [])) <= set(r['sources']) and set(c.get('expected_themes', [])) <= themes and not set(c.get('forbidden_sources', [])) & set(r['sources']) and r['abstained'] == c.get('abstain', False)
        hits += passed
        print(f"{'PASS' if passed else 'FAIL'}: {c['question']} tags={c['tags']}")
    print(f'Cases passed: {hits}/{len(cases)}')
    if hits != len(cases): raise SystemExit(1)

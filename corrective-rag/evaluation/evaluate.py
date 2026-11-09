import json
from pathlib import Path
from crag.core import Evidence, Issue, Triage, retrieve_local
from crag.adapters import DemoEvaluator, DemoGenerator
from crag.cli import FixtureGateway

root = Path(__file__).resolve().parents[1]
docs = [Evidence(**e) for e in json.loads((root / 'data/local.json').read_text())]
cases = json.loads((root / 'data/evaluation.json').read_text())
triage = Triage(DemoEvaluator(), FixtureGateway(root / 'data/gateway.json'), DemoGenerator())
hits = 0
for case in cases:
    issue = Issue(case['issue'], case['version'])
    result = triage.run(issue, retrieve_local(issue, docs))
    ok = result.route == case['expected_route'] and (case['expected_evidence'] is None or case['expected_evidence'] in [e.id for e in result.evidence])
    hits += ok
    print(f"{'PASS' if ok else 'FAIL'}: {issue.text} v{issue.version} -> {result.route}")
print(f'Route and evidence accuracy: {hits}/{len(cases)}')
if hits != len(cases):
    raise SystemExit(1)

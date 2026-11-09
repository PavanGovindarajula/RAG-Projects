import argparse
import json
from pathlib import Path
from .core import Evidence, Issue, Triage, retrieve_local
from .adapters import DemoEvaluator, DemoGenerator, MCPGateway, ModelEvaluator, ModelGenerator

class FixtureGateway:
    def __init__(self, path):
        self.items = [Evidence(**x) for x in json.loads(Path(path).read_text())]
    def retrieve(self, issue):
        return retrieve_local(issue, self.items)

def main():
    p = argparse.ArgumentParser(description='Corrective RAG incident triage')
    p.add_argument('issue')
    p.add_argument('--version', required=True)
    p.add_argument('--local', default='data/local.json')
    p.add_argument('--gateway-fixture', default='data/gateway.json')
    p.add_argument('--gateway', choices=['fixture', 'mcp'], default='fixture')
    p.add_argument('--evaluator-model', help='OpenAI-compatible evaluator model')
    p.add_argument('--answer-model', help='OpenAI-compatible generation model')
    p.add_argument('--threshold', type=float, default=0.8)
    a = p.parse_args()
    issue = Issue(a.issue, a.version)
    docs = [Evidence(**x) for x in json.loads(Path(a.local).read_text())]
    evaluator = ModelEvaluator(a.evaluator_model) if a.evaluator_model else DemoEvaluator()
    generator = ModelGenerator(a.answer_model) if a.answer_model else DemoGenerator()
    gateway = MCPGateway('crag-gateway', []) if a.gateway == 'mcp' else FixtureGateway(a.gateway_fixture)
    result = Triage(evaluator, gateway, generator, a.threshold).run(issue, retrieve_local(issue, docs))
    print(json.dumps({'route': result.route, 'answer': result.answer, 'evidence': [e.__dict__ for e in result.evidence], 'grades': [g.__dict__ for g in result.grades], 'note': result.note}, indent=2))

if __name__ == '__main__':
    main()

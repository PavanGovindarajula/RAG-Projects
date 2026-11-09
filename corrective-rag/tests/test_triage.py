import unittest
from crag.core import Evidence, Grade, Issue, Triage

issue = Issue('API-503 connection pool exhausted', '2.0')
good = Evidence('good', '2.0', 'pool', 'evidence', 'local')
old = Evidence('old', '1.0', 'pool', 'old evidence', 'local')

class Evaluator:
    def __init__(self, score=.9, fail=False): self.score, self.fail = score, fail
    def grade(self, issue, item):
        if self.fail: raise ValueError('invalid model response')
        return Grade(item.id, self.score, True, True, 'test')

class Gateway:
    def __init__(self, items=(), fail=False): self.items, self.fail, self.calls = list(items), fail, 0
    def retrieve(self, issue):
        self.calls += 1
        if self.fail: raise RuntimeError('offline')
        return self.items

class Generator:
    def __init__(self): self.calls = 0
    def answer(self, issue, items):
        self.calls += 1
        return 'supported answer'

class TriageTests(unittest.TestCase):
    def test_high_local_skips_gateway(self):
        gateway, generator = Gateway(), Generator()
        result = Triage(Evaluator(), gateway, generator).run(issue, [good])
        self.assertEqual(result.route, 'local')
        self.assertEqual((gateway.calls, generator.calls), (0, 1))

    def test_wrong_version_triggers_gateway(self):
        gateway, generator = Gateway([good]), Generator()
        result = Triage(Evaluator(), gateway, generator).run(issue, [old])
        self.assertEqual(result.route, 'gateway')
        self.assertEqual([e.id for e in result.evidence], ['good'])

    def test_low_scores_escalate_without_generation(self):
        generator = Generator()
        result = Triage(Evaluator(.5), Gateway([good]), generator).run(issue, [good])
        self.assertEqual(result.route, 'human')
        self.assertEqual(generator.calls, 0)

    def test_empty_results_escalate(self):
        self.assertEqual(Triage(Evaluator(), Gateway(), Generator()).run(issue, []).route, 'human')

    def test_gateway_failure_escalates(self):
        self.assertEqual(Triage(Evaluator(), Gateway(fail=True), Generator()).run(issue, []).route, 'human')

    def test_evaluator_failure_escalates(self):
        gateway = Gateway([good])
        self.assertEqual(Triage(Evaluator(fail=True), gateway, Generator()).run(issue, [good]).route, 'human')
        self.assertEqual(gateway.calls, 0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError): Triage(Evaluator(), Gateway(), Generator(), 1.1)

if __name__ == '__main__': unittest.main()

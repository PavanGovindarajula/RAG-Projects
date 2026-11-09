from dataclasses import dataclass, field
import re
from typing import Protocol


@dataclass(frozen=True)
class Issue:
    text: str
    version: str


@dataclass(frozen=True)
class Evidence:
    id: str
    version: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Grade:
    evidence_id: str
    confidence: float
    version_relevant: bool
    technically_supported: bool
    reason: str

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError('confidence must be between 0 and 1')


@dataclass
class Result:
    route: str
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    grades: list[Grade] = field(default_factory=list)
    note: str = ''


class Evaluator(Protocol):
    def grade(self, issue: Issue, evidence: Evidence) -> Grade: ...


class Gateway(Protocol):
    def retrieve(self, issue: Issue) -> list[Evidence]: ...


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def retrieve_local(issue, documents, limit=5):
    query = terms(issue.text)
    scored = [(len(query & terms(d.title + ' ' + d.text)), d) for d in documents]
    return [d for score, d in sorted(scored, key=lambda x: (-x[0], x[1].id)) if score > 0][:limit]


class Triage:
    def __init__(self, evaluator, gateway, generator, threshold=0.8):
        if not 0 < threshold <= 1:
            raise ValueError('threshold must be in (0, 1]')
        self.evaluator, self.gateway, self.generator, self.threshold = evaluator, gateway, generator, threshold

    def _grade(self, issue, items):
        grades = [self.evaluator.grade(issue, item) for item in items]
        if any(g.evidence_id != item.id for item, g in zip(items, grades)):
            raise ValueError('Evaluator returned a mismatched evidence ID')
        accepted = [item for item, g in zip(items, grades) if item.version == issue.version and g.version_relevant and g.technically_supported and g.confidence >= self.threshold]
        return accepted, grades

    def run(self, issue, local):
        if not issue.text.strip() or not issue.version.strip():
            raise ValueError('issue text and version are required')
        try:
            accepted, grades = self._grade(issue, local)
        except Exception as exc:
            return Result('human', '', note=f'Local evaluation failed: {type(exc).__name__}')
        if accepted:
            return self._generate(issue, accepted, grades, 'local')
        try:
            remote = self.gateway.retrieve(issue)
            accepted, remote_grades = self._grade(issue, remote)
            grades.extend(remote_grades)
        except Exception as exc:
            return Result('human', '', grades=grades, note=f'Gateway or evaluation failed: {type(exc).__name__}')
        if not accepted:
            return Result('human', '', grades=grades, note='No sufficiently supported, version-matched evidence. Tier 3 review required.')
        return self._generate(issue, accepted, grades, 'gateway')

    def _generate(self, issue, evidence, grades, route):
        try:
            answer = self.generator.answer(issue, evidence)
            if not answer or not answer.strip():
                raise ValueError('Empty answer')
            return Result(route, answer, evidence, grades)
        except Exception as exc:
            return Result('human', '', evidence, grades, f'Generation failed: {type(exc).__name__}')

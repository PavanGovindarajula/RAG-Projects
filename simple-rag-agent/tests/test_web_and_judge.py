import io
import importlib.util
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from simple_rag import RAGAgent, TfidfRetriever
from simple_rag.documents import Document
from simple_rag.web import make_handler


class FakeModel:
    def answer(self, question, chunks):
        return "Leave is 20 days."


class WebTests(unittest.TestCase):
    def setUp(self):
        agent = RAGAgent(TfidfRetriever([Document("guide.md", "Annual leave is 20 days.")]), FakeModel())
        self.handler = make_handler(agent, "test-token")

    def request(self, method, path, body=None, headers=None):
        class Connection:
            def __init__(self, request):
                self.input = io.BytesIO(request)
                self.output = io.BytesIO()
            def makefile(self, mode, *args, **kwargs):
                return self.input
            def sendall(self, data):
                self.output.write(data)
        body = body or ""
        headers = dict(headers or {})
        headers["Content-Length"] = str(len(body.encode()))
        raw = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n" + body
        connection = Connection(raw.encode())
        self.handler(connection, ("127.0.0.1", 12345), None)
        response = connection.output.getvalue().decode()
        head, payload = response.split("\r\n\r\n", 1)
        return int(head.split()[1]), payload

    def test_query_and_retrieved_sources(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("test-token", page)
        status, body = self.request("POST", "/ask", json.dumps({"question": "annual leave"}), {"X-Local-Token": "test-token"})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["answer"], "Leave is 20 days.")
        self.assertEqual(result["sources"][0]["source"], "guide.md")

    def test_rejects_unauthorized_and_invalid_question(self):
        self.assertEqual(self.request("POST", "/ask", "{}", {})[0], 403)
        status, _ = self.request("POST", "/ask", '{"question":" "}', {"X-Local-Token": "test-token"})
        self.assertEqual(status, 400)


class JudgeTests(unittest.TestCase):
    def test_judge_parses_and_validates_result(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "judge.py"
        spec = importlib.util.spec_from_file_location("judge", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        judge = module.LLMJudge.__new__(module.LLMJudge)
        judge.model = "fake"
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"correctness": true, "groundedness": false, "reason": "unsupported"}'))])
        judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
        self.assertFalse(judge.score("q", "reference", "answer", [])["groundedness"])
        response.choices[0].message.content = '{"correctness": "true"}'
        with self.assertRaises(ValueError):
            judge.score("q", "reference", "answer", [])


if __name__ == "__main__":
    unittest.main()

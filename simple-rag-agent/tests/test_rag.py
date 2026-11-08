import tempfile
import unittest
from pathlib import Path
from simple_rag import DocumentError, RAGAgent, TfidfRetriever, load_documents
from simple_rag.documents import Document


class RecordingModel:
    def __init__(self):
        self.calls = []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        return f"Found in {chunks[0].source}"


class RAGTests(unittest.TestCase):
    def test_load_and_retrieve_configured_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("Annual leave is twenty days.")
            (root / "b.txt").write_text("The cafeteria serves lunch.")
            results = TfidfRetriever(load_documents(root, ["a.md", "b.txt"])).search("annual leave")
            self.assertEqual(results[0].source, "a.md")
            self.assertGreater(results[0].score, 0)

    def test_reject_path_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "secret.txt"
            outside.write_text("secret")
            (root / "link.txt").symlink_to(outside)
            for name in ("../secret.txt", "link.txt"):
                with self.assertRaises(DocumentError):
                    load_documents(root, [name])

    def test_reject_unsupported_empty_and_large(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.exe").write_text("data")
            (root / "empty.txt").write_text(" ")
            (root / "big.txt").write_text("too big")
            for name in ("a.exe", "empty.txt", "big.txt"):
                with self.assertRaises(DocumentError):
                    load_documents(root, [name], max_bytes=4)

    def test_agent_passes_evidence_and_abstains_without_overlap(self):
        retriever = TfidfRetriever([Document("leave.txt", "Annual leave is twenty days.")])
        model = RecordingModel()
        agent = RAGAgent(retriever, model)
        result = agent.ask("annual leave")
        self.assertEqual(result.sources[0].source, "leave.txt")
        self.assertEqual(model.calls[0][0], "annual leave")
        self.assertEqual(agent.ask("quasar").sources, [])
        self.assertEqual(len(model.calls), 1)

    def test_invalid_input_and_chunk_boundaries(self):
        with self.assertRaises(ValueError):
            TfidfRetriever([], chunk_words=2, overlap=2)
        retriever = TfidfRetriever([Document("a", "one two three four five")], chunk_words=3, overlap=1)
        self.assertEqual([c.text for c in retriever.chunks], ["one two three", "three four five"])
        with self.assertRaises(ValueError):
            RAGAgent(retriever, RecordingModel()).ask("  ")
        with self.assertRaises(ValueError):
            retriever.search("one", 0)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from hybrid_rag.retrieval import fuse, load_chunks, tokens

class RetrievalTests(unittest.TestCase):
    def test_fusion_rewards_agreement_and_deduplicates(self):
        self.assertEqual(fuse(['a', 'b', 'b'], ['b', 'c'])[0], 'b')
        self.assertEqual(len(fuse(['a', 'a'], [])), 1)

    def test_error_codes_are_searchable(self):
        self.assertEqual(tokens('VPN-104'), ['vpn', '104'])

    def test_sections_and_changed_content_have_new_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.md'
            p.write_text('## One\nFirst\n\n## Two\nSecond')
            before = load_chunks(tmp)
            self.assertEqual(len(before), 2)
            p.write_text('## One\nChanged')
            self.assertNotEqual(before[0]['id'], load_chunks(tmp)[0]['id'])

if __name__ == '__main__':
    unittest.main()

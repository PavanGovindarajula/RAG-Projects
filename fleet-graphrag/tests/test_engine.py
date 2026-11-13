import json
import tempfile
import unittest
from pathlib import Path
from fleet_graphrag.engine import build_index, search

DATA = Path(__file__).resolve().parents[1] / 'data/incidents.json'

class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'index.sqlite'
        build_index(DATA, self.db)
    def tearDown(self): self.tmp.cleanup()
    def test_global_themes_and_provenance(self):
        result = search(self.db, 'What systemic themes span the fleet?', ['ops'])
        self.assertEqual(result['mode'], 'global')
        themes = {x['theme']: x for x in result['details']}
        self.assertEqual(themes['cache exhaustion']['services'], ['checkout', 'inventory', 'payment'])
        self.assertEqual(themes['cache exhaustion']['sources'], ['INC-101', 'INC-102', 'INC-106'])
    def test_local_entity_neighbor(self):
        result = search(self.db, 'What happened to checkout?', ['ops'])
        self.assertEqual(set(result['sources']), {'INC-101', 'INC-103'})
    def test_access_is_applied_before_graph_and_summary(self):
        result = search(self.db, 'What happened to billing?', ['ops'])
        self.assertTrue(result['abstained'])
        self.assertNotIn('INC-105', str(result))
        permitted = search(self.db, 'What happened to billing?', ['restricted'])
        self.assertEqual(permitted['sources'], ['INC-105'])
    def test_no_tags_or_no_match_abstains(self):
        self.assertTrue(search(self.db, 'themes across fleet', [])['abstained'])
        self.assertTrue(search(self.db, 'quantum spaceship', ['ops'])['abstained'])
    def test_failed_rebuild_preserves_index(self):
        invalid = Path(self.tmp.name) / 'invalid.json'
        invalid.write_text(json.dumps([{'id':'bad'}]))
        with self.assertRaises(ValueError): build_index(invalid, self.db)
        self.assertFalse(search(self.db, 'checkout', ['ops'])['abstained'])
    def test_invalid_model_output_preserves_index(self):
        with self.assertRaises(ValueError): build_index(DATA, self.db, lambda _: [['bad']])
        self.assertFalse(search(self.db, 'checkout', ['ops'])['abstained'])

if __name__ == '__main__': unittest.main()

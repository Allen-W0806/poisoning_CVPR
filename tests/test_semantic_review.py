import json
import unittest
from src.evaluation.semantic_review import FIELDS,parse_inventory,compare_inventories

class SemanticReviewTests(unittest.TestCase):
    def test_changed_color_flag_is_not_human_ground_truth(self):
        source=parse_inventory(json.dumps({k:"red" for k in FIELDS}))
        edited={**source,"color":"blue"}
        result=compare_inventories(source,edited,"What color is the car?")
        self.assertEqual(result['checks']['color']['status'],'DESCRIPTION_DIFFERENCE_REVIEW')
        self.assertFalse(result['human_ground_truth']);self.assertFalse(result['independent_model_family'])

    def test_missing_and_unknown_are_not_passes(self):
        source=parse_inventory("malformed")
        result=compare_inventories(source,source,"How many birds?")
        self.assertEqual(result['checks']['count']['status'],'UNKNOWN')
        uncertain=parse_inventory(json.dumps({k:'unknown' for k in FIELDS}))
        self.assertEqual(compare_inventories(uncertain,uncertain,"How many birds?")['checks']['count']['status'],'UNKNOWN')

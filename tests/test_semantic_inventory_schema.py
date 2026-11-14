import json
import unittest
from src.evaluation.semantic_review import FIELDS
from src.evaluation.semantic_inventory_schema import parse_inventory_v2

class InventorySchemaTests(unittest.TestCase):
    def test_actual_integer_count_representation(self):
        value={k:'visible' for k in FIELDS};value['count']=1;raw=json.dumps(value)
        parsed=parse_inventory_v2(raw)
        self.assertTrue(parsed['parse_ok']);self.assertEqual(parsed['count'],'1');self.assertEqual(parsed['raw_output'],raw)

    def test_invalid_counts_and_missing_fields_stay_unknown(self):
        for count in [True,-1,1.5,None]:
            value={k:'visible' for k in FIELDS};value['count']=count
            self.assertFalse(parse_inventory_v2(json.dumps(value))['parse_ok'])
        self.assertFalse(parse_inventory_v2('{"count":1}')['parse_ok'])

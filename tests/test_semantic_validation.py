import unittest
import numpy as np
from PIL import Image
from src.evaluation.semantic_validation import semantic_decision, question_attributes, parse_judgment
from src.evaluation.edit_localization import edit_localization


class SemanticValidationTest(unittest.TestCase):
    def test_question_specific_and_unknown(self):
        row = {"sample_id": "car", "question": "What color is the car?"}
        red, blue = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
        self.assertTrue(question_attributes(row["question"])["color"])
        identity = semantic_decision(red, red, row, {"score": 1}, {})
        self.assertTrue(identity["valid"] and identity["verified"])
        changed = semantic_decision(red, blue, row, {"score": 0}, {"parse_ok": True, "color": False})
        self.assertFalse(changed["valid"])
        agreeable = {"parse_ok": True, **{k: True for k in ["object_identity", "color", "count", "spatial", "attributes", "answerable", "answer_preserved"]}}
        proxy = semantic_decision(red, blue, row, {"score": 1}, agreeable)
        self.assertFalse(proxy["valid"])
        self.assertFalse(proxy["verified"])
        self.assertTrue(proxy["provisionally_valid"])

    def test_invalid_control_and_localization(self):
        a, b = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "white")
        result = semantic_decision(a, b, {"sample_id": "blank", "question": "How many cars?"}, {"score": 0}, {}, "Original scene completely occluded")
        self.assertFalse(result["valid"])
        self.assertTrue(result["verified"])
        self.assertEqual(edit_localization(a, a)["changed_pixel_fraction"], 0)
        self.assertEqual(edit_localization(a, b)["changed_pixel_fraction"], 1)
        self.assertFalse(parse_judgment("not json")["parse_ok"])

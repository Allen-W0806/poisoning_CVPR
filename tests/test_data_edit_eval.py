import copy
import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from src.data.prepare_vqa import prepare_fixture, select_subsets
from src.data.dataset import VQADataset, validate_disjoint
from src.editing.transforms import BenignEditor
from src.editing.semantic_validator import validate_semantics
from src.evaluation.image_quality import image_quality
from src.utils.scratch import scratch_directory


class DataEditTest(unittest.TestCase):
    def test_fixture_editor_and_integrity(self):
        with scratch_directory() as tmp:
            root = Path(tmp)
            path = prepare_fixture(root)
            datasets = [VQADataset(path / f"{s}.json", root, s) for s in ["train", "validation", "test"]]
            self.assertTrue(validate_disjoint(*datasets))
            self.assertEqual(len(datasets[0]), 10)
            saved = (path / "train.json").read_bytes()
            self.assertEqual(prepare_fixture(root), path)
            self.assertEqual(saved, (path / "train.json").read_bytes())
            rows = [datasets[0][i] for i in range(2)]
            editor = BenignEditor()
            outputs = editor([r["image"] for r in rows], seed=42)
            replay = editor([r["image"] for r in rows], seed=42)
            for r, edited, repeat in zip(rows, outputs, replay):
                np.testing.assert_array_equal(edited, repeat)
                self.assertEqual(np.asarray(edited).dtype, np.uint8)
                self.assertEqual(edited.size, r["image"].size)
                self.assertEqual(validate_semantics(r["image"], edited, r).score, 1)
                self.assertTrue(all(np.isfinite(v) for k, v in image_quality(r["image"], edited).items() if k != "ssim_kind"))
                editor.save(edited, root / "edited.png")
                self.assertTrue((root / "edited.png").exists())
            unknown = validate_semantics(rows[0]["image"], outputs[0], {"source": "natural"})
            self.assertFalse(unknown.verified)
            invalid = validate_semantics(rows[0]["image"], Image.new("RGB", rows[0]["image"].size, "white"), rows[0])
            self.assertEqual(invalid.score, 0)
            with self.assertRaises(ValueError):
                editor([rows[0]["image"]], [{"brightness": 2}])
            with self.assertRaises(ValueError):
                validate_disjoint(datasets[0], datasets[0])

    def test_official_sampling_deterministic(self):
        annotations = [{"image_id": i // 2, "question_id": i, "answers": [{"answer": "yes"}] * 10,
                        "answer_type": ["yes/no", "number", "other"][i % 3]} for i in range(120)]
        questions = [{"question_id": i, "question": "original question"} for i in range(120)]
        original = copy.deepcopy(annotations)
        a = select_subsets(annotations, questions, sizes=(20, 10, 10))
        self.assertEqual(a, select_subsets(annotations, questions, sizes=(20, 10, 10)))
        self.assertEqual(annotations, original)
        ids = [{r["image_id"] for r in a[s]} for s in a]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


if __name__ == "__main__":
    unittest.main()

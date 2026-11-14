from pathlib import Path
import random
from PIL import Image, ImageEnhance

ACTION_FACTORS = (1.0, 0.97, 1.03)


class BenignEditor:
    """PIL backend; CPU execution is intentional, with CUDA policy support upstream."""
    backend = "pillow_brightness_v1"

    def __init__(self, max_delta=0.03):
        self.max_delta = max_delta
        self.last_parameters = []

    def __call__(self, images, conditions=None, seed=42):
        if conditions is not None and len(images) != len(conditions):
            raise ValueError("Image/condition batch size mismatch")
        rng = random.Random(seed)
        self.last_parameters = []
        outputs = []
        for i, source in enumerate(images):
            image = self.load(source)
            condition = conditions[i] if conditions is not None else {}
            if not isinstance(condition, dict) or set(condition) - {"brightness", "action"}:
                raise ValueError("Editor supports only brightness/action conditions")
            if "action" in condition:
                factor = ACTION_FACTORS[int(condition["action"])]
            else:
                factor = float(condition.get("brightness", rng.uniform(1 - self.max_delta, 1 + self.max_delta)))
            if abs(factor - 1) > self.max_delta + 1e-8:
                raise ValueError("Brightness exceeds benign engineering bound")
            outputs.append(ImageEnhance.Brightness(image).enhance(factor))
            self.last_parameters.append({"brightness": factor, "seed": seed, "index": i, "backend": self.backend})
        return outputs

    @staticmethod
    def load(source):
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        with Image.open(source) as image:
            return image.convert("RGB")

    @staticmethod
    def save(image, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

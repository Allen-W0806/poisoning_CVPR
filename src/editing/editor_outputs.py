from dataclasses import dataclass
from PIL import Image


@dataclass
class EditedOutput:
    image: Image.Image
    metadata: dict

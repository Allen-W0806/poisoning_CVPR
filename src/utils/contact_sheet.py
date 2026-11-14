from pathlib import Path
from PIL import Image, ImageOps, ImageDraw


def contact_sheet(pairs, output, thumbnail=256):
    width, row_height = 2 * thumbnail, thumbnail + 54
    canvas = Image.new("RGB", (width, row_height * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (original, edited, label) in enumerate(pairs):
        y = i * row_height
        for j, image in enumerate([original, edited]):
            canvas.paste(ImageOps.pad(image.convert("RGB"), (thumbnail, thumbnail), color="white"), (j * thumbnail, y))
        text = str(label)
        draw.text((5, y + thumbnail + 3), text[:85], fill="black")
        draw.text((5, y + thumbnail + 20), text[85:170], fill="black")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

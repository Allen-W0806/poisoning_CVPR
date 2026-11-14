"""Native aspect-ratio preprocessing for inference; old padded contracts preserved."""
from PIL import Image,ImageOps


def native_aspect_image(image,max_side=512):
    image=ImageOps.exif_transpose(image).convert('RGB')
    if max_side<64 or max_side%8:raise ValueError('Max side must be >=64 and divisible by eight')
    scale=max_side/max(image.size)
    width=max(8,int(round(image.width*scale/8))*8)
    height=max(8,int(round(image.height*scale/8))*8)
    # Small rounding to the VAE grid; full scene is resized, never cropped.
    return image.resize((width,height),Image.Resampling.LANCZOS)

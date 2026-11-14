import unittest
from PIL import Image
from src.editing.native_preprocessing import native_aspect_image


class NativeAspectTests(unittest.TestCase):
    def test_aspect_grid_and_full_scene_no_white_border(self):
        for dimensions in [(640,427),(427,640),(343,25088),(256,256)]:
            source=Image.new('RGB',dimensions,(100,30,20));result=native_aspect_image(source)
            self.assertEqual(max(result.size),512)
            self.assertTrue(all(value%8==0 for value in result.size))
            self.assertEqual(result.getpixel((0,0)),(100,30,20))
            self.assertEqual(result.getpixel((result.width-1,result.height-1)),(100,30,20))

    def test_invalid_grid_and_resolution_rejected(self):
        for value in (0,63,511):
            with self.assertRaises(ValueError):native_aspect_image(Image.new('RGB',(100,100)),value)

    def test_noop_dimensions_keep_rgb(self):
        result=native_aspect_image(Image.new('L',(512,256),100))
        self.assertEqual(result.size,(512,256));self.assertEqual(result.mode,'RGB')

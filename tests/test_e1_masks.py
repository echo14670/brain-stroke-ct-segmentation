"""Regression checks for E1 mask extraction and baseline compatibility."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from src.data.dataset import extract_overlay_mask


class E1MaskTest(unittest.TestCase):
    def test_union_preserves_red_and_adds_blue_only(self):
        rgb = np.array([[[220,80,80],[80,80,220],[128,128,128],[0,255,0]]],dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'overlay.png'
            Image.fromarray(rgb).save(path)
            red=extract_overlay_mask(path,'red')
            union=extract_overlay_mask(path,'red_blue')
        np.testing.assert_array_equal(red,[[1,0,0,0]])
        np.testing.assert_array_equal(union,[[1,1,0,0]])
        self.assertEqual(union.dtype,np.uint8)

    def test_thresholds_and_signed_channel_differences(self):
        rgb=np.array([[[80,60,60],[60,60,80],[79,0,0],[0,0,79],[0,0,255]]],dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'overlay.png'
            Image.fromarray(rgb).save(path)
            red=extract_overlay_mask(path,'red')
            union=extract_overlay_mask(path,'red_blue')
            with self.assertRaises(ValueError):
                extract_overlay_mask(path,'invalid')
        np.testing.assert_array_equal(red,[[1,0,0,0,0]])
        np.testing.assert_array_equal(union,[[1,1,0,0,1]])


if __name__=='__main__':
    unittest.main()

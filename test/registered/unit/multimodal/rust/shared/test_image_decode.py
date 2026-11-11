"""Model-independent image decode parity for Rust MM.

Covers ``decode_rgb`` in ``rust/sglang-mm/src/common/mod.rs`` (via the
``_core.common.image_decode_rgb`` binding), against PIL's
``Image.open(...).convert("RGB")``.
"""

import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CORE = load_core()
DECODE = CORE and CORE.common.image_decode_rgb


def encode(image, fmt, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@unittest.skipUnless(DECODE, "sglang-mm decode binding not built")
class TestRustImageDecode(CustomTestCase):
    def assert_matches_pil(self, data, tolerance=0):
        height, width, pixels = DECODE(data)
        expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        self.assertEqual((height, width), expected.shape[:2])
        actual = np.asarray(pixels).reshape(expected.shape)
        diff = np.abs(actual.astype(int) - expected.astype(int))
        self.assertLessEqual(diff.max(), tolerance)

    def test_png_modes_match_pil(self):
        rgb = np.random.default_rng(1).integers(0, 256, (19, 23, 3), dtype=np.uint8)
        cases = [
            ("RGB", Image.fromarray(rgb)),
            ("L", Image.fromarray(rgb[..., 0])),
            ("RGBA", Image.fromarray(np.dstack((rgb, rgb[..., 0])))),
            ("P", Image.fromarray(rgb).quantize(colors=16)),
        ]
        for mode, image in cases:
            with self.subTest(mode=mode):
                self.assert_matches_pil(encode(image, "PNG"))

    def test_jpeg_modes_match_with_decoder_tolerance(self):
        rgb = np.random.default_rng(2).integers(0, 256, (31, 29, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        exif = image.getexif()
        exif[274] = 6  # EXIF orientation: neither decoder applies it
        cases = [
            ("RGB", encode(image, "JPEG")),
            ("L", encode(Image.fromarray(rgb[..., 0]), "JPEG")),
            ("CMYK", encode(image.convert("CMYK"), "JPEG")),
            ("EXIF-rotated", encode(image, "JPEG", exif=exif)),
        ]
        for mode, data in cases:
            with self.subTest(mode=mode):
                self.assert_matches_pil(data, tolerance=3)

    def test_lossless_formats_match_pil_exactly(self):
        """GIF/BMP/lossless-WebP joined the native decoder with the pure-Rust
        webp/gif/bmp enablement. Their reconstruction is exact by format spec,
        so parity with PIL is pinned at zero tolerance like PNG."""
        rgb = np.random.default_rng(4).integers(0, 256, (17, 21, 3), dtype=np.uint8)
        image = Image.fromarray(rgb)
        cases = {
            "gif": encode(image.quantize(colors=64), "GIF"),
            "bmp": encode(image, "BMP"),
            "webp-lossless": encode(image, "WEBP", lossless=True),
        }
        for name, data in cases.items():
            with self.subTest(fmt=name):
                self.assert_matches_pil(data)

    def test_pillow_decompression_bomb_maps_to_value_error(self):
        """The Python decode entry point must convert Pillow's bomb check
        exception into a ValueError so callers classify it as client error."""
        from sglang.srt.multimodal.mm_utils import load_image_from_base64

        data = encode(Image.new("RGB", (20, 20), (0, 255, 0)), "PNG")
        encoded = base64.b64encode(data).decode()
        with patch.object(Image, "MAX_IMAGE_PIXELS", 50):
            with self.assertRaisesRegex(ValueError, "Failed to load image") as ctx:
                load_image_from_base64(encoded)
            self.assertIsInstance(ctx.exception.__cause__, Image.DecompressionBombError)

    def test_normal_image_decodes_through_python_entry(self):
        from sglang.srt.multimodal.mm_utils import load_image_from_base64

        data = encode(Image.new("RGB", (20, 20), (0, 255, 0)), "PNG")
        image = load_image_from_base64(base64.b64encode(data).decode())
        self.assertEqual(image.size, (20, 20))

    def test_pillow_bomb_check_on_decode_maps_to_value_error(self):
        """Image.open is lazy, so the bomb check may fire on the first decode
        access; the decode entry point must still convert it to ValueError."""
        from sglang.srt.multimodal.mm_utils import load_image_from_base64

        data = encode(Image.new("RGB", (20, 20), (0, 255, 0)), "PNG")
        encoded = base64.b64encode(data).decode()
        with patch.object(Image, "MAX_IMAGE_PIXELS", 50):
            image = load_image_from_base64(encoded)
            with self.assertRaises(Image.DecompressionBombError):
                image.load()

    def test_unsupported_inputs_fail(self):
        gray16 = Image.fromarray(np.zeros((9, 9), dtype=np.uint16))
        cases = {
            "corrupt": b"not an image",
            # >8-bit depths must error (the request is then rejected — there
            # is no Python fallback) — never silently diverge from PIL's
            # clipping.
            "png16": encode(gray16, "PNG"),
        }
        for name, data in cases.items():
            with self.subTest(input=name):
                with self.assertRaises(ValueError):
                    DECODE(data)


if __name__ == "__main__":
    unittest.main()

import unittest

from quantum_framework.utils import BaseViTConfig


class ConfigTests(unittest.TestCase):
    def test_computed_tensor_dimensions(self):
        config = BaseViTConfig(image_size=28, patch_size=7, n_channels=3, embed_dim=8)
        self.assertEqual(config.n_patches, 16)
        self.assertEqual(config.seq_len, 17)
        self.assertEqual(config.patch_dim, 147)
        self.assertEqual(config.n_qubits, 8)

    def test_incompatible_attention_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            BaseViTConfig(embed_dim=10, n_head=3)


if __name__ == "__main__":
    unittest.main()

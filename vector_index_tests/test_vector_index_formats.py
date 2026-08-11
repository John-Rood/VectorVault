import os
import tempfile
import threading
import unittest

import numpy as np
from annoy import AnnoyIndex

from vectorvault.itemize import FAISSIndex


class VectorIndexFormatTests(unittest.TestCase):
    DIMS = 3
    VECTORS = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def _legacy_annoy_file(self, directory, name="legacy.ann"):
        path = os.path.join(directory, name)
        index = AnnoyIndex(self.DIMS, "angular")
        for item_id, vector in enumerate(self.VECTORS):
            index.add_item(item_id, vector)
        index.build(2)
        self.assertTrue(index.save(path))
        return path

    def test_loads_historical_annoy_index_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._legacy_annoy_file(directory)
            index = FAISSIndex(self.DIMS)

            index.load(path)

            self.assertEqual(index.get_n_items(), 2)
            np.testing.assert_allclose(index.get_item_vector(0), self.VECTORS[0])
            self.assertEqual(index.get_nns_by_vector(self.VECTORS[0], 1), [0])

    def test_loads_current_npz_from_extensionless_download(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = FAISSIndex(self.DIMS)
            for item_id, vector in enumerate(self.VECTORS):
                saved.add_item(item_id, vector)
            extensionless = os.path.join(directory, "downloaded-object")
            saved.save(extensionless)

            loaded = FAISSIndex(self.DIMS)
            loaded.load(extensionless)

            self.assertEqual(loaded.get_n_items(), 2)
            np.testing.assert_allclose(loaded.get_item_vector(1), self.VECTORS[1])

    def test_rejects_pickle_payload_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unsafe.npy")
            np.save(path, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)

            with self.assertRaisesRegex(ValueError, "pickle|format"):
                FAISSIndex(self.DIMS).load(path)

    def test_rejects_corrupt_unknown_format_without_fake_npz_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corrupt.ann")
            with open(path, "wb") as handle:
                handle.write(b"not-a-vector-index")

            with self.assertRaisesRegex(ValueError, "Unsupported or corrupt vector index format"):
                FAISSIndex(self.DIMS).load(path)
            self.assertFalse(os.path.exists(path + ".npz"))

    def test_repeated_concurrent_legacy_loads_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._legacy_annoy_file(directory)
            results = []
            errors = []
            result_lock = threading.Lock()

            def load_once():
                try:
                    index = FAISSIndex(self.DIMS)
                    index.load(path)
                    value = (index.get_n_items(), index.get_nns_by_vector(self.VECTORS[1], 1))
                    with result_lock:
                        results.append(value)
                except Exception as exc:  # pragma: no cover - captured for the assertion
                    with result_lock:
                        errors.append(exc)

            threads = [threading.Thread(target=load_once) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(results, [(2, [1])] * 10)


if __name__ == "__main__":
    unittest.main()

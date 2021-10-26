import unittest
import numpy as np

from cytoemd import distances
from cytoemd import CytoEMD


class TestDistances(unittest.TestCase):
    def test_euclidean(self):
        self.assertEqual(distances.euclidean(np.zeros((3,))), 0)

    def test_manhattan(self):
        self.assertEqual(distances.manhattan(np.zeros((3,))), 0)

    def test_chebyshev(self):
        self.assertEqual(distances.chebyshev(np.zeros((3,))), 0)

    def test_emd_samples(self):
        self.assertEqual(distances.emd_samples(np.zeros((3,)), np.zeros((3,)))[0], 0)


class TestCytoEMD(unittest.TestCase):
    def test_CytoEMD(self):
        n_samples = 20
        n_cells = 200
        n_dimensions = 20
        data_list = [np.random.normal(size=(n_cells, n_dimensions)) for _ in range(n_samples)]

        model = CytoEMD(emd_type='UMAP', use_fast=True)
        embed = model.fit_transform(data_list)
        assert isinstance(embed, np.ndarray) and embed.shape == (n_samples, 2)


if __name__ == '__main__':
    unittest.main()
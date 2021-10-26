import numpy as np

from cytoemd import (
    CytoEMD,
    distances
)


def test_distance():
    x = np.random.normal(size=(3, ))
    y = x

    dis_euc = distances.euclidean(y - x)
    dis_man = distances.manhattan(y - x)
    dis_cheb = distances.chebyshev(y - x)
    dis_mink = distances.minkowski(y - x)
    dis_emd = distances.emd_samples(x, y)

    assert dis_euc == 0
    assert dis_man == 0
    assert dis_cheb == 0
    assert dis_mink == 0
    assert dis_emd == (0, None, None)


def test_CytoEMD():
    n_samples = 10
    n_cells = 200
    n_dimensions = 20
    data_list = [np.random.normal(size=(n_cells, n_dimensions)) for _ in range(n_samples)]

    model = CytoEMD(emd_type='UMAP', use_fast=True)
    embed = model.fit_transform(data_list)
    assert isinstance(embed, np.ndarray) and embed.shape == (n_samples, 2)

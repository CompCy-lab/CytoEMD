from typing import Union
from pkg_resources import parse_version

import scipy
import numpy as np
import pyemd


def euclidean(dx):
    """Standard euclidean distance.

    ..math::
        D(dx) = \sqrt{\sum_i (delta_x_i)^2}
    """
    return np.sqrt(np.sum(dx ** 2, axis=0))


def manhattan(dx):
    """Manhattan, taxicab, or l1 disance

    ..math::
        D(dx) = \sum_i |dx_i|
    """
    return np.sum(dx, axis=0)


def chebyshev(dx):
    """Chebyshev or l-infinity distance.

    ..math::
        D(dx) = \max_i |dx_i|
    """
    return np.max(dx, axis=0)


def minkowski(dx, p=2):
    """Minkowski distance

    ..math::
        D(dx) = \left(\sum_i |dx_i|^p\right)^{\frac{1}{p}}
    """
    return np.sum(dx ** p, axis=0) ** (1.0 / p)


if parse_version(np.__version__) >= parse_version('1.15.0'):
    get_bins = np.histogram_bin_edges
else:
    def get_bins(a, bins=10, **kwargs):
        if isinstance(bins, str):
            hist, bins = np.histogram(a, bins=bins, **kwargs)
        return bins


def euclidean_pairwise_distance_matrix(x):
    """Calculate the Euclidean pairwise distance matrix for a 1D array."""
    distance_matrix = np.abs(np.repeat(x, len(x)) - np.tile(x, len(x)))
    return distance_matrix.reshape(len(x), len(x))


def emd_samples(
    first_array,
    second_array,
    use_fast: bool = True,
    distance: str = 'euclidean',
    normalized: bool = True,
    bins: Union[str, int] = 'auto',
    range=None
):
    first_array = np.array(first_array)
    second_array = np.array(second_array)
    # Validate arrays
    if not (first_array.size > 0 and second_array.size > 0):
        raise ValueError('Arrays of samples cannot be empty.')
    # Get the default range
    if range is None:
        range = (min(np.min(first_array), np.min(second_array)),
                 max(np.max(first_array), np.max(second_array)))
    # Get bin edges using both arrays
    bins = get_bins(np.concatenate([first_array, second_array]),
                    range=range,
                    bins=bins)
    # Compute histograms
    first_histogram, bin_edges = np.histogram(first_array,
                                              range=range,
                                              bins=bins)
    second_histogram, _ = np.histogram(second_array,
                                       range=range,
                                       bins=bins)
    if normalized:
        first_histogram = first_histogram / np.sum(first_histogram)
        second_histogram = second_histogram / np.sum(second_histogram)
    # Compute the distance matrix between the center of each bin
    bin_locations = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)

    if not use_fast:
        # Cast to C++ long
        first_histogram = first_histogram.astype(np.float64)
        second_histogram = second_histogram.astype(np.float64)
        if distance == 'euclidean':
            distance = euclidean_pairwise_distance_matrix
        distance_matrix = distance(bin_locations).astype(np.float64)
        # Validate distance matrix
        if len(distance_matrix) != len(distance_matrix[0]):
            raise ValueError(
                'Distance matrix must be square; check your `distance` function.')
        if (first_histogram.shape[0] > len(distance_matrix) or second_histogram.shape[0] > len(distance_matrix)):
            raise ValueError(
                'Distance matrix must have at least as many rows/columns as there '
                'are bins in the histograms; check your `distance` function.')
        emd_value, min_flow = pyemd.emd_with_flow(first_histogram,
                                                  second_histogram,
                                                  distance_matrix)
        return emd_value, bin_edges, min_flow
    else:
        return (scipy.stats.wasserstein_distance(
            bin_locations, bin_locations, first_histogram, second_histogram), None, None)

from typing import Dict, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from tqdm import tqdm
from itertools import combinations
from functools import partial

import umap
import meld
import multiprocessing
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.manifold import MDS
from sklearn import preprocessing

from . import utils

cores = multiprocessing.cpu_count()


class CytoEMD(object):
    """
    The main class of EMD(Earth Moving Distance)-based embedding methods of CytoF Data
    """
    def __init__(
        self,
        emd_type: Literal['UMAP', 'MDS'] = 'UMAP',
        metric: str = "manhattan",
        bins: Union[int, str] = 'auto',
        random_state: int = 0,
        use_fast: bool = True,
        n_cpus: int = 1,
        verbose: bool = True
    ):
        self.bins = bins
        self.metric = metric
        self.emd_type = emd_type
        self.random_state = random_state
        self.n_cpus = max(1, min(n_cpus, cores))
        self.use_fast = use_fast
        self.verbose = verbose

    def _gen_pair_list(self, N, data_list):
        idx_list = list(range(N))
        return zip(combinations(idx_list, r=2), combinations(data_list, r=2))

    def fit(
        self,
        data_list: List[Union[np.ndarray, ad.AnnData, pd.DataFrame]],
        fp32: bool = True
    ) -> None:
        dtype = np.float32 if (fp32 and self.use_fast) else np.float64
        # input check
        assert len(data_list) > 0, "only list can be used as input, please check the input data type"
        num_sample, num_marker = len(data_list), data_list[0].shape[1]

        if hasattr(self, 'num_sample') and hasattr(self, 'num_marker'):
            if (self.num_sample != num_sample) or (self.num_marker != num_marker):
                raise RuntimeError(
                    "input data_list is different from the one used in the fitted model, "
                    "please double check the input data before run this program!"
                )
        else:
            self.num_sample = num_sample
            self.num_marker = num_marker

        # data type check
        if isinstance(data_list[0], ad.AnnData):
            assert all(
                [(v.var.index == data_list[0].var.index).all() for v in data_list]
            ), "Error: the markers are different across different samples."
            self.markers = data_list[0].var.index.tolist()
            data_list = [adata.X.astype(dtype=dtype) for adata in data_list]
        elif isinstance(data_list, pd.DataFrame):
            assert all(
                [(v.columns == data_list[0].columns).all() for v in data_list]
            ), "Error: the markers are different across different samples."
            self.markers = data_list[0].columns.tolist()
            data_list = [df.to_numpy(dtype=dtype) for df in data_list]
        else:
            if not isinstance(data_list[0], np.ndarray):
                raise NotImplementedError("We only supports nd.ndarray, anndata or pandas dataframe as input data.")

        # compute the distance matrix
        distance_tensor = np.zeros((num_marker, num_sample, num_sample), dtype=dtype)
        bin_edges, min_flows = dict(), dict()
        num_pairs = num_sample * (num_sample - 1) // 2

        pool = multiprocessing.Pool(self.n_cpus)
        pair_distances = pool.imap(
            partial(utils.compute_pair_emd, use_fast=self.use_fast, bins=self.bins),
            self._gen_pair_list(num_sample, data_list)
        )

        count = 0
        with tqdm(range(num_pairs)) as t:
            for (i, j, dis_vec), bg, mf in pair_distances:
                distance_tensor[:, i, j] = dis_vec
                distance_tensor[:, j, i] = dis_vec
                if not self.use_fast:
                    bin_edges[(i, j)] = bg
                    min_flows[(i, j)] = mf
                count += 1
                t.update()
        pool.close()
        pool.join()
        # check the processed number
        assert count == num_pairs, "the processed number != data number, some data is missing."
        # generate distance matrix and make symmetric
        self.distance_tensor = distance_tensor
        if not self.use_fast:
            self.bin_edge = bin_edges
            self.min_flow = min_flows

        # compute the distance matrix
        dis_fn = utils.get_distance_func(self.metric)
        self.distance_matrix = dis_fn(distance_tensor)

    def distance_to_embed(self, dis_mtx, emd_type: str = 'UMAP', **kwargs):
        if emd_type == 'UMAP':
            embeddings = umap.UMAP(
                metric="precomputed",
                random_state=self.random_state,
                **kwargs
            ).fit_transform(dis_mtx)
        elif emd_type == 'MDS':
            embeddings = MDS(
                dissimilarity='precomputed',
                random_state=self.random_state,
                **kwargs
            ).fit_transform(dis_mtx)
        else:
            raise ValueError(f"{self.emb_type} is unknown, please choose MDS or UMAP")
        return embeddings

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        # get the embeddings with MDS or UMAP
        embeddings = self.distance_to_embed(self.distance_matrix,
                                            self.emd_type,
                                            **kwargs)
        return embeddings

    def predict_prob(self, embedding, sample_labels, meld_kwargs: Dict = {}):
        # pred_densities = G_MELD(**meld_kwargs).fit_transform(self.distance_matrix, sample_labels)
        pred_densities = meld.MELD(**meld_kwargs).fit_transform(embedding, sample_labels)
        pred_prob = meld.utils.normalize_densities(pred_densities)
        return pred_prob

    def rank_markers(
            self,
            X,
            sample_labels: List,
            eval_pair: bool = False,
            normalize_embed: bool = True,
            meld_kwargs: Dict = {},
            **kwargs
    ):
        if not hasattr(self, 'distance_matrix'):
            self.fit(X, **kwargs)

        meld_model = meld.MELD(**meld_kwargs)
        entropy_score = []

        def _normalize_embed(x):
            scaler = preprocessing.StandardScaler().fit(x)
            return scaler.transform(x)

        if eval_pair:
            marker_pair_list = []
            marker_list = self.markers if hasattr(self, 'markers') else range(self.num_marker)
            for (var_i, var_j), (dis_i, dis_j) in zip(
                combinations(marker_list, r=2), combinations(self.distance_tensor, r=2)
            ):
                dis_tensor = np.stack([dis_i, dis_j], axis=0)
                dis_mtx = utils.get_distance_func(self.metric)(dis_tensor)
                embedding = self.distance_to_embed(dis_mtx, self.emd_type)
                if normalize_embed:
                    embedding = _normalize_embed(embedding)
                marker_pair_list.append('-'.join([str(var_i), str(var_j)]))
                entropy_score.append(utils.embed_to_cross_entropy(meld_model, embedding, sample_labels))
            entropy_score = dict(zip(marker_pair_list, entropy_score))
        else:
            for dis_mtx in self.distance_tensor:
                embedding = self.distance_to_embed(dis_mtx, self.emd_type)
                if normalize_embed:
                    embedding = _normalize_embed(embedding)
                entropy_score.append(utils.embed_to_cross_entropy(meld_model, embedding, sample_labels))
            if hasattr(self, 'markers'):
                entropy_score = dict(zip(self.markers, entropy_score))

        return entropy_score

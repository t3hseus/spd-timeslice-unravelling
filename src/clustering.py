import gin
import importlib

import numpy as np
from sklearn.cluster import KMeans

from collections import defaultdict


@gin.configurable(allowlist=None)
def clustering_algorithm(module: str, class_: str, **kwargs: dict):
    module = importlib.import_module(module)
    cls = getattr(module, class_)
    return cls(**kwargs)


class Clustering:
    def __init__(self, clustering_algorithm):
        self.clustering_algorithm = clustering_algorithm

    def compute_clusters(self, embeddings: np.ndarray):
        cluster_assignments = self.clustering_algorithm.fit_predict(embeddings)
        return cluster_assignments

    def link_clusters(self, cluster_assignments: np.ndarray, labels: np.ndarray):

        labels_dict = defaultdict(set)
        cluster_dict = defaultdict(set)

        # add indices to sets for each label
        for i, val in enumerate(labels.tolist()):
            labels_dict[val].add(i)

        for i, val in enumerate(cluster_assignments.tolist()):
            cluster_dict[val].add(i)

        # calculate intersections and link clusters and events
        cluster_evt_dict = {}
        for cluster, cluster_set in cluster_dict.items():
            max_intersect = 0
            max_evt = None
            for evt, evt_set in labels_dict.items():
                intersect = len(cluster_set & evt_set)
                if intersect > max_intersect:
                    max_intersect = intersect
                    max_evt = evt
            cluster_evt_dict[cluster] = max_evt

        vfunc = np.vectorize(cluster_evt_dict.get)
        return vfunc(cluster_assignments)

    def cluster_and_link(self, embeddings: np.ndarray, labels: np.ndarray):
        cluster_assignments = self.compute_clusters(embeddings)
        return self.link_clusters(cluster_assignments, labels)
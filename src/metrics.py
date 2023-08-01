import gin
import importlib

from torchmetrics import Metric
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from functools import partial
from collections import defaultdict
import numpy as np
import torch


@gin.configurable(allowlist=None)
def clustering_algorithm(module: str, class_: str, **kwargs: dict):
    module = importlib.import_module(module)
    cls = getattr(module, class_)
    return cls(**kwargs)

class Clustering(Metric):
    def __init__(self, clustering_algorithm, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.clustering_algorithm = clustering_algorithm
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings.append(embeddings)
        self.labels.append(labels)

    def compute(self):
        embeddings = torch.cat(self.embeddings, dim=0).cpu().numpy()
        labels = torch.cat(self.labels, dim=0).cpu().numpy()

        cluster_assignments = self.clustering_algorithm.fit(embeddings).labels_

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

    def reset(self):
        self.embeddings = []
        self.labels = []


class ClusterMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("clusters", default=[], dist_reduce_fx="cat")

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, clusters: np.ndarray):
        self.embeddings.append(embeddings)
        self.labels.append(labels)
        self.clusters.append(clusters)

    def compute(self):
        embeddings = torch.cat(self.embeddings, dim=0).cpu().numpy()
        labels = torch.cat(self.labels, dim=0).cpu().numpy()
        clusters = np.concatenate(self.clusters)

        metrics = {}
        averages = ['macro', 'micro', 'weighted']

        metrics_functions = {
            'internal': {
                'silhouette_score': silhouette_score,
                'davies_bouldin_score': davies_bouldin_score,
                'calinski_harabasz_score': calinski_harabasz_score,
            },
            'external': {
                'accuracy_score': accuracy_score,
            }
        }

        score_ext_metrics = {
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
        }

        for average in averages:
            for metric, metric_func in score_ext_metrics.items():
                metric_name = f'{average}_{metric}'
                metrics_functions['external'][metric_name] = partial(metric_func, average=average, zero_division=0)

        for metric_name, metric_function in metrics_functions['internal'].items():
            metrics['int_'+metric_name] = metric_function(embeddings, clusters)

        for metric_name, metric_function in metrics_functions['external'].items():
            metrics['ext_'+metric_name] = metric_function(labels, clusters)

        return metrics

    def reset(self):
        self.embeddings = []
        self.labels = []
        self.clusters = []
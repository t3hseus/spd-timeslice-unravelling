import gin
import torch
import pytorch_lightning as pl

from torch import nn
from enum import Enum
from typing import Optional
from umap import UMAP
from pytorch_metric_learning import (
    distances,
    reducers,
    losses,
    miners,
)

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from collections import defaultdict
from functools import partial
import numpy as np

from typing import Dict

from .visualization import draw_embeddings


@gin.constants_from_enum
class TripletType(str, Enum):
    all = "all"
    semihard = "semihard"
    hard = "hard"
    easy = "easy"


@gin.constants_from_enum
class DistanceType(str, Enum):
    cosine_similarity = "CosineSimilarity"
    euclidean_distance = "LpDistance"


class TripletTracksEmbedder(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        triplet_margin: float = 0.2,
        type_of_triplets: TripletType = "semihard",
        distance: DistanceType = DistanceType.cosine_similarity,
        learning_rate: float = 1e-4,
        umapper: Optional[UMAP] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'umapper'])

        if umapper is None:
            # configure default one
            umapper = UMAP()
        self.umapper = umapper

        self.model = model
        self._distance = getattr(distances, distance)()
        self.criterion = losses.TripletMarginLoss(
            margin=triplet_margin,
            distance=self._distance,
            reducer=reducers.ThresholdReducer(low=0)
        )
        self.learning_rate = learning_rate
        self.triplet_miner = miners.TripletMarginMiner(
            margin=triplet_margin,
            type_of_triplets=type_of_triplets,
            distance=self._distance
        )
        self.validation_step_outputs = []

    def forward(self, inputs):
        return self.model(inputs)

    def _forward_batch(self, batch, return_embeddings=False):
        tracks, event_ids = batch
        embeddings = self.model(tracks)
        indices_tuple = self.triplet_miner(embeddings, event_ids)
        loss = self.criterion(embeddings, event_ids, indices_tuple)
        if return_embeddings:
            return loss, embeddings
        # else
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_batch(batch)
        self.log_dict(
            {
                "train_loss": loss,
                "train_triplets": float(self.triplet_miner.num_triplets)
            },
            prog_bar=True
        )
        return loss

    def compute_metrics(self, embeddings: torch.Tensor, labels: torch.Tensor, clusters: np.ndarray) -> Dict[str, float]:
        """
        Compute various internal and external clustering metrics.

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings on which clustering is performed.
        labels : torch.Tensor
            The actual labels for the embeddings.
        clusters : np.ndarray
            The clusters assignment from clustering algorithm.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the computed metrics.
        """

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
                **{
                    f'{average}_{metric}': partial(metric_func, average=average, zero_division=0)
                    for metric, metric_func in {
                        'precision_score': precision_score,
                        'recall_score': recall_score,
                        'f1_score': f1_score,
                    }.items()
                    for average in averages
                }
            }
        }

        embeddings = embeddings.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        for metric_name, metric_function in metrics_functions['internal'].items():
            metrics['int_'+metric_name] = metric_function(embeddings, labels)

        for metric_name, metric_function in metrics_functions['external'].items():
            metrics['ext_'+metric_name] = metric_function(labels, clusters)

        return metrics

    def clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Perform clustering on the given embeddings using KMeans and map the labels according to maximum set intersection.

        Parameters
        ----------
        embeddings : np.ndarray
            The embeddings on which clustering is to be performed.
        labels : np.ndarray
            The actual labels for the given embeddings.

        Returns
        -------
        np.ndarray
            The labels predicted by KMeans mapped according to maximum intersection with actual labels.
        """

        kmeans = KMeans(n_clusters=40, random_state=0, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.labels_

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

    def validation_step(self, batch, batch_idx):
        loss, embeddings = self._forward_batch(batch, return_embeddings=True)

        emb_np = embeddings.cpu().detach().numpy()
        evt_ids_np = batch[1].cpu().numpy()

        cluster_preds = self.clustering(emb_np, evt_ids_np)
        metrics = self.compute_metrics(embeddings, batch[1], cluster_preds)

        self.validation_step_outputs.append({
            "embeddings": emb_np,
            "event_ids": evt_ids_np,
            "metrics": metrics
        })
        self.log_dict(
            {
                "val_loss": loss,
                "val_triplets": float(self.triplet_miner.num_triplets),
            },
            prog_bar=True
        )

    def on_validation_epoch_end(self):
        # take first batch for validation
        sample_idx = 0
        sample_for_visualization = self.validation_step_outputs[sample_idx]

        umap_embeddings = self.umapper.fit_transform(
            sample_for_visualization["embeddings"]
        )

        plt_obj = draw_embeddings(
            embeddings=umap_embeddings,
            labels=sample_for_visualization["event_ids"],
            sample_idx=sample_idx,
            split_name="validation"
        )

        all_metrics = [output["metrics"] for output in self.validation_step_outputs]
        avg_metrics = {
            metric: sum(metrics[metric] for metrics in all_metrics) / len(all_metrics)
            for metric in all_metrics[0]
        }

        self.logger.experiment.add_figure(
            "Embeddings visualization", plt_obj.gcf(), self.current_epoch
        )

        for metric, value in avg_metrics.items():
            self.logger.experiment.add_scalar(f'avg_{metric}', value, self.current_epoch)

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

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

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

from typing import Type, Any, Dict

from .metrics import ClusterMetric, Clustering
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
    dot_product = "DotProductSimilarity"
    snr_distance = "SNRDistance"


class TripletTracksEmbedder(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        triplet_margin: float = 0.2,
        type_of_triplets: TripletType = "semihard",
        distance: DistanceType = DistanceType.cosine_similarity,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2, # default AdamW param
        umapper: Optional[UMAP] = None,
        clustering_algorithm: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'umapper', 'clustering_algorithm'])

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

        if clustering_algorithm is None:
            clustering_algorithm = KMeans(n_clusters=40, random_state=42, n_init=10)
        self.clustering = Clustering(clustering_algorithm)

        self.cluster_metric = ClusterMetric()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

    def validation_step(self, batch, batch_idx):
        loss, embeddings = self._forward_batch(batch, return_embeddings=True)

        emb_np = embeddings.cpu().detach().numpy()
        evt_ids_np = batch[1].cpu().numpy()

        cluster_preds = self.clustering(embeddings, batch[1])
        metrics = self.cluster_metric(embeddings, batch[1], cluster_preds)

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

        avg_metrics = self.cluster_metric.compute()
        self.cluster_metric.reset()
        self.clustering.reset()

        self.logger.experiment.add_figure(
            "Embeddings visualization", plt_obj.gcf(), self.current_epoch
        )

        for metric, value in avg_metrics.items():
            self.logger.experiment.add_scalar(f'avg_{metric}', value, self.current_epoch)

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

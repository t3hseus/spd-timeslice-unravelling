import pytorch_lightning as pl
import torch
from torch import nn
from enum import Enum
from typing import Optional
from umap import UMAP
from pytorch_metric_learning import (
    distances,
    reducers,
    losses,
    miners
)

from .visualization import draw_embeddings


class TripletType(str, Enum):
    all = "all"
    semihard = "semihard"
    easy = "easy"


class TripletTracksEmbedder(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        triplet_margin: float = 0.2,
        type_of_triplets: TripletType = "semihard",
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
        self._distance = distances.CosineSimilarity()
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
                "train_triplets": self.triplet_miner.num_triplets
            },
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, embeddings = self._forward_batch(batch, return_embeddings=True)
        # save predictions for visualization
        self.validation_step_outputs.append({
            "embeddings": embeddings.cpu().numpy(),
            "event_ids": batch[1].cpu().numpy()
        })
        self.log_dict(
            {
                "val_loss": loss,
                "val_triplets": self.triplet_miner.num_triplets
            },
            prog_bar=True
        )

    def on_validation_epoch_end(self):
        # take first batch for validation
        sample_idx = 0
        sample_for_visualization = self.validation_step_outputs[sample_idx]
        umap_embeddings = self.umapper.fit_transform(
            sample_for_visualization["embeddings"])
        plt_obj = draw_embeddings(
            embeddings=umap_embeddings,
            labels=sample_for_visualization["event_ids"],
            sample_idx=sample_idx,
            split_name="validation"
        )
        self.logger.experiment.add_figure(
            "Embeddings visualization", plt_obj.gcf(), self.current_epoch)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

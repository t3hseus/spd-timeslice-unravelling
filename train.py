import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import os
import gin
import logging
import pytorch_lightning as pl

from torch import nn
from absl import app
from absl import flags
from typing import Optional, Callable

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.logging_utils import setup_logger
from src.training import TripletTracksEmbedder, TripletType, DistanceType
from src.dataset import time_slice_collator, SPDTimesliceTracksDataset, DatasetMode

# these imports are needed for gin config
from src.model import TrackEmbedder 
from src.transformations import ConstraintsNormalizer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='config', default=None,
    help='Path to the config file to use.'
)
flags.DEFINE_enum(
    name='log', default='INFO',
    enum_values=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
    help='Level of logging'
)


BATCH_SIZE = 1  # one timeslice in a batch

LOGGER = logging.getLogger("train")


@gin.configurable
def experiment(
        model: nn.Module,
        logging_dir: str = "experiment_logs",
        random_seed: int = 42,
        num_epochs: int = 10,
        train_samples: int = 1000,
        test_samples: int = 10,
        detector_efficiency: float = 1.0,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-2,
        triplet_margin: float = 0.1,
        type_of_triplets: TripletType = TripletType.semihard,
        distance: DistanceType = DistanceType.euclidean_distance,
        hits_normalizer: Optional[Callable] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        # path to checkpoint to resume
        resume_from_checkpoint: Optional[str] = None,
):
    os.makedirs(logging_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(logging_dir, name=model.__class__.__name__)
    setup_logger(LOGGER, tb_logger.log_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{tb_logger.log_dir}",
        filename=f'{{epoch}}-{{step}}'
    )

    with open(os.path.join(tb_logger.log_dir, "train_config.cfg"), "w") as f:
        f.write(gin.config_str())

    LOGGER.info(f"Log directory {tb_logger.log_dir}")
    LOGGER.info(
        "GOT config: \n======config======\n "
        f"{gin.config_str()} "
        "\n========config======="
    )

    LOGGER.info(f"Setting random seed to {random_seed}")
    pl.seed_everything(random_seed)

    LOGGER.info("Preparing datasets for training and validation")
    train_data = SPDTimesliceTracksDataset(
        n_samples=train_samples, 
        detector_eff=detector_efficiency, 
        hits_normalizer=hits_normalizer,
        mode=DatasetMode.train
    )
    test_data = SPDTimesliceTracksDataset(
        n_samples=test_samples, 
        detector_eff=detector_efficiency, 
        hits_normalizer=hits_normalizer,
        mode=DatasetMode.test
    )

    # check both determinism and test != train
    assert train_data[0][0].mean() == train_data[0][0].mean()
    assert test_data[0][0].mean() == test_data[0][0].mean()
    assert train_data[0][0].mean() != test_data[0][0].mean()

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=time_slice_collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=time_slice_collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    LOGGER.info('Creating model for training')
    tracks_embedder = TripletTracksEmbedder(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        triplet_margin=triplet_margin,
        type_of_triplets=type_of_triplets,
        distance=distance
    )
    LOGGER.info(tracks_embedder)

    if resume_from_checkpoint is not None:
        LOGGER.info(f"Resuming from checkpoint {resume_from_checkpoint}")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        deterministic=True,
        accelerator="auto",
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(
        model=tracks_embedder,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )


def main(argv):
    del argv
    gin.parse_config(open(FLAGS.config))
    LOGGER.setLevel(FLAGS.log)
    experiment()
    LOGGER.info("End of training")


if __name__ == "__main__":
    app.run(main)

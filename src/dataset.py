import numpy as np
import torch
from enum import IntEnum
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .data_generation import SPDEventGenerator


class DatasetMode(IntEnum):
    train = 0
    val = 1
    test = 2


def time_slice_collator(x):
    return x[0]  # only one timeslice in a batch


class SPDTimesliceTracksDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        detector_eff: float = 0.98,
        mean_events_timeslice: int = 30,
        fixed_num_events: bool = False,
        hits_normalizer: Optional[Callable] = None,
        mode: DatasetMode = DatasetMode.train
    ):
        self.spd_gen = SPDEventGenerator(
            mean_events_timeslice=mean_events_timeslice,
            fixed_num_events=fixed_num_events,
            detector_eff=detector_eff,
            add_fakes=False
        )
        self._n_samples = n_samples
        self.hits_normalizer = hits_normalizer
        # get initial random seed for reproducibility
        # mode helps to ensure that datasets don't intersect
        self._initial_seed = np.random.get_state()[1][mode]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        # prevent dataset from generation of new samples each epoch
        np.random.seed(self._initial_seed + idx)
        # generate sample
        time_slice = self.spd_gen.generate_time_slice()
        _, uniq_track_ids_counts = np.unique(
            time_slice["track_ids"], return_counts=True)

        if self.hits_normalizer:
            time_slice["hits"] = self.hits_normalizer(time_slice["hits"])

        # split hits array by tracks and convert to tensors
        tracks_by_hits = torch.split(
            torch.tensor(time_slice["hits"]).reshape(-1),
            (uniq_track_ids_counts*3).tolist()
        )
        # pad short tracks by zeros and create single matrix N_tracks x N_stations*3
        tracks = pad_sequence(tracks_by_hits).T
        # create event label for each track
        labels = torch.tensor(
            time_slice["event_ids"][np.cumsum(uniq_track_ids_counts)-1])
        return tracks, labels

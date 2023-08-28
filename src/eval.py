import torch
from sklearn.cluster import KMeans
from src.model import TrackEmbedder
from src.clustering import Clustering
from src.logging_utils import setup_logger
from src.training import TripletTracksEmbedder, TripletType, DistanceType
from src.dataset import time_slice_collator, SPDTimesliceTracksDataset, DatasetMode
from torch.utils.data import DataLoader
from src.visualization import draw_embeddings
from src.metrics import *
import src.utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib
from tqdm.auto import tqdm
import time
import gin
import re
import os


class ModelEvaluator:
    def __init__(self, config, parsed_gin):
        self.base_path = config.get("base_path", "./experiment_logs/TrackEmbedder/")
        self.run_name = config.get("run_name", "version_0/")
        self.model_path = config.get("model_path", "epoch=99-step=100000.ckpt")
        self.cfg_path = config.get("cfg_path", 'train_config.cfg')
        self.run_path = os.path.join(self.base_path, self.run_name, self.cfg_path)
        self.res = parsed_gin
        self.BATCH_SIZE = config.get("BATCH_SIZE", 1)
        self.n_samples = config.get("n_samples", 10)
        self.map_location = config.get("map_location", 'mps')
        self.model = self._initialize_model()
        self.clustering_algorithm = self._initialize_clustering_model()
        self.metrics = self.res['experiment']['metrics']
        self.dataset = self.generate_data()
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            collate_fn=time_slice_collator,
            num_workers=2,
            pin_memory=self.res['experiment']['pin_memory'],
        )

    def _initialize_model(self):
        model = self.res['experiment']['model']
        model_state = torch.load(os.path.join(
            self.base_path, self.run_name, self.model_path), map_location=self.map_location)
        fixed_state_dict = {k.replace("model.", ""): v for k,
                            v in model_state['state_dict'].items()}
        model.load_state_dict(fixed_state_dict)
        model.eval()
        return model

    def _initialize_clustering_model(self):
        clust = self.res['clustering_algorithm']
        params = {k: v for k, v in clust.items() if k not in ['class_', 'module']}
        module = importlib.import_module(clust['module'])
        cls = getattr(module, clust['class_'])
        return Clustering(cls(**params))

    def generate_data(self):
        return SPDTimesliceTracksDataset(
            n_samples=self.n_samples,
            detector_eff=self.res['experiment']['detector_efficiency'],
            hits_normalizer=self.res['experiment']['hits_normalizer'],
            mode=DatasetMode.val,
        )

    def visualize_embeddings(self, num_samples=6):

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'UMAP plots for the first {num_samples} samples')

        for i, batch in enumerate(self.data_loader):
            if i >= num_samples:
                break

            tracks, evt_ids = batch
            embeddings = self.model(tracks).detach().numpy()
            evt_ids_np = evt_ids.detach().numpy()

            ax = axes[i // 3, i % 3]
            plt.sca(ax)

            draw_embeddings(embeddings=embeddings, labels=evt_ids_np, sample_idx=i, split_name='')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(os.path.join(self.base_path,self.run_name,"embeddings_subplot.png"))

    def evaluate(self):
        embedding_times = []
        clustering_times = []
        metrics_results = {}

        for batch in tqdm(self.data_loader):
            tracks, evt_ids = batch
            start_time = time.time()
            embeddings = self.model(tracks)
            end_time = time.time()
            embedding_times.append(end_time - start_time)

            emb_np = embeddings.detach().numpy()
            evt_ids_np = evt_ids.detach().numpy()
            start_time = time.time()
            cluster_preds = self.clustering_algorithm.cluster_and_link(emb_np, evt_ids_np)
            end_time = time.time()
            clustering_times.append(end_time - start_time)

            for metric in self.metrics:
                if isinstance(metric, (
                    F1ScoreMetric,
                    PrecisionScoreMetric,
                    RecallScoreMetric,
                    AccuracyScoreMetric
                )):
                    metric.update(evt_ids_np, cluster_preds)
                elif isinstance(metric, BaseScoreMetric):
                    metric.update(emb_np, cluster_preds)

        avg_embedding_time = np.mean(embedding_times)
        avg_clustering_time = np.mean(clustering_times)

        for metric in self.metrics:
            metric_name = type(metric).__name__
            metric_val = metric.compute()
            metrics_results[metric_name] = metric_val

        results = {
            'n_samples': self.n_samples,
            'avg_time': {
                'emb_time': avg_embedding_time,
                'cluster_time': avg_clustering_time
            },
            'metrics': metrics_results
        }

        df_time_res = pd.DataFrame({'emb_time': embedding_times, 'cluster_time': clustering_times})


        return results, df_time_res.describe()
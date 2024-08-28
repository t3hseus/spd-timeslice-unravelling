# SPD Timeslice Unravelling project

## Getting started

### 1. Prepare environment
```bash
make env
conda activate spd_triplet_embedder
```

### 2. Train the model
```bash
make train
```

### 3. Evaluate the model
```bash
make eval MODEL_DIR="./experiment_logs/TrackEmbedder/version_0"
```

To check Tensorboard logs:
```bash
tensorboard --logdir experiment_logs
```

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import src.utils
from src.eval import ModelEvaluator
import os
import json


if __name__ == "__main__":
    base_path = "./experiment_logs/TrackEmbedder/" # set your base path here
    run_name = "version_0/" # set your run name here
    cfg_path = 'train_config.cfg'
    run_path = os.path.join(base_path, run_name, cfg_path)
    parsed_gin = src.utils.parse_gin(run_path)

    config = {
        "base_path": base_path,
        "run_name": run_name,
        "model_path": "epoch=99-step=100000.ckpt",
        "cfg_path": cfg_path,
        "BATCH_SIZE": 1,
        "n_samples": 10,
        "map_location": 'cpu' # set your device here
    }

    evaluator = ModelEvaluator(config, parsed_gin)
    # evaluator.visualize_embeddings(num_samples=6) # not implemented
    result, df_time = evaluator.evaluate()

    print(result)

    df_time.to_csv(os.path.join(base_path, run_name, 'eval_time.csv'))

    with open(os.path.join(base_path, run_name, 'eval_res.json'), 'w') as f:
        json.dump(result, f)

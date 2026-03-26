import numpy as np
import torch
import os

# Update these to include the package prefix
from imbalanceddl.utils.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.utils.config import get_args

# Note the change in path for selection_method, which is inside strategy
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.strategy.selection_method.random_selection import random_selection
from imbalanceddl.dataset.lava_dataset import LavaDataset

def main():
    # 1. Load Configuration
    config = get_args()
    
    # 2. Setup Logging and Folders
    prepare_store_name(config)
    print(f"=> Store Name = {config.store_name}")
    prepare_folders(config)

    # 3. Seed for Reproducibility
    if config.seed is None:
        config.seed = np.random.randint(10000)
    fix_all_seed(config.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # 4. Build Model
    model = build_model(config)

    # 5. Build Initial Dataset
    imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset)

    if config.ratio < 1.0:
        print(f"=> Applying Data Selection: {config.method} (Ratio: {config.ratio})")
        imbalance_dataset = LavaDataset(
            config, 
            imbalance_dataset, 
            config.ratio, 
            config.method, 
            device=device
        )

    # 7. Build Trainer
    trainer = build_trainer(config,
                            imbalance_dataset,
                            model=model,
                            strategy=config.strategy)

    # 8. Execution
    if config.best_model is not None:
        print("=> Eval with Best Model !")
        trainer.eval_best_model()
    else:
        print("=> Start Train Val !")
        if config.strategy == 'M2m':
            trainer.do_train_val_m2m()
        else:
            trainer.do_train_val()
            
    print("=> All Completed !")

if __name__ == "__main__":
    main()
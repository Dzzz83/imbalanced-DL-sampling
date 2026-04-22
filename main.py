import sys
from unittest.mock import MagicMock
import logging
def silence_torchtext():
    """Bypasses the C++ linkage error in torchtext for image-only projects."""
    modules_to_mock = [
        "torchtext", 
        "torchtext.data", 
        "torchtext.data.utils", 
        "torchtext.datasets", 
        "torchtext.vocab"
    ]
    for mod in modules_to_mock:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

silence_torchtext()

import numpy as np
import torch
import os

from imbalanceddl.utils.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.utils.config import get_args

from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.strategy.selection_method.random_selection import random_selection
from imbalanceddl.dataset.lava_dataset import LavaDataset

def main():
    # 1. Load Configuration
    config = get_args()
    config.strategy ="DeepSMOTE_Selection"
    ratios =[0.7,0.8,0.9,1.0]
    for ratio in ratios:
        config.selection_ratio = ratio
        # 2. Setup Logging and Folders
        print(f"DEEP SMOTE TRAINING WITH  SELECTION RATIO+{ratio}")
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
        print(f"Creating training dataset with {config.augmentation} augmentation...")
        imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset, augmentation=config.augmentation)

        # Skip automatic data selection for DeepSMOTE_Selection (handles it internally)
        if config.strategy == "DeepSMOTE_Selection":
            print("=> DeepSMOTE_Selection handles selection internally. Skipping main script selection.")
        else:
            if config.selection_ratio < 1.0:
                print(f"=> Applying Data Selection: {config.selection_method} (Ratio: {config.selection_ratio})")
                imbalance_dataset = LavaDataset(
                    config, 
                    imbalance_dataset, 
                    config.selection_ratio, 
                    config.selection_method, 
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
        logging.shutdown() 

if __name__ == "__main__":
    main()
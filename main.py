import sys
from pathlib import Path

# Prioritize sava's otdd over LAVA's
sava_root = Path(__file__).parent / 'sava'
if str(sava_root) not in sys.path:
    sys.path.insert(0, str(sava_root))
# Remove any cached otdd module (from previous imports)
if 'otdd' in sys.modules:
    del sys.modules['otdd']

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

from imbalanceddl.utils.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.utils.config import get_args

# Only keep SavaDataset (random selection is handled inside SavaDataset)
from imbalanceddl.dataset.sava_dataset import SavaDataset

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

    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if hasattr(config, 'gpu') and config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            print(f"=> Using GPU {config.gpu}")

    # 4. Build Model
    model = build_model(config)
    
    # 5. Build Initial Dataset
    print(f"Creating training dataset with {config.augmentation} augmentation...")
    imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset, augmentation=config.augmentation)

    # 6. Data Selection (if ratio < 1.0 and method is sava or random)
    if config.strategy in ["DeepSMOTE_Selection", "RandomOversampling_Selection", "Selection_RandomOversampling", 
                           "DeepSMOTE_Sava"]:
        print(f"=> {config.strategy} handles selection internally. Skipping main script selection.")
    else:
        if config.selection_ratio < 1.0:
            print(f"=> Applying Data Selection: {config.selection_method} (Ratio: {config.selection_ratio})")
            if config.selection_method == 'sava':
                imbalance_dataset = SavaDataset(
                    config, imbalance_dataset, config.selection_ratio,
                    method='sava', device=device
                )
            elif config.selection_method == 'random':
                imbalance_dataset = SavaDataset(
                    config, imbalance_dataset, config.selection_ratio,
                    method='random', device=device
                )
            elif config.selection_method == 'none':
                print("=> selection_method = 'none', using full dataset.")
            else:
                raise ValueError(f"Unknown selection method: {config.selection_method}. Use 'sava', 'random', or 'none'.")
        else:
            print("=> selection_ratio == 1.0, using full dataset (no selection).")

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
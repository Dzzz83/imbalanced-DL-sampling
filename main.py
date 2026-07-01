import sys
from pathlib import Path

# Prioritize sava's otdd over LAVA's
sava_root = Path(__file__).parent / 'sava'
if str(sava_root) not in sys.path:
    sys.path.insert(0, str(sava_root))
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
from imbalanceddl.utils.debug_logger import get_debug_logger
from imbalanceddl.dataset.sava_dataset import SavaDataset

def main():
    config = get_args()
    
    # Explicitly define num_classes for downstream components
    if config.dataset in ['cifar10', 'cinic10', 'svhn10', 'cifar10_noisy']:
        config.num_classes = 10
    elif config.dataset == 'cifar100':
        config.num_classes = 100
    elif config.dataset == 'tiny200':
        config.num_classes = 200
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not mapped to num_classes.")

    # Override batch size if training the 3 experts
    if config.strategy == 'Experts':
        config.batch_size = config.expert_batch_size
        print(f"=> Overriding batch size to {config.batch_size} for Expert training.")

    prepare_store_name(config)
    print(f"=> Store Name = {config.store_name}")
    prepare_folders(config)

    if getattr(config, 'debug', False):
        get_debug_logger(debug=True)
        logger = get_debug_logger(debug=True)
        logger.debug("Debug logging enabled for this run.")
    else:
        get_debug_logger(debug=False)

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

    model = build_model(config)
    
    print(f"Creating training dataset with {config.augmentation} augmentation...")
    imbalance_dataset = ImbalancedDataset(config, dataset_name=config.dataset, augmentation=config.augmentation)

    if config.strategy in ["DeepSMOTE_Selection", "RandomOversampling_Selection", "Selection_RandomOversampling", 
                           "DeepSMOTE_Sava"]:
        print(f"=> {config.strategy} handles selection internally. Skipping main script selection.")
    else:
        if config.selection_method == 'sava':
            print(f"=> Applying SAVA scoring (ratio={config.selection_ratio})")
            imbalance_dataset = SavaDataset(
                config, imbalance_dataset, config.selection_ratio,
                method='sava', device=device
            )
        elif config.selection_method == 'random' and config.selection_ratio < 1.0:
            print(f"=> Applying random selection (ratio={config.selection_ratio})")
            imbalance_dataset = SavaDataset(
                config, imbalance_dataset, config.selection_ratio,
                method='random', device=device
            )
        elif config.selection_method == 'none' or config.selection_ratio >= 1.0:
            print("=> No selection or ratio = 1.0, using full dataset.")
        else:
            raise ValueError(f"Unknown selection method: {config.selection_method}. Use 'sava', 'random', or 'none'.")

    trainer = build_trainer(config,
                            imbalance_dataset,
                            model=model,
                            strategy=config.strategy)

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
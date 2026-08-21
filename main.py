import logging
import numpy as np
import torch

from imbalanceddl.utils.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.net.network import build_model
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.build_trainer import build_trainer
from imbalanceddl.utils.config import get_args
from imbalanceddl.utils.debug_logger import get_debug_logger

def main():
    config = get_args()
    
    if config.dataset in ['cifar10', 'cinic10', 'svhn10', 'cifar10_noisy']:
        config.num_classes = 10
    elif config.dataset == 'cifar100':
        config.num_classes = 100
    elif config.dataset == 'tiny200':
        config.num_classes = 200
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not mapped to num_classes.")

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

    # Inject cls_num_list into config for global access
    train_set = imbalance_dataset.train_val_sets[0]
    if hasattr(train_set, 'get_cls_num_list'):
        config.cls_num_list = train_set.get_cls_num_list()
    else:
        targets = np.array(train_set.targets)
        config.cls_num_list = np.bincount(targets, minlength=config.num_classes).tolist()

    trainer = build_trainer(config, imbalance_dataset, model=model, strategy=config.strategy)

    if config.best_model is not None:
        print("=> Eval with Best Model !")
        trainer.eval_best_model()
    else:
        print("=> Start Train Val !")
        trainer.do_train_val()
            
    print("=> All Completed !")
    logging.shutdown()

if __name__ == "__main__":
    main()
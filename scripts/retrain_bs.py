import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imbalanceddl.utils.config import get_args
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.strategy.retrain_bs import BSRetrainer

def main():
    cfg = get_args()
    dataset = ImbalancedDataset(cfg, cfg.dataset, augmentation='weak')
    trainer = BSRetrainer(cfg, dataset)
    trainer.do_train_val()

if __name__ == "__main__":
    main()
from imbalanceddl.strategy._experts import ExpertsTrainer
from imbalanceddl.strategy._gate_trainer import GateTrainer

def build_trainer(cfg, imbalance_dataset, model=None, strategy=None):
    """
    Build various strategy (trainer) specified by users
    """
    if strategy == 'Experts':
        print("=> 3-Expert Trainer !")
        trainer = ExpertsTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)

    elif strategy == 'Gate':
        print("=> Gate Trainer (Stage 2) !")
        trainer = GateTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
        
    else:
        raise NotImplementedError(f"Strategy {strategy} not recognized or deleted.")
    return trainer
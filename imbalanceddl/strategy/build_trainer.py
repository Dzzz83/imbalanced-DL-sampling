from imbalanceddl.strategy import MixupTrainer
from imbalanceddl.strategy import RemixTrainer
from imbalanceddl.strategy import MAMixTrainer
from imbalanceddl.strategy import ERMTrainer
from imbalanceddl.strategy import DRWTrainer
from imbalanceddl.strategy import LDAMDRWTrainer
from imbalanceddl.strategy import ReweightCBTrainer
from imbalanceddl.strategy import M2mTrainer
from imbalanceddl.strategy import DeepSMOTETrainer
from imbalanceddl.strategy._experts import ExpertsTrainer

def build_trainer(cfg, imbalance_dataset, model=None, strategy=None):
    """
    Build various strategy (trainer) specified by users
    """
    if strategy == 'Mixup_DRW':
        print("=> Mixup Trainer !")
        trainer = MixupTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'Mixup':
        trainer = MixupTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'M2m':
        print("=> M2m Trainer !")
        trainer = M2mTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'DeepSMOTE':
        print("=> DeepSMOTE Trainer !")
        trainer = DeepSMOTETrainer(cfg, imbalance_dataset, model=model, strategy=strategy)                                    
    elif strategy == 'Remix_DRW':
        print("=> Remix Trainer !")
        trainer = RemixTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'MAMix_DRW':
        print("=> MAMix Trainer !")
        trainer = MAMixTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'ERM':
        print("=> ERM Trainer !")
        trainer = ERMTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'DRW':
        print("=> DRW Trainer !")
        trainer = DRWTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'LDAM_DRW':
        print("=> LDAM_DRW Trainer !")
        trainer = LDAMDRWTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'Reweight_CB':
        print("=> Reweight_CB Trainer !")
        trainer = ReweightCBTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == "DeepSMOTE_Sava":
        from imbalanceddl.strategy._deepsmote_sava import DeepSMOTESavaTrainer
        print("=> DeepSMOTE + SAVA Trainer !")
        trainer = DeepSMOTESavaTrainer(cfg, imbalance_dataset, model, strategy)    
    elif strategy == 'RandomOversampling_Selection':
        from imbalanceddl.strategy._randOversampling import RandomOversamplingTrainer
        print("=> RandomOversampling + SAVA Trainer !")
        trainer = RandomOversamplingTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
    elif strategy == 'SAVA_Reweight':
        from imbalanceddl.strategy._sava_reweight import SAVAReweightTrainer
        print("=> SAVA Reweight Trainer !")
        trainer = SAVAReweightTrainer(cfg, imbalance_dataset, model, strategy)

    elif strategy == 'ClassBalanced_ERM':
        from imbalanceddl.strategy._class_balanced_erm import ClassBalancedERMTrainer
        print("=> ClassBalanced ERM Trainer (sqrt class balance only) !")
        trainer = ClassBalancedERMTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)

    elif strategy == 'ClassBalanced_ERM_DRW':
        from imbalanceddl.strategy._class_balance_erm_drw import ClassBalancedERM_DRW_Trainer
        print("=> ClassBalanced ERM DRW Trainer !")
        trainer = ClassBalancedERM_DRW_Trainer(cfg, imbalance_dataset, model=model, strategy=strategy)
        
    elif strategy == 'SAVA_Reweight_DRW':
        from imbalanceddl.strategy._sava_reweight_drw import SAVAReweightDRWTrainer
        print("=> SAVA Reweight DRW Trainer !")
        trainer = SAVAReweightDRWTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)

    elif strategy == 'SAVA_Mixup_DRW':
        from imbalanceddl.strategy._sava_mixup_drw import SAVAMixupDRWTrainer
        print("=> SAVA Mixup DRW Trainer !")
        trainer = SAVAMixupDRWTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
        
    elif strategy == 'Experts':
        print("=> 3-Expert Trainer !")
        trainer = ExpertsTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)

    elif strategy == 'Gate':
        from imbalanceddl.strategy._gate_trainer import GateTrainer
        print("=> Gate Trainer (Stage 2) !")
        trainer = GateTrainer(cfg, imbalance_dataset, model=model, strategy=strategy)
        
    else:
        raise NotImplementedError
    
    

    return trainer
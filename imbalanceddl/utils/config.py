import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(
        add_help=False, description='PyTorch Deep Imbalanced Training')

    # Load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    # Default settings
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # Imbalance dataset
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset to use')
    parser.add_argument('--imb_type', default="exp", type=str, choices=['exp', 'step'], help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    # Classifier type
    parser.add_argument('--classifier', default='dot_product_classifier', type=str,
                        choices=['dot_product_classifier', 'cosine_classifier'],
                        help='Type of classifier head to use')
    # Strategy
    parser.add_argument('--strategy', default="ERM", type=str, 
                    choices=['ERM', 'DRW', 'LDAM_DRW', 'Mixup_DRW', 'Remix_DRW',
                            'Reweight_CB', 'MAMix_DRW', 'M2m', 'DeepSMOTE', 
                            'DeepSMOTE_LAVA', 'LAVA_Reweight', 'ClassBalanced_ERM', 
<<<<<<< HEAD
                            'ClassBalanced_ERM_DRW', 'SAVA_Reweight_DRW', 'SAVA_Mixup_DRW', 'Experts'],
=======
                            'ClassBalanced_ERM_DRW', 'SAVA_Reweight_DRW', 'SAVA_Mixup_DRW', 'Experts', 'Gate'],
>>>>>>> expert
                        help='select strategy for trainer')
    parser.add_argument('--base_strategy', default='ERM', type=str,
                    choices=['ERM', 'Mixup', 'DRW', 'LDAM_DRW', 'Reweight_CB'],
                    help='Base strategy for two-stage methods')
    
    # CRISP Paper Specifications for CIFAR-100-LT
    parser.add_argument('--learning_rate', default=0.4, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    
    # Seed
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

    # M2m
    parser.add_argument('--resume', '-r', action='store_false',help='resume from checkpoint')
    parser.add_argument('--net_g', default=None, type=str, help='checkpoint path of network for generation')
    parser.add_argument('--net_g2', default=None, type=str, help='checkpoint path of network for generation')
    parser.add_argument('--net_t', default=None, type=str, help='checkpoint path of network for train')
    parser.add_argument('--net_both', default=None, type=str, help='checkpoint path of both networks')
    parser.add_argument('--backbone', default='resnet32', type=str, help='model type (default: ResNet18)')
    parser.add_argument('--classifier', default='dot_product_classifier', type=str, help='classifier type')
    parser.add_argument('--effect_over', action='store_true', help='Use effective number in oversampling')
    parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')
    parser.add_argument('--gen', '-gen', action='store_false', help='')
    parser.add_argument('--warm', default=160, type=int, help='Deferred strategy for re-balancing')
    parser.add_argument('--epochs', default=256, type=int,help='total epochs to run')
    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'Focal', 'LDAM'], help='Type of loss for imbalance')
    parser.add_argument('--reweight', '-reweight', action='store_true', help='oversampling')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
    parser.add_argument('--attack_iter', default=10, type=int, help='')
    parser.add_argument('--lam', default=0.5, type=float, help='Hyper-parameter for regularization of translation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Threshold of the generation')
    parser.add_argument('--beta', default=0.999, type=float, help='Hyper-parameter for rejection/sampling')
    parser.add_argument('--step_size', default=0.1, type=float, help='')
    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')

    # Log
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    # Assign GPU
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    # Evaluation with Best Model
    parser.add_argument('--best_model', default=None, type=str, metavar='PATH', help='Path to Best Model')
    
    # Sampling
    parser.add_argument('--sampling', default='Random', type=str, help='For Balance - Sampler to use')
    parser.add_argument('--batch_size', default=128, type=int, help='For Balance - batch size')
    parser.add_argument('--n_batches', default=400, type=int, help='For Balance - number of batches per epoch')
    parser.add_argument('--alpha', default=0.5, type=float, help='For Balance - alpha')
    parser.add_argument('--kind', default='random', type=str, help='For Balance - Kind of sampler')

    # Data Selection Method
    parser.add_argument('--selection_ratio', default=1.0, type=float, help='Ratio of data to keep after selection')
    # noise ratio
    parser.add_argument('--noise_ratio', default=0.0, type=float, help='Ratio of label noise (0.0 to 1.0) for noisy datasets')
    # mamix ratio
    parser.add_argument('--mamix_ratio', default=1.0, type=float, help='MAMix interpolation ratio')
    
    # Mixup Alpha
    parser.add_argument('--mixup_alpha', default=1.0, type=float, help='Mixup interpolation alpha')
    
    # Augmentation
    parser.add_argument('--augmentation', default='weak', type=str, help='Select the augmentation')
    
    # Rand Number
    parser.add_argument('--rand_number', default=0, type=int, help='Rand Number')
    
    # Device
    parser.add_argument('--device', default='cuda', type=str, help='Device for computation (cuda/cpu)')

    # Sample Cap
    parser.add_argument('--cap_per_class', default=None, type=int, help='Cap each class to this many samples (for balanced datasets)')
    
    # Noise order flag
    parser.add_argument('--noise_first', action='store_true', help='Inject label noise before oversampling (instead of after)')

    # Selection method
    parser.add_argument('--selection_method', default='none', type=str,
                        choices=['lava', 'random', 'none', 'sava'], help='Method for data selection/filtering')

    # SAVA parameters
    parser.add_argument('--sava_batch_size', default=1024, type=int, help='Batch size for SAVA hierarchical OT (default: 1024)')
    parser.add_argument('--sava_cache_label_distances', default=True, type=bool, help='Cache label-to-label OT distances across batches')
    
    # Debug 
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug prints')

    # Save checkpoint
    parser.add_argument('--save_checkpoint', action='store_true', help='Save model checkpoints (default: False)')
    parser.add_argument('--save_interval', default=50, type=int, help='Save checkpoint every N epochs (in addition to best)')
    
    # wandb
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging (default: False)')

    # SAVA Reweighting
    parser.add_argument('--reweight_mode', default='loss', type=str, choices=['loss', 'sampler'], help='How to apply SAVA weights: loss weighting or weighted sampler')
    parser.add_argument('--sava_reweight_temp', default=1.0, type=float, help='Temperature for exponential scaling of SAVA scores (higher = softer)')
    parser.add_argument('--sava_scores_file', default=None, type=str, help='Path to precomputed SAVA scores .npy file (optional)')
    parser.add_argument('--sava_weights_clip', default=1e-3, type=float, help='Clip weights to this minimum value to avoid extreme values')

    # Experts Strategy Parameters
    parser.add_argument('--expert_batch_size', default=256, type=int, help='Batch size for expert training')
    parser.add_argument('--gating_batch_size', default=128, type=int, help='Batch size for gating training')

    # Gate parameters
    parser.add_argument('--lambda_ent', default=0.01, type=float, help='Entropy regularization coefficient for gate')
    parser.add_argument('--lambda_bal', default=0.05, type=float, help='Balance regularization coefficient for gate')
    parser.add_argument('--gate_lr', default=1e-3, type=float, help='Learning rate for gate optimizer')
    parser.add_argument('--gate_weight_decay', default=1e-4, type=float, help='Weight decay for gate optimizer')
    parser.add_argument('--gate_epochs', default=100, type=int, help='Number of epochs for gate training')
    parser.add_argument('--gate_hidden_size', default=256, type=int, help='Hidden size of gate MLP (first layer)')
    parser.add_argument('--gate_hidden_size2', default=128, type=int, help='Second hidden size of gate MLP')
    parser.add_argument('--gate_split_ratio', default=0.9, type=float, help='Fraction of training data for expert training; remaining for gate')
    parser.add_argument('--routing_sparsity', default=2, type=int, help='Number of top experts to keep (k)')
    parser.add_argument('--expert_checkpoint', default=None, type=str, help='Path to pre-trained expert model checkpoint for gate training')    
    
    # Stage 3: Plug-in Rule parameters
    parser.add_argument('--plugin_algo', default='Bal', type=str, choices=['Bal', 'Worst'], help='Plug-in algorithm to use for Stage 3 (Bal or Worst)')
    
    # For ultra_debug.py
    parser.add_argument('--experts_dir', type=str, default=None, help='Directory containing expert_0.pth, expert_1.pth, expert_2.pth')
    parser.add_argument('--gate_ckpt', type=str, default=None, help='Path to gate checkpoint file')

    # FIX: Added gate_dropout and the expert specific loading parameters
    parser.add_argument('--gate_dropout', default=0.0, type=float, help='Dropout rate for gate MLP')
    parser.add_argument('--ce_bias', default=False, type=bool, help='Bias setting for CE expert')
    parser.add_argument('--la_bias', default=False, type=bool, help='Bias setting for LA expert')
    parser.add_argument('--la_tau', default=1.5, type=float, help='Tau setting for LA expert')
    parser.add_argument('--bs_bias', default=False, type=bool, help='Bias setting for BS expert')

    # FIX: Added sweep parameters
    parser.add_argument('--gate_batch_sizes', nargs='+', type=int, default=[128], help='List of batch sizes to sweep for the gate')
    parser.add_argument('--gate_temperatures', nargs='+', type=float, default=[1.0], help='List of temperatures to sweep for the gate')
    parser.add_argument('--eval_interval', default=10, type=int, help='Evaluate gate every N epochs to find the best one')
    
    # update config from command line
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args
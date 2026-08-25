# Project Blueprint: Imbalanced-DL-Sampling (MoE Routing)

## 1. PROJECT IDENTITY & STACK
- **Name:** Imbalanced-DL-Sampling
- **Goal:** Address class imbalance in deep learning by training a Mixture of Experts (CE, Logit Adjusted, Balanced Softmax) and developing a routing mechanism to direct head samples to CE and tail samples to LA.
- **Current Phase:** Stage 2 - Diagnostic phase [COMPLETE]; transitioning to new routing strategy design.
- **Diagnostic Status (2025-07-22):** Six experiments (Exp 12-16) have conclusively identified the routing bottleneck as an **SNR-to-sample-size mismatch**: the three 100-dim L2-normalized probability blocks have moderate per-sample correlation (r approx 0.68), not strict collinearity, and each has high effective rank (~75/100). However, 70% of total 316-dim feature variance is shared across experts (top 5 PCs), and the residual expert-discriminative routing signal (~30% variance spread across ~66 components) is too diluted for the gate's 1,125 training samples to extract. The gate converges to per-class biases (from 4 frequency features) but cannot learn per-sample routing. No target function (mix_nll, logprob, correctness) or architecture (MLP, linear) can overcome this within the 316-dim calibrated probability feature space. Recommended next strategies: (1) DaWin-style confidence routing (3-dim, training-free), or (2) penultimate feature routing (192-dim embeddings).
- **Language:** Python 3.x
- **Framework:** PyTorch, torchvision
- **Key Libraries:** `scikit-learn`, `numpy`, `pyyaml`
- **Hardware Target:** CUDA GPUs (configurable via `--gpu` flag)
- **Repository Structure:** Modularized PyTorch project with specific directories for datasets, loss functions, network architectures, training strategies, and utility functions, driven by a central entry point and YAML configurations.

## 2. ARCHITECTURE OVERVIEW
- **Design Pattern:** Strategy Pattern with Factory

**Core Components:**
- **ExpertEnsemble:** A wrapper module that loads 3 frozen ResNet32 models (trained with CE, LA, and BS losses respectively).
- **GateMLP:** A routing network that takes expert outputs (calibrated probability features: 316-dim) and outputs 3 routing weights. **As of Exp 14**: supports `linear_router=True` (Linear(316,3), recommended — 951 params) and `linear_router=False` (MLP: BN→Linear(316,64)→ReLU→Linear(64,3), 20k params). Linear router matches the ~1,125-sample gate training set capacity.
- **Strategy Trainers:** 
  - `ExpertsTrainer` (Stage 1): Trains individual models.
  - `GateTrainer` (Stage 2): Freezes the experts and trains the `GateMLP`.
- **Loss Functions:** Custom implementations of `LogitAdjustedLoss` and `BalancedSoftmaxLoss` used in Stage 1.

**Data Flow:** 
`main.py` parses YAML config -> `ImbalancedDataset` creates imbalanced CIFAR-100 train/val sets (90/10 split) -> `build_trainer` initializes `GateTrainer` -> `GateTrainer._split_dataset` splits the 10% train data further for gate training (~1,125 samples) -> Strategy trainer executes train/val loop -> Checkpoints saved to `checkpoint/gate_cifar100_new/` -> `ultra_debug.py` runs Stage 3 evaluation and debugging.

**Network Structure:**

### Candidate Routing Designs (Post-Diagnostic, Under Evaluation)

The diagnostic phase (Exp 12–16) proved the 316-dim calibrated probability feature space has an **SNR bottleneck**: 70% of variance is shared across experts, and the residual routing signal is too diluted for 1,125 training samples. The following candidate designs replace the current `GateMLP` input and have been ranked by feasibility, theoretical fit, and implementation risk.

| Rank | Design | Input Dim | Params | Training Required | Key Advantage |
|:----:|:-------|:---------:|:------:|:-----------------|:--------------|
| **1** | **DaWin confidence routing** | 3 (max-prob per expert) | 0 (just scalar T̂) | No (post-hoc) | Bypasses 316-dim space entirely; proven to beat learned routers for frozen experts |
| **2** | **Penultimate feature routing** | 192 (3 × 64-dim embeddings) | 579 (linear) | Yes (1,125 samples) | Richer features, lower expected cross-expert correlation |
| **3** | **Stats-only logistic routing** | 13 (9 stats + 4 freq) | 42 (linear) | Yes (1,125 samples) | Tests whether 300 prob dims contribute any signal |

See `experiment-log.md` → PLANNED EXPERIMENTS for the ordered evaluation plan with decision tree.

## 3. DETAILED FILE MAP & UTILITIES

**Module: `imbalanceddl/strategy/`**
- `base.py`: Defines `BaseTrainer` abstract class handling dataset parsing, dataloaders, logging, and standard metric computation.
- `_experts.py`: Defines `ExpertsTrainer` for Stage 1.
- `_gate_trainer.py`: Defines `ExpertEnsemble`, `GateMLP` (supports linear_router flag), and `GateTrainer` for Stage 2. Freezes experts, trains a routing network, and evaluates Stage 3 plugin rules.
- `build_trainer.py`: Factory function that instantiates the correct trainer based on the `strategy` parameter in the config.

**Module: `imbalanceddl/loss/`**
- `loss.py`: Contains `LogitAdjustedLoss`, `BalancedSoftmaxLoss`, `LDAMLoss`, and `FocalLoss`.

**Module: `imbalanceddl/net/`**
- `network.py`: Defines `Network` class combining backbone + classifier head. `build_model()` factory.
- `resnet_cifar.py`: ResNet implementations optimized for CIFAR (e.g., `resnet32`). Outputs 64-dim features.

**Module: `imbalanceddl/utils/`**
- `config.py`: `get_args()` parses YAML and command-line arguments.
- `metrics.py`: `shot_acc()` calculates head/medium/tail accuracy.
- `plugin_rule.py`: Stage 3 selective prediction logic.
- `debug/`: Scripts for deep evaluation (`ultra_debug.py` dependencies).

## 4. DATA PIPELINE & TRAINING STRATEGIES

**Data Loading Flow:**
`ImbalancedDataset` → `IMBALANCECIFAR100` (creates imbalance) → `get_weak_augmentation` → `DataLoader` → `BaseTrainer` / `GateTrainer`

**Sampling Strategies:**
- `BaseTrainer` creates the initial train loader with `shuffle=True` (Instance-balanced).
- `GateTrainer._split_dataset` uses a `WeightedRandomSampler` with inverse class frequency weights to balance the Head/Tail classes in the gate training data.

## 5. CONFIGURATION & CONVENTIONS

**Entry Point:** `python main.py --config config/what_to_train/cifar100/_gate_train.yaml`

**Main Configs:** 
- `config/what_to_train/cifar100/_experts_train_90.yaml` (Stage 1)
- `config/what_to_train/cifar100/_gate_train.yaml` (Stage 2)

**Key Parameters (from _gate_train.yaml):**
- `strategy`: Selects trainer (`Experts` or `Gate`)
- `imb_factor`: Imbalance ratio (default `0.01`)
- `gate_split_ratio`: Train/val split for gate data (default `0.9` -> 10% gate split)
- `la_tau`: Temperature for Logit Adjustment (default `1.5`)

**Testing:**
- `ultra_debug.py`: Primary verification script. Runs full Stage 3 evaluation (Tables 2, 3), per-class routing analysis, LA-saves-the-day checks, oracle diagnostic, and Stage 3 plug-in parameter tuning. Requires `weights_only=False` in `torch.load` for PyTorch 2.6+ compatibility.
- `inspect_gate_gradients.py`: Replicates the checkpoint's training loss on one batch and reports the gradient norm of `gate.fc.weight` (benchmarks: 0.251 for mix_nll starvation, 0.892 for logprob healthy).
- `diagnose_feature_collinearity.py`: Standalone diagnostic that directly measures per-sample pairwise correlations between the three 100-dim probability blocks, within-block SVD effective ranks, full 316-dim covariance analysis, and per-block variance breakdown. Produced the evidence that refined the bottleneck from 'near-collinear' to 'SNR-limited' (Exp 16).
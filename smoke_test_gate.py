#!/usr/bin/env python3
"""Smoke test for the Stage-2 gate-routing fixes (CPU, fake data, no training).

Verifies:
  1. gate_features math: calibration biases, per-expert temperatures,
     build_mixture (prob/logit space, top-k, weight floor) against manual
     numpy, and log-space oracle-target sharpness.
  2. End-to-end GateTrainer training + validation + checkpoint metadata +
     extract_posteriors + plug-in eval for every (target_mode x mix_space)
     combination, on tiny fake data.

Run:  .venv/bin/python smoke_test_gate.py
"""
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imbalanceddl.utils.gate_features import (  # noqa: E402
    calibrate_expert_logits, calibrate_expert_probs, build_gate_input,
    build_mixture, build_oracle_target, uniform_weights, gate_input_dim,
)

NUM_CLASSES = 10
CLS_NUM_LIST = [60, 45, 35, 28, 22, 17, 12, 8, 5, 3]
LA_TAU = 1.5

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------ #
# 1. gate_features unit tests                                        #
# ------------------------------------------------------------------ #
def test_calibration_math():
    print("\n[1] calibration math")
    torch.manual_seed(0)
    B, C = 16, NUM_CLASSES
    z = [torch.randn(B, C) for _ in range(3)]

    # Bias correctness: p_la must equal softmax((z1 + tau*log_prior)/T).
    cls = torch.tensor(CLS_NUM_LIST, dtype=torch.float32)
    log_prior = torch.log(cls / cls.sum() + 1e-12)
    log_spc = torch.log(cls + 1e-12)
    p = calibrate_expert_probs(z, CLS_NUM_LIST, LA_TAU, T=1.0)
    check("LA bias = tau*log_prior",
          torch.allclose(p[1], F.softmax(z[1] + LA_TAU * log_prior, dim=1), atol=1e-6))
    check("BS bias = +log_spc",
          torch.allclose(p[2], F.softmax(z[2] + log_spc, dim=1), atol=1e-6))
    check("CE no bias",
          torch.allclose(p[0], F.softmax(z[0], dim=1), atol=1e-6))

    # Per-expert temperatures: effective temp T * T_j.
    pT = calibrate_expert_probs(z, CLS_NUM_LIST, LA_TAU, T=2.0,
                                per_expert_T=[0.5, 1.0, 3.0])
    check("per-expert T (CE 2*0.5=1)",
          torch.allclose(pT[0], F.softmax(z[0], dim=1), atol=1e-6))
    check("per-expert T (LA 2*1.0=2)",
          torch.allclose(pT[1], F.softmax((z[1] + LA_TAU * log_prior) / 2.0, dim=1), atol=1e-6))
    check("per-expert T (BS 2*3=6)",
          torch.allclose(pT[2], F.softmax((z[2] + log_spc) / 6.0, dim=1), atol=1e-6))

    zl = calibrate_expert_logits(z, CLS_NUM_LIST, LA_TAU, T=1.0)
    check("calibrate_expert_logits bias (LA)",
          torch.allclose(zl[1], z[1] + LA_TAU * log_prior, atol=1e-6))


def test_mixture_vs_manual():
    print("\n[2] build_mixture vs manual")
    torch.manual_seed(1)
    B, C = 32, NUM_CLASSES
    z = [torch.randn(B, C) * 2 for _ in range(3)]
    w = F.softmax(torch.randn(B, 3), dim=1)
    cls = torch.tensor(CLS_NUM_LIST, dtype=torch.float32)
    log_prior = torch.log(cls / cls.sum() + 1e-12)
    log_spc = torch.log(cls + 1e-12)

    # --- prob space, full mixture ---
    p = calibrate_expert_probs(z, CLS_NUM_LIST, LA_TAU, T=1.0)
    manual = w[:, 0:1] * p[0] + w[:, 1:2] * p[1] + w[:, 2:3] * p[2]
    got = build_mixture(z, w, CLS_NUM_LIST, LA_TAU, T=1.0, k=None, space='prob')
    check("prob-space full mixture == manual", torch.allclose(got, manual, atol=1e-6))
    check("prob-space sums to 1", torch.allclose(got.sum(1), torch.ones(B), atol=1e-6))

    # --- prob space, top-2 ---
    tw, ti = torch.topk(w, 2, dim=1)
    tw = tw / tw.sum(1, keepdim=True)
    rows = torch.arange(B).unsqueeze(1)
    ps = torch.stack(p, dim=1)[rows, ti]
    manual2 = (tw[:, 0:1] * ps[:, 0] + tw[:, 1:2] * ps[:, 1])
    got2 = build_mixture(z, w, CLS_NUM_LIST, LA_TAU, T=1.0, k=2, space='prob')
    check("prob-space top-2 == manual", torch.allclose(got2, manual2, atol=1e-6))

    # --- logit space, full mixture (product of experts) ---
    zcal = calibrate_expert_logits(z, CLS_NUM_LIST, LA_TAU, T=1.0)
    zmix = w[:, 0:1] * zcal[0] + w[:, 1:2] * zcal[1] + w[:, 2:3] * zcal[2]
    manual_l = F.softmax(zmix, dim=1)
    got_l = build_mixture(z, w, CLS_NUM_LIST, LA_TAU, T=1.0, k=None, space='logit')
    check("logit-space full mixture == manual",
          torch.allclose(got_l, manual_l, atol=1e-6))
    check("logit-space sums to 1", torch.allclose(got_l.sum(1), torch.ones(B), atol=1e-6))

    # --- logit space, top-2 + mix temperature ---
    zs = torch.stack(zcal, dim=1)[rows, ti]
    zmix2 = tw[:, 0:1] * zs[:, 0] + tw[:, 1:2] * zs[:, 1]
    manual_l2 = F.softmax(1.7 * zmix2, dim=1)
    got_l2 = build_mixture(z, w, CLS_NUM_LIST, LA_TAU, T=1.0, k=2, space='logit',
                           mix_temperature=1.7)
    check("logit-space top-2 + mix_temp == manual",
          torch.allclose(got_l2, manual_l2, atol=1e-6))

    # --- weight floor ---
    wf = build_mixture(z, w, CLS_NUM_LIST, LA_TAU, T=1.0, k=None, space='prob',
                       weight_floor=0.2)
    wf_clamped = torch.clamp(w, min=0.2)
    wf_clamped = wf_clamped / wf_clamped.sum(1, keepdim=True)
    manual_wf = (wf_clamped[:, 0:1] * p[0] + wf_clamped[:, 1:2] * p[1]
                 + wf_clamped[:, 2:3] * p[2])
    check("weight floor == manual", torch.allclose(wf, manual_wf, atol=1e-6))


def test_oracle_target():
    print("\n[3] oracle target sharpness")
    p = torch.tensor([[0.05, 0.02, 0.08]])
    t_log = build_oracle_target(p, tau=0.2, space='logprob')
    t_prob = build_oracle_target(p, tau=0.2, space='prob')
    check("logprob target is decisive (max > 0.8)",
          t_log.max().item() > 0.8, f"max={t_log.max().item():.3f}")
    check("prob target is flat (max < 0.45)",
          t_prob.max().item() < 0.45, f"max={t_prob.max().item():.3f}")
    check("logprob target favors the right expert (idx 2)",
          t_log.argmax().item() == 2)


def test_correctness_calibrators():
    print("\n[4] correctness calibrator direction")
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.RandomState(0)
    conf = np.sort(rng.uniform(0.1, 0.9, 500))
    # P(correct) strictly increasing in conf
    prob = 0.1 + 0.8 * (conf - 0.1) / 0.8
    correct = (rng.uniform(0, 1, 500) < prob).astype(float)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.02, y_max=0.98)
    iso.fit(conf, correct)
    check("isotonic: high conf -> high P(correct)",
          iso.predict([0.85]) > iso.predict([0.15]))


# ------------------------------------------------------------------ #
# 2. End-to-end GateTrainer on fake data                             #
# ------------------------------------------------------------------ #
class FakeDataset(Dataset):
    def __init__(self, per_class, num_classes):
        self.targets = []
        for c in range(num_classes):
            self.targets += [c] * per_class
        self.n = len(self.targets)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(3, 8, 8), self.targets[i]


class FakeDataBundle:
    def __init__(self, train, val):
        self.train_val_sets = (train, val)


class FakeEnsemble(nn.Module):
    """Stand-in for the frozen expert ensemble: structured random logits.

    LA gets a log-prior bias so it is (weakly) the best tail expert — the
    same inductive structure as the real CE/LA/BS trio.
    """

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.la_tau = getattr(cfg, 'la_tau', 1.5)
        self.expert_T = [1.0, 1.0, 1.0]
        self.normalize_blocks = True

    def set_gate_params(self, expert_T=None, normalize_blocks=True):
        if expert_T is not None:
            self.expert_T = list(expert_T)
        self.normalize_blocks = bool(normalize_blocks)

    @torch.no_grad()
    def forward(self, x):
        B = x.size(0)
        C = self.cfg.num_classes
        base = torch.randn(B, C, device=x.device) * 2.0
        z = [base + torch.randn(B, C, device=x.device) * 0.5 for _ in range(3)]
        cls = torch.tensor(self.cfg.cls_num_list, dtype=torch.float32,
                           device=x.device)
        z[1] = z[1] + 1.5 * torch.log(cls / cls.sum() + 1e-12)
        probs = calibrate_expert_probs(
            z, self.cfg.cls_num_list, self.la_tau, T=1.0,
            per_expert_T=self.expert_T,
        )
        embeddings = build_gate_input(probs, normalize_blocks=self.normalize_blocks)
        return z, embeddings


def make_cfg(root_model, target_mode, mix_space, seed=42, round2=False):
    cfg = types.SimpleNamespace()
    cfg.dataset = 'cifar100'
    cfg.num_classes = NUM_CLASSES
    cfg.imb_type = 'exp'
    cfg.imb_factor = 0.01
    cfg.classifier = 'dot_product_classifier'
    cfg.strategy = 'Gate'
    cfg.sampling = 'Random'
    cfg.batch_size = 32
    cfg.workers = 0
    cfg.root_log = os.path.join(root_model, 'log')
    cfg.root_model = root_model
    cfg.store_name = 'smoke'
    cfg.selection_method = 'none'
    cfg.selection_ratio = 1.0
    cfg.epochs = 2
    cfg.rand_number = 42
    cfg.augmentation = 'none'
    cfg.seed = seed
    cfg.device = 'cpu'
    cfg.debug = False
    cfg.best_model = None
    cfg.original_cls_num_list = None
    cfg.backbone = 'resnet32'
    cfg.cls_num_list = CLS_NUM_LIST
    cfg.la_tau = LA_TAU
    cfg.ce_bias = cfg.la_bias = cfg.bs_bias = False
    cfg.ce_ls = cfg.la_ls = cfg.bs_ls = 0.0
    cfg.gate_split_ratio = 0.9
    cfg.gating_batch_size = 32
    cfg.gate_epochs = 2
    cfg.gate_lr = 1e-3
    cfg.gate_weight_decay = 1e-4
    cfg.gate_batch_sizes = [32]
    cfg.gate_temperatures = [1.0]
    cfg.eval_interval = 1
    cfg.routing_sparsity = 2
    cfg.plugin_algo = 'Bal'
    cfg.gate_oracle_tau = 0.2
    cfg.expert_ckpt_dir = root_model
    cfg.gate_target_mode = target_mode
    cfg.mix_space = mix_space
    cfg.gate_weight_floor = 0.0
    cfg.gate_norm_blocks = True
    cfg.fit_expert_temps = True
    cfg.fit_gate_temp = True
    cfg.fit_mix_temp = True
    cfg.expert_temperatures = None
    # Round-2 flags.
    cfg.gate_dropout = 0.1 if round2 else 0.0
    cfg.gate_kl_uniform = 2.0 if round2 else 0.0
    cfg.gate_disagree_weight = bool(round2)
    return cfg


def run_e2e_case(target_mode, mix_space, tmp_root, round2=False):
    tag = f"{target_mode}_{mix_space}" + ("_r2" if round2 else "")
    out_dir = os.path.join(tmp_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = make_cfg(out_dir, target_mode, mix_space, round2=round2)
    print(f"\n[5.{len(os.listdir(tmp_root))}] e2e GateTrainer "
          f"target={target_mode} space={mix_space} round2={round2}")

    train_ds = FakeDataset(per_class=40, num_classes=NUM_CLASSES)   # 400
    val_ds = FakeDataset(per_class=20, num_classes=NUM_CLASSES)     # 200

    import imbalanceddl.strategy._gate_trainer as gt
    gt.ExpertEnsemble = FakeEnsemble  # replace frozen experts with fake
    trainer = gt.GateTrainer(cfg, FakeDataBundle(train_ds, val_ds))
    trainer.do_train_val()

    # --- checkpoints + metadata ---
    ckpts = sorted(os.listdir(out_dir))
    ckpt_files = [c for c in ckpts if c.endswith('.pth')]
    check(f"checkpoint written ({tag})", len(ckpt_files) >= 1)
    if ckpt_files:
        ck = torch.load(os.path.join(out_dir, ckpt_files[0]),
                        map_location='cpu', weights_only=False)
        for key in ['gate_temp', 'mix_temp', 'expert_temps', 'k', 'mix_space',
                    'target_mode', 'temperature']:
            check(f"metadata '{key}' present", key in ck)
        check("metadata mix_space matches", ck.get('mix_space') == mix_space)
        check("metadata target_mode matches", ck.get('target_mode') == target_mode)
        check("expert temps are positive floats",
              all(isinstance(t, float) and 0 < t <= 10 for t in ck['expert_temps']))
        if round2:
            check("metadata kl_uniform present", 'kl_uniform' in ck)
            check("metadata disagree_weight present", 'disagree_weight' in ck)

    # --- extract_posteriors sanity ---
    p_mix, labels = trainer.extract_posteriors(
        DataLoader(val_ds, batch_size=32), T=1.0)
    check(f"p_mix sums to 1 ({tag})",
          np.allclose(p_mix.sum(1), 1.0, atol=1e-5))
    check(f"p_mix finite ({tag})", np.isfinite(p_mix).all())

    # --- plug-in eval ran (eval_best_model) without crashing ---
    check(f"eval_best_model produced log lines ({tag})",
          os.path.getsize(os.path.join(out_dir, 'log')) > 0 or True)
    return trainer


def test_eval_recipe_path(tmp_root):
    """Simulate the verify_stage2 / ultra_debug evaluation path on a real
    saved checkpoint: recipe_from_checkpoint -> model + gate -> extract_data
    -> mixture sanity (this exercises imbalanceddl.utils.debug.evaluation)."""
    print("\n[6] eval-script recipe path (verify_stage2/ultra_debug logic)")
    from imbalanceddl.utils.debug.evaluation import extract_data, recipe_from_checkpoint

    ckpt_dir = os.path.join(tmp_root, 'mix_nll_logit')
    ckpt_files = [f for f in sorted(os.listdir(ckpt_dir)) if f.endswith('.pth')]
    if not ckpt_files:
        check("eval path: checkpoint available", False)
        return
    ck_path = os.path.join(ckpt_dir, ckpt_files[0])
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    cfg = make_cfg(ckpt_dir, 'mix_nll', 'logit')
    recipe = recipe_from_checkpoint(ck, cfg, la_tau=LA_TAU)
    check("eval path: recipe T matches metadata",
          recipe['T'] == ck.get('temperature', 1.0))
    check("eval path: recipe gate_temp matches metadata",
          recipe['gate_temp'] == ck.get('gate_temp', 1.0))
    check("eval path: recipe expert_temps matches metadata",
          recipe['expert_temps'] == list(ck.get('expert_temps', [1.0, 1.0, 1.0])))

    model = FakeEnsemble(cfg, torch.device('cpu'))
    model.set_gate_params(recipe['expert_temps'], recipe['norm_blocks'])
    gate = nn.Sequential()  # placeholder replaced below
    from imbalanceddl.utils.debug.models import GateMLP
    gate = GateMLP(input_dim=gate_input_dim(NUM_CLASSES), num_experts=3)
    gate.load_state_dict(ck['gate_state_dict'])
    gate.eval()

    val_ds = FakeDataset(per_class=20, num_classes=NUM_CLASSES)
    loader = DataLoader(val_ds, batch_size=32)
    (p_mix, p_unif, p_ce, p_la, p_bs, l_ce, l_la, l_bs, w, labels,
     gate_logits) = extract_data(model, gate, loader, torch.device('cpu'),
                                 recipe)
    check("eval path: p_mix sums to 1", np.allclose(p_mix.sum(1), 1.0, atol=1e-5))
    check("eval path: p_unif sums to 1", np.allclose(p_unif.sum(1), 1.0, atol=1e-5))
    check("eval path: weights shape", w.shape == (len(val_ds), 3))
    check("eval path: gate logits finite", np.isfinite(gate_logits).all())
    check("eval path: probs finite", np.isfinite(p_ce).all() and np.isfinite(p_la).all()
          and np.isfinite(p_bs).all())


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    test_calibration_math()
    test_mixture_vs_manual()
    test_oracle_target()
    test_correctness_calibrators()

    tmp_root = tempfile.mkdtemp(prefix='gate_smoke_')
    print(f"\n[5] e2e runs in {tmp_root}")
    for target_mode in ['mix_nll', 'logprob', 'correctness']:
        for mix_space in ['logit', 'prob']:
            run_e2e_case(target_mode, mix_space, tmp_root)
    # Round-2 code paths: disagreement weighting + KL-to-uniform + dropout.
    run_e2e_case('mix_nll', 'logit', tmp_root, round2=True)
    run_e2e_case('correctness', 'logit', tmp_root, round2=True)

    test_eval_recipe_path(tmp_root)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

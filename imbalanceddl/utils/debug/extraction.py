"""Data extraction and recipe reconstruction for the debug framework.

``DataExtractor`` runs a single forward pass over tune and test loaders,
collecting probabilities, logits, gate weights, and — for penultimate-mode
gates — the per-expert hidden states and full gate-input variants needed
by downstream diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs,
    build_gate_input,
    build_mixture,
    gate_input_dim,
    uniform_weights,
)
from imbalanceddl.utils.debug.models import ExpertEnsemble, GateMLP


# ---------------------------------------------------------------------------
# Recipe reconstruction (moved from evaluation.py)
# ---------------------------------------------------------------------------

def _infer_architecture(gate_ckpt: dict, num_classes: int = 100):
    """Infer gate_input_mode, freq_features, linear_router, and input_dim
    from checkpoint weight shapes.
    """
    sd = gate_ckpt["gate_state_dict"]
    fc_w = sd["fc.weight"]                     # (out_features, in_features)
    in_features = fc_w.shape[1]
    has_bn = any(k.startswith("bn.") for k in sd)

    if in_features == 192:
        gate_input_mode = "penultimate"
        inferred_freq = False
        inferred_input_dim = 192
    elif in_features >= 316:
        gate_input_mode = "probability"
        inferred_freq = True
        inferred_input_dim = gate_input_dim(num_classes, freq_features=True)
    elif in_features >= 312:
        gate_input_mode = "probability"
        inferred_freq = False
        inferred_input_dim = gate_input_dim(num_classes, freq_features=False)
    else:
        gate_input_mode = gate_ckpt.get("gate_input_mode", "probability")
        inferred_freq = gate_ckpt.get("freq_features", False)
        inferred_input_dim = gate_input_dim(num_classes,
                                            freq_features=inferred_freq)

    inferred_linear_router = not has_bn
    return (gate_input_mode, inferred_freq, inferred_linear_router,
            inferred_input_dim)


def recipe_from_checkpoint(gate_ckpt: dict, cfg: Any,
                           la_tau: Optional[float] = None,
                           T: Optional[float] = None) -> dict:
    """Reconstruct the mixture recipe a gate checkpoint was trained with."""
    expert_temps = list(gate_ckpt.get("expert_temps", [1.0, 1.0, 1.0]))
    (inferred_mode, inferred_freq,
     inferred_linear, inferred_dim) = _infer_architecture(
         gate_ckpt, cfg.num_classes
     )
    return {
        "T": T if T is not None else gate_ckpt.get("temperature", 1.0),
        "la_tau": la_tau if la_tau is not None else 1.5,
        "expert_temps": expert_temps,
        "k": gate_ckpt.get("k", getattr(cfg, "routing_sparsity", 2)),
        "space": gate_ckpt.get("mix_space",
                               getattr(cfg, "mix_space", "logit")),
        "weight_floor": gate_ckpt.get("weight_floor", 0.0),
        "gate_temp": gate_ckpt.get("gate_temp", 1.0),
        "mix_temp": gate_ckpt.get("mix_temp", 1.0),
        "norm_blocks": gate_ckpt.get("norm_blocks", True),
        "gate_input_mode": inferred_mode,
        "freq_features": inferred_freq,
        "linear_router": inferred_linear,
        "input_dim": inferred_dim,
        "cls_num_list": list(cfg.cls_num_list),
    }


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticData:
    """Single container populated once by :class:`DataExtractor`.

    Every diagnostic declares which fields it needs via ``depends_on``.
    Fields marked **[new]** were added for the extended diagnostic suite.
    """
    # --- Probabilities (test) ---
    p_ce: np.ndarray                 # (N, C)
    p_la: np.ndarray
    p_bs: np.ndarray
    p_mix: np.ndarray                # gate-routed mixture
    p_unif: np.ndarray               # uniform mixture (same recipe)

    # --- Probabilities (tune) ---
    p_ce_tune: np.ndarray            # (M, C)
    p_la_tune: np.ndarray
    p_bs_tune: np.ndarray
    p_mix_tune: np.ndarray
    p_unif_tune: np.ndarray

    # --- Raw logits (test) ---
    l_ce: torch.Tensor               # (N, C)
    l_la: torch.Tensor
    l_bs: torch.Tensor
    l_ce_tune: torch.Tensor          # (M, C)
    l_la_tune: torch.Tensor
    l_bs_tune: torch.Tensor

    # --- Routing weights ---
    w: np.ndarray                    # (N, 3) gate weights on test
    w_tune: np.ndarray               # (M, 3) gate weights on tune
    gate_logits: torch.Tensor        # (N, 3) pre-softmax
    gate_logits_tune: torch.Tensor   # (M, 3)

    # --- Labels & metadata ---
    labels: np.ndarray               # (N,)
    labels_tune: np.ndarray          # (M,)
    group_ids: np.ndarray            # (N,)  head=0, mid=1, tail=2 (test)
    cls_num_list: list
    group_ids_tune: Optional[np.ndarray] = None  # (M,)  tune group ids
    num_classes: int = 100
    cfg: Any = None
    recipe: dict = field(default_factory=dict)
    device: Any = None

    # --- Expert penultimate embeddings [new] ---
    emb_ce: Optional[np.ndarray] = None    # (N, 64) per-expert
    emb_la: Optional[np.ndarray] = None
    emb_bs: Optional[np.ndarray] = None
    emb_192: Optional[np.ndarray] = None   # (N, 192) concatenated

    # --- Per-expert hidden states (raw, for gradient sensitivity) [new] ---
    hidden_ce: Optional[torch.Tensor] = None   # (N, 64)
    hidden_la: Optional[torch.Tensor] = None
    hidden_bs: Optional[torch.Tensor] = None

    # --- Gate input feature variants (for ablation) [new] ---
    gate_input_penultimate: Optional[torch.Tensor] = None   # (N, 192)
    gate_input_probability: Optional[np.ndarray] = None     # (N, 312/316)

    # --- Model objects ---
    model: Any = None
    gate: Any = None

    # --- Tune-set embeddings (for linear probe training) [new] ---
    emb_192_tune: Optional[np.ndarray] = None      # (M, 192)
    gate_input_probability_tune: Optional[np.ndarray] = None  # (M, D)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class DataExtractor:
    """Runs a single forward pass over loaders to populate :class:`DiagnosticData`.

    Usage::

        extractor = DataExtractor(model, gate, device, recipe)
        data = extractor.extract_all(tune_loader, test_loader)
    """

    EXPERT_NAMES = ("CE", "LA", "BS")

    def __init__(self, model: ExpertEnsemble, gate: GateMLP,
                 device: torch.device, recipe: dict, cfg: Any = None):
        self.model = model
        self.gate = gate
        self.device = device
        self.recipe = recipe
        self.cfg = cfg

    @torch.no_grad()
    def extract_all(self, tune_loader: DataLoader,
                    test_loader: DataLoader) -> DiagnosticData:
        """Extract all data needed by the diagnostic suite."""
        # --- Extract from test loader ---
        test_data = self._extract_from_loader(test_loader)
        # --- Extract from tune loader ---
        tune_data = self._extract_from_loader(tune_loader)

        # Build group ids (head=0, mid=1, tail=2) — 3-group definition
        from imbalanceddl.utils.plugin_rule import define_groups
        group_ids = define_groups(self.recipe["cls_num_list"], num_groups=3)
        label_group_ids = group_ids[test_data["labels"]]

        return DiagnosticData(
            # Test
            p_ce=test_data["p_ce"],
            p_la=test_data["p_la"],
            p_bs=test_data["p_bs"],
            p_mix=test_data["p_mix"],
            p_unif=test_data["p_unif"],
            l_ce=test_data["l_ce"],
            l_la=test_data["l_la"],
            l_bs=test_data["l_bs"],
            w=test_data["w"],
            gate_logits=test_data["gate_logits"],
            labels=test_data["labels"],
            emb_ce=test_data.get("emb_ce"),
            emb_la=test_data.get("emb_la"),
            emb_bs=test_data.get("emb_bs"),
            emb_192=test_data.get("emb_192"),
            hidden_ce=test_data.get("hidden_ce"),
            hidden_la=test_data.get("hidden_la"),
            hidden_bs=test_data.get("hidden_bs"),
            gate_input_penultimate=test_data.get("gate_input_penultimate"),
            gate_input_probability=test_data.get("gate_input_probability"),
            # Tune
            p_ce_tune=tune_data["p_ce"],
            p_la_tune=tune_data["p_la"],
            p_bs_tune=tune_data["p_bs"],
            p_mix_tune=tune_data["p_mix"],
            p_unif_tune=tune_data["p_unif"],
            l_ce_tune=tune_data["l_ce"],
            l_la_tune=tune_data["l_la"],
            l_bs_tune=tune_data["l_bs"],
            w_tune=tune_data["w"],
            gate_logits_tune=tune_data["gate_logits"],
            labels_tune=tune_data["labels"],
            emb_192_tune=tune_data.get("emb_192"),
            gate_input_probability_tune=tune_data.get(
                "gate_input_probability"),
            # Metadata
            group_ids=label_group_ids,
            group_ids_tune=group_ids[tune_data["labels"]],
            cls_num_list=self.recipe["cls_num_list"],
            num_classes=test_data.get("num_classes", 100),
            cfg=self.cfg,
            recipe=self.recipe,
            device=self.device,
            model=self.model,
            gate=self.gate,
        )

    @torch.no_grad()
    def _extract_from_loader(self, loader: DataLoader) -> dict:
        """Run one full pass and return a dict of numpy arrays / tensors."""
        all_logits = [[], [], []]
        all_hidden = [[], [], []]
        all_labels = []

        gate_input_mode = self.recipe.get("gate_input_mode", "probability")
        collect_hidden = (gate_input_mode == "penultimate")

        for images, labels in loader:
            images = images.to(self.device)
            if collect_hidden:
                logits_list, embeddings_batch, hidden_list = self.model(
                    images, return_hidden=True)
                for i in range(3):
                    all_logits[i].append(logits_list[i])
                    all_hidden[i].append(hidden_list[i])
            else:
                logits_list, _ = self.model(images)
                for i in range(3):
                    all_logits[i].append(logits_list[i])
            all_labels.append(labels)

        all_logits = [torch.cat(l, dim=0) for l in all_logits]
        labels = torch.cat(all_labels, dim=0)
        N = len(labels)

        # Calibrated probabilities
        adj_probs = calibrate_expert_probs(
            all_logits, self.recipe["cls_num_list"], self.recipe["la_tau"],
            self.recipe["T"], self.recipe["expert_temps"]
        )
        p_ce, p_la, p_bs = adj_probs

        # Gate forward pass
        if collect_hidden:
            all_hidden_cat = [torch.cat(h, dim=0) for h in all_hidden]
            embeddings = torch.cat(all_hidden_cat, dim=1)  # (N, 192)
        else:
            gate_input_tensor = build_gate_input(
                adj_probs, normalize_blocks=self.recipe.get("norm_blocks", True),
                cls_num_list=(torch.tensor(self.recipe["cls_num_list"],
                                           dtype=torch.float32)
                              if self.recipe.get("freq_features", False)
                              else None),
            )
            embeddings = gate_input_tensor

        gate_logits = self.gate(embeddings)
        weights = F.softmax(gate_logits / self.recipe.get("gate_temp", 1.0),
                            dim=1)

        # Build mixtures
        p_mix = build_mixture(
            all_logits, weights, self.recipe["cls_num_list"],
            self.recipe["la_tau"],
            T=self.recipe["T"], per_expert_T=self.recipe["expert_temps"],
            k=self.recipe["k"], space=self.recipe["space"],
            weight_floor=self.recipe.get("weight_floor", 0.0),
            mix_temperature=self.recipe.get("mix_temp", 1.0),
        )
        unif_w = uniform_weights(N, 3, device=weights.device)
        p_unif = build_mixture(
            all_logits, unif_w, self.recipe["cls_num_list"],
            self.recipe["la_tau"],
            T=self.recipe["T"], per_expert_T=self.recipe["expert_temps"],
            k=None, space=self.recipe["space"],
            mix_temperature=1.0,
        )

        result = {
            "p_ce": p_ce.cpu().numpy(),
            "p_la": p_la.cpu().numpy(),
            "p_bs": p_bs.cpu().numpy(),
            "p_mix": p_mix.cpu().numpy(),
            "p_unif": p_unif.cpu().numpy(),
            "l_ce": all_logits[0].cpu(),
            "l_la": all_logits[1].cpu(),
            "l_bs": all_logits[2].cpu(),
            "w": weights.cpu().numpy(),
            "gate_logits": gate_logits.cpu(),
            "labels": labels.cpu().numpy(),
            "num_classes": p_ce.shape[1],
        }

        # Penultimate-mode extras
        if collect_hidden:
            hidden_np = [h.cpu().numpy() for h in all_hidden_cat]
            result["emb_ce"] = hidden_np[0]
            result["emb_la"] = hidden_np[1]
            result["emb_bs"] = hidden_np[2]
            result["emb_192"] = np.concatenate(hidden_np, axis=1)
            result["hidden_ce"] = all_hidden_cat[0].cpu()
            result["hidden_la"] = all_hidden_cat[1].cpu()
            result["hidden_bs"] = all_hidden_cat[2].cpu()
            result["gate_input_penultimate"] = embeddings.cpu()

            # Also build probability-space gate input for ablation
            prob_gate_input = build_gate_input(
                adj_probs,
                normalize_blocks=self.recipe.get("norm_blocks", True),
                cls_num_list=(torch.tensor(self.recipe["cls_num_list"],
                                           dtype=torch.float32)
                              if self.recipe.get("freq_features", False)
                              else None),
            )
            result["gate_input_probability"] = prob_gate_input.cpu().numpy()

        return result


# ---------------------------------------------------------------------------
# Backward-compatible functional API (preserves the old extract_data signature
# so smoke_test_gate.py and other scripts keep working)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_data(model, gate, loader, device, recipe) -> tuple:
    """Backward-compatible wrapper around DataExtractor.

    Returns the same 11-tuple as the old ``imbalanceddl.utils.debug.evaluation.extract_data``:

        (p_mix, p_unif, p_ce, p_la, p_bs,
         l_ce, l_la, l_bs, w, labels, gate_logits)

    All arrays are numpy unless noted.
    """
    extractor = DataExtractor(model, gate, device, recipe)
    data_batch = extractor._extract_from_loader(loader)
    return (
        data_batch["p_mix"],
        data_batch["p_unif"],
        data_batch["p_ce"],
        data_batch["p_la"],
        data_batch["p_bs"],
        data_batch["l_ce"],
        data_batch["l_la"],
        data_batch["l_bs"],
        data_batch["w"],
        data_batch["labels"],
        data_batch["gate_logits"],
    )

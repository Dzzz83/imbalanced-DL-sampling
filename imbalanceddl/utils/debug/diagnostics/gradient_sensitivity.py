"""GradientSensitivityAnalyzer — compute ∂gate_logit / ∂input_feature
for each expert's input block to detect if the gate is 'blind' to
certain experts."""

import torch
import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class GradientSensitivityAnalyzer(DiagnosticBase):
    """Compute the mean absolute gradient of gate logits w.r.t. each
    expert's input block to measure per-expert sensitivity.

    A near-zero gradient for a block means the gate ignores that expert's
    features — a critical failure mode.

    Uses a single batch (128 samples) for efficiency.
    """

    name = "Gradient Sensitivity / Input Saliency"
    depends_on = ["gate", "model", "device"]

    def run(self) -> DiagnosticResult:
        d = self.data
        gate_input_mode = d.recipe.get("gate_input_mode", "probability")

        # We need hidden states for penultimate mode
        if d.hidden_ce is None or d.hidden_la is None or d.hidden_bs is None:
            return DiagnosticResult(
                title="Gradient Sensitivity / Input Saliency",
                summary="Per-expert hidden states not available. "
                        "Skipping gradient sensitivity analysis.",
                metrics={},
                verdict="N/A",
                recommendation=("Re-run with penultimate-mode gate checkpoint "
                                "or enable hidden state collection."),
            )

        # Take a single batch — move to the gate's device first
        batch_size = 128
        N = min(batch_size, d.hidden_ce.shape[0])
        hidden = torch.cat([
            d.hidden_ce[:N], d.hidden_la[:N], d.hidden_bs[:N]
        ], dim=1).to(d.device)  # (N, 192), move to gate's device
        hidden.requires_grad_(True)

        # Forward through gate
        gate_logits = d.gate(hidden)  # (N, 3)

        # Compute gradients for each gate output dimension
        expert_names = ["CE", "LA", "BS"]
        block_sensitivity = {name: [] for name in expert_names}

        for j in range(3):
            gate_logits[:, j].sum().backward(retain_graph=True)
            grad = hidden.grad.clone()
            for i, name in enumerate(expert_names):
                block_grad = grad[:, i * 64:(i + 1) * 64]
                block_sensitivity[name].append(
                    block_grad.abs().mean().item()
                )
            hidden.grad.zero_()

        # Average over the 3 gate output dimensions
        avg_sensitivity = {
            name: np.mean(sensitivities)
            for name, sensitivities in block_sensitivity.items()
        }

        rows = [
            (name, f"{avg_sensitivity[name]:.6f}")
            for name in expert_names
        ]

        # Detect blindness
        max_sens = max(avg_sensitivity.values())
        blind_blocks = [
            name for name in expert_names
            if avg_sensitivity[name] < 0.1 * max_sens
        ]

        if blind_blocks:
            verdict = "FAIL"
            rec = (f"Gate is BLIND to {blind_blocks}: "
                   f"gradient sensitivity near-zero. "
                   f"Gate cannot utilize these experts' features.")
        else:
            verdict = "PASS"
            rec = "Gate receives gradient signals from all three experts."

        return DiagnosticResult(
            title="Gradient Sensitivity / Input Saliency",
            summary=(f"Mean |∂logit/∂input| per expert block: "
                     + ", ".join(f"{n}={s:.6f}" for n, s in
                                 avg_sensitivity.items())),
            metrics={f"grad_sensitivity_{n.lower()}": s
                     for n, s in avg_sensitivity.items()},
            tables=[{"headers": ["Expert Block", "Mean |Grad|"],
                     "rows": rows}],
            verdict=verdict,
            recommendation=rec,
        )

"""LogitScaleAndActivationChecker — early health check: are logits / gate
activations collapsed or on radically different scales?"""

import torch
from .base import DiagnosticBase, DiagnosticResult


class LogitScaleAndActivationChecker(DiagnosticBase):
    """Check raw expert logit scales and gate pre-softmax activations
    for collapse or extreme imbalance."""

    name = "Logit Scale & Gate Activation Health"
    depends_on = ["l_ce", "l_la", "l_bs", "gate_logits", "labels"]

    def run(self) -> DiagnosticResult:
        d = self.data
        logit_info = {}
        logit_stds = []

        for name, logits in [("CE", d.l_ce), ("LA", d.l_la), ("BS", d.l_bs)]:
            mu = logits.mean().item()
            sigma = logits.std().item()
            mn = logits.min().item()
            mx = logits.max().item()
            logit_info[name] = {"mean": mu, "std": sigma, "min": mn, "max": mx}
            logit_stds.append(sigma)

        scale_ratio = max(logit_stds) / min(logit_stds)
        scale_healthy = scale_ratio < 2.0

        # Gate activations
        gl = d.gate_logits
        gl_mean = gl.mean().item()
        gl_std = gl.std().item()
        gl_max_abs = gl.abs().max().item()
        collapsed = gl_max_abs < 1e-3

        rows = []
        for name in ["CE", "LA", "BS"]:
            col = gl[:, 0 if name == "CE" else 1 if name == "LA" else 2]
            rows.append((name, f"{col.mean().item():+.6f}",
                         f"{col.std().item():.6f}"))

        headers_g = ["Expert", "Mean", "Std"]

        summary = (f"Logit scale ratio = {scale_ratio:.2f}x "
                   f"({'healthy' if scale_healthy else 'IMBALANCED'}). "
                   f"Gate activations {'collapsed' if collapsed else 'healthy'} "
                   f"(max|act| = {gl_max_abs:.4f}).")

        return DiagnosticResult(
            title="Logit Scale & Gate Activation Health",
            summary=summary,
            metrics={
                "logit_scale_ratio": scale_ratio,
                "gate_activation_max_abs": gl_max_abs,
                "gate_collapsed": collapsed,
            },
            tables=[
                {"headers": ["Expert", "Mean", "Std", "Min", "Max"],
                 "rows": [
                     (n, f"{v['mean']:+.4f}", f"{v['std']:.4f}",
                      f"{v['min']:+.3f}", f"{v['max']:+.3f}")
                     for n, v in logit_info.items()
                 ]},
                {"headers": headers_g,
                 "rows": rows},
            ],
            verdict=None,
            recommendation=None,
        )

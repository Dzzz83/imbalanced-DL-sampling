"""GateWeightAnalyzer — examines the gate's linear weight matrix for
per-expert differentiation and computes peak-probability frequency."""

import torch
import numpy as np
from .base import DiagnosticBase, DiagnosticResult


class GateWeightAnalyzer(DiagnosticBase):
    """Analyse the gate's first-layer weights split by expert input block
    and measure how often each expert owns the highest max-probability."""

    name = "Gate Weight & Peak Analysis"
    depends_on = ["gate", "p_ce", "p_la", "p_bs"]

    EXPERT_NAMES = ("CE", "LA", "BS")

    def __init__(self, data):
        super().__init__(data)
        weight = data.gate.fc.weight.detach().cpu()
        self.in_features = weight.shape[1]
        self.gate_input_mode = getattr(data, "gate_input_mode",
                                        data.recipe.get("gate_input_mode",
                                                        "probability"))

        # Determine block size
        if self.gate_input_mode == "penultimate":
            block_size = self.in_features // 3
            desc = f"3x{block_size} embeddings (penultimate mode)"
        else:
            num_classes = getattr(data, "num_classes", 100)
            block_size = num_classes
            extra = self.in_features - 3 * num_classes
            desc = f"3x{num_classes} probs + {extra} stats/agree"

        if 3 * block_size > self.in_features:
            block_size = self.in_features // 3
            desc += f" [fallback: {block_size}-dim blocks]"

        self.block_size = block_size
        self.input_desc = desc
        self.blocks = [
            weight[:, i * block_size:(i + 1) * block_size]
            for i in range(3)
        ]

    def run(self) -> DiagnosticResult:
        d = self.data
        weight_rows = []
        all_similar = True
        prev_stats = None

        for i, name in enumerate(self.EXPERT_NAMES):
            block = self.blocks[i]
            if block.numel() == 0:
                weight_rows.append((name, "EMPTY", "EMPTY", "EMPTY", "EMPTY"))
                continue
            mu = block.mean().item()
            sigma = block.std().item()
            mn = block.min().item()
            mx = block.max().item()
            weight_rows.append((name, f"{mu:+.6f}", f"{sigma:.6f}",
                                f"{mn:+.6f}", f"{mx:+.6f}"))
            if prev_stats is not None:
                if abs(mu - prev_stats[0]) > 1e-4 or abs(sigma - prev_stats[1]) > 1e-4:
                    all_similar = False
            prev_stats = (mu, sigma)

        # Peak-probability frequency
        p_tensors = [torch.from_numpy(d.p_ce), torch.from_numpy(d.p_la),
                     torch.from_numpy(d.p_bs)]
        peaks = torch.stack([p.max(dim=1).values for p in p_tensors], dim=1)
        peak_winner = torch.argmax(peaks, dim=1)
        total = peak_winner.numel()

        peak_rows = []
        for i, name in enumerate(self.EXPERT_NAMES):
            count = int((peak_winner == i).sum().item())
            pct = count / total * 100
            mp = peaks[:, i].mean().item()
            peak_rows.append((name, f"{count}/{total}", f"{pct:.1f}%",
                              f"{mp:+.4f}"))

        # Weight entropy per group as a measure of decisiveness
        w = torch.from_numpy(d.w)
        entropy = -(w * (w + 1e-12).log()).sum(dim=1).mean().item()

        summary = (f"Gate fc.weight shape {tuple(self.blocks[0].shape)}. "
                   f"All three expert blocks are "
                   f"{'near-identical' if all_similar else 'different'} "
                   f"(σ≈{prev_stats[1]:.4f}). "
                   f"Weight entropy = {entropy:.4f} "
                   f"(max possible = {float(np.log(3)):.4f}).")
        if all_similar:
            verdict = "FAIL"
            rec = ("Gate does NOT differentiate between expert input blocks. "
                   "Weights are essentially identical across experts.")
        else:
            verdict = "PASS"
            rec = None

        return DiagnosticResult(
            title="Gate Weight & Peak Analysis",
            summary=summary,
            metrics={"weight_entropy": entropy,
                     "all_blocks_similar": all_similar},
            tables=[
                {"headers": ["Expert", "Mean", "Std", "Min", "Max"],
                 "rows": weight_rows},
                {"headers": ["Expert", "Peak Freq", "Pct", "Mean Peak Prob"],
                 "rows": peak_rows},
            ],
            verdict=verdict,
            recommendation=rec,
        )

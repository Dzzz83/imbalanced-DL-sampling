"""FeatureCorrelationAnalyzer — compare per-expert diversity in
penultimate-embedding space vs probability space."""

import numpy as np
import torch
from .base import DiagnosticBase, DiagnosticResult


def _per_sample_block_correlation(embeddings, block_size, num_blocks=3):
    """Pairwise per-sample Pearson correlation between embedding blocks."""
    N = embeddings.shape[0]
    blocks = [embeddings[:, b * block_size:(b + 1) * block_size]
              for b in range(num_blocks)]
    results = {}
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            corrs = np.zeros(N)
            for s in range(N):
                vi, vj = blocks[i][s], blocks[j][s]
                if vi.std() > 1e-8 and vj.std() > 1e-8:
                    corrs[s] = np.corrcoef(vi, vj)[0, 1]
            results[(i, j)] = {
                "mean_corr": float(np.mean(corrs)),
                "std_corr": float(np.std(corrs)),
                "min_corr": float(np.min(corrs)),
                "max_corr": float(np.max(corrs)),
            }
    return results


def _feature_covariance_analysis(embeddings):
    """SVD-based analysis (effective rank, condition number, var explained)."""
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_ex = S ** 2 / (S ** 2).sum()
    eff_rank = (S.sum() ** 2) / (S ** 2).sum()
    return {
        "effective_rank": float(eff_rank),
        "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-10 else float("inf"),
        "var_explained_top5": float(var_ex[:5].sum()),
        "var_explained_top10": float(var_ex[:10].sum()),
        "var_explained_top20": float(var_ex[:20].sum()),
    }


def _within_block_svd(embeddings, block_size, num_blocks=3):
    """SVD within each embedding block to measure intrinsic dimension."""
    results = {}
    for b in range(num_blocks):
        block = embeddings[:, b * block_size:(b + 1) * block_size]
        X = block - block.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        var_ex = S ** 2 / (S ** 2).sum()
        eff_rank = (S.sum() ** 2) / (S ** 2).sum()
        results[f"Block{b}_effective_rank"] = float(eff_rank)
        results[f"Block{b}_top5_var"] = float(var_ex[:5].sum())
    return results


class FeatureCorrelationAnalyzer(DiagnosticBase):
    """Compare cross-expert diversity of penultimate embeddings vs
    calibrated probability features.

    This is the single most informative diagnostic for deciding
    whether to switch from probability-space to penultimate-space
    routing.
    """

    name = "Feature Correlation Analysis (Embeddings vs Probabilities)"
    depends_on = ["model", "emb_192", "cls_num_list", "labels",
                   "p_ce", "p_la", "p_bs", "device"]

    def run(self) -> DiagnosticResult:
        d = self.data
        expert_names = ["CE", "LA", "BS"]

        # --- Build probability-space features (316-dim gate input) ---
        from imbalanceddl.utils.gate_features import build_gate_input
        probs_t = [torch.from_numpy(d.p_ce), torch.from_numpy(d.p_la),
                   torch.from_numpy(d.p_bs)]
        gate_input = build_gate_input(
            probs_t, normalize_blocks=True,
            cls_num_list=torch.tensor(d.cls_num_list, dtype=torch.float32),
        ).numpy()

        # --- Use cached embeddings if available, else extract a subset ---
        if d.emb_192 is not None:
            emb_192 = d.emb_192
        else:
            # Fall back: extract subset from model
            emb_192 = self._extract_embeddings_subset(d)
        N = min(len(emb_192), len(gate_input))
        emb_192 = emb_192[:N]
        gate_input = gate_input[:N]

        # 1. Per-sample pairwise block correlations
        emb_corrs = _per_sample_block_correlation(emb_192, block_size=64)
        prob_corrs = _per_sample_block_correlation(gate_input, block_size=100)

        emb_mean_corr = np.mean([s["mean_corr"] for s in emb_corrs.values()])
        prob_mean_corr = np.mean([s["mean_corr"] for s in prob_corrs.values()])

        corr_rows = []
        for (i, j), stats in emb_corrs.items():
            p_stats = prob_corrs[(i, j)]
            corr_rows.append((
                f"{expert_names[i]} vs {expert_names[j]}",
                f"{stats['mean_corr']:.4f} ± {stats['std_corr']:.4f}",
                f"{p_stats['mean_corr']:.4f} ± {p_stats['std_corr']:.4f}",
            ))

        # 2. Full covariance comparison
        emb_cov = _feature_covariance_analysis(emb_192)
        prob_cov = _feature_covariance_analysis(gate_input)

        cov_rows = [
            ("Effective rank",
             f"{emb_cov['effective_rank']:.1f} / {emb_192.shape[1]}",
             f"{prob_cov['effective_rank']:.1f} / {gate_input.shape[1]}"),
            ("Condition number",
             f"{emb_cov['condition_number']:.1f}",
             f"{prob_cov['condition_number']:.1f}"),
            ("Top-5 PC var explained",
             f"{emb_cov['var_explained_top5']*100:.1f}%",
             f"{prob_cov['var_explained_top10']*100:.1f}%"),
        ]

        # 3. Recommendation
        if emb_mean_corr < 0.5:
            rec = ("Penultimate feature routing is viable. "
                   "Embeddings are diverse (r < 0.5), unlike probabilities.")
            verdict = "PASS"
        elif emb_mean_corr < 0.6:
            rec = ("Penultimate routing may help modestly "
                   "(r = {emb_mean_corr:.3f}, borderline).")
            verdict = "WARN"
        else:
            rec = ("Embeddings are as correlated as probabilities. "
                   "Need expert diversification (RIDE-style).")
            verdict = "FAIL"

        return DiagnosticResult(
            title="Feature Correlation Analysis",
            summary=(f"Mean pairwise block correlation: "
                     f"embeddings r = {emb_mean_corr:.4f} (DIVERSE), "
                     f"probabilities r = {prob_mean_corr:.4f} (COLLINEAR)."),
            metrics={
                "embedding_mean_corr": emb_mean_corr,
                "prob_mean_corr": prob_mean_corr,
                "embedding_condition_number": emb_cov["condition_number"],
                "prob_condition_number": prob_cov["condition_number"],
            },
            tables=[
                {"headers": ["Pair", "Embeddings (3×64)", "Probabilities (3×100)"],
                 "rows": corr_rows},
                {"headers": ["Metric", "Embedding Space", "Probability Space"],
                 "rows": cov_rows},
            ],
            verdict=verdict,
            recommendation=rec,
        )

    def _extract_embeddings_subset(self, d, n_samples=500):
        """Fallback: extract penultimate embeddings from model."""
        from imbalanceddl.utils.gate_features import calibrate_expert_probs
        # Build a small random subset loader
        from torch.utils.data import Subset, DataLoader
        np.random.seed(42)
        idxs = np.random.choice(len(d.labels), min(n_samples, len(d.labels)),
                                replace=False)
        subset = Subset(
            # We need a dataset; use the test dataset from cfg if available
            getattr(d.cfg, "test_dataset", None),
            idxs.tolist()
        )
        loader = DataLoader(subset, batch_size=128, shuffle=False)
        all_hidden = [[], [], []]
        for images, _ in loader:
            images = images.to(d.device)
            _, _, hidden_list = d.model(images, return_hidden=True)
            for i in range(3):
                all_hidden[i].append(hidden_list[i].cpu().numpy())
        hidden = [np.concatenate(h, axis=0) for h in all_hidden]
        return np.concatenate(hidden, axis=1)[:n_samples]

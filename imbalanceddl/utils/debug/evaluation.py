import torch
import torch.nn.functional as F
import numpy as np
from imbalanceddl.utils.metrics import shot_acc
from imbalanceddl.utils.plugin_rule import define_groups_2, compute_aurc_metrics
from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs, build_mixture, uniform_weights,
)
from .metrics import compute_chow_aurc, compute_all_metrics
from .metrics import print_uniform_comparison, print_method_vs_uniform_comparison, print_ce_comparison, print_final_method_comparison
from .diagnostics import print_expert_agreement, print_stage3_plugin_params, print_per_class_extreme_routing


def recipe_from_checkpoint(gate_ckpt, cfg, la_tau=None, T=None):
    """Extract the exact mixture recipe a gate checkpoint was trained with.

    Every eval script must use this recipe (via ``extract_data`` /
    ``build_mixture``) so evaluation matches training (RC2 fix). Keys that
    predate the metadata get safe defaults.
    """
    expert_temps = list(gate_ckpt.get('expert_temps', [1.0, 1.0, 1.0]))
    return {
        'T': T if T is not None else gate_ckpt.get('temperature', 1.0),
        'la_tau': la_tau if la_tau is not None else 1.5,
        'expert_temps': expert_temps,
        'k': gate_ckpt.get('k', getattr(cfg, 'routing_sparsity', 2)),
        'space': gate_ckpt.get('mix_space', getattr(cfg, 'mix_space', 'logit')),
        'weight_floor': gate_ckpt.get('weight_floor', 0.0),
        'gate_temp': gate_ckpt.get('gate_temp', 1.0),
        'mix_temp': gate_ckpt.get('mix_temp', 1.0),
        'norm_blocks': gate_ckpt.get('norm_blocks', True),
        'freq_features': gate_ckpt.get('freq_features', False),
        'cls_num_list': list(cfg.cls_num_list),
    }


@torch.no_grad()
def extract_data(model, gate, loader, device, recipe):
    """Extract posteriors with the checkpoint's exact mixture recipe.

    Returns (p_mix, p_uniform, p_ce, p_la, p_bs, l_ce, l_la, l_bs, weights,
    labels, gate_logits). ``p_uniform`` is the same recipe with equal weights
    over all experts (the fair baseline for "did routing help").
    """
    all_logits = [[], [], []]
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits_list, embeddings = model(images)
        for i in range(3):
            all_logits[i].append(logits_list[i])
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_logits = [torch.cat(l, dim=0) for l in all_logits]
    all_embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    cls_num_list = recipe['cls_num_list']
    la_tau = recipe['la_tau']
    T = recipe['T']
    expert_temps = recipe['expert_temps']
    k = recipe['k']
    space = recipe['space']
    weight_floor = recipe['weight_floor']
    gate_temp = recipe['gate_temp']
    mix_temp = recipe['mix_temp']

    adj_probs = calibrate_expert_probs(
        all_logits, cls_num_list, la_tau, T, expert_temps
    )
    p_ce, p_la, p_bs = adj_probs

    gate_logits = gate(all_embeddings)
    weights = F.softmax(gate_logits / gate_temp, dim=1)

    p_mix = build_mixture(
        all_logits, weights, cls_num_list, la_tau, T=T,
        per_expert_T=expert_temps, k=k, space=space,
        weight_floor=weight_floor, mix_temperature=mix_temp,
    )
    unif_w = uniform_weights(weights.size(0), 3, device=weights.device)
    p_uniform = build_mixture(
        all_logits, unif_w, cls_num_list, la_tau, T=T,
        per_expert_T=expert_temps, k=None, space=space,
        mix_temperature=1.0,
    )

    return (p_mix.cpu().numpy(), p_uniform.cpu().numpy(),
            p_ce.cpu().numpy(), p_la.cpu().numpy(), p_bs.cpu().numpy(),
            all_logits[0].cpu(), all_logits[1].cpu(), all_logits[2].cpu(),
            weights.cpu().numpy(), labels.cpu().numpy(),
            gate_logits.cpu())


def run_metric_comparisons(p_mix_tune, p_unif_tune, p_ce_tune, p_la_tune, p_mix_test, p_unif_test, p_ce_test, p_la_test, p_bs_test, l_ce_test, l_la_test, l_bs_test, labels_tune, labels_test, group_ids_2, cfg, train_dataset):
    print("\n[INFO] Computing AURC & Calibration metrics...")
    chow_bal = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='bal')
    chow_wst = compute_chow_aurc(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, mode='worst')

    la_bal = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    la_wst = compute_aurc_metrics(p_la_tune, labels_tune, p_la_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    ce_bal = compute_aurc_metrics(p_ce_tune, labels_tune, p_ce_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    ce_wst = compute_aurc_metrics(p_ce_tune, labels_tune, p_ce_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    unif_bal = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    unif_wst = compute_aurc_metrics(p_unif_tune, labels_tune, p_unif_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    method_bal = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='bal')
    method_wst = compute_aurc_metrics(p_mix_tune, labels_tune, p_mix_test, labels_test, group_ids_2, cls_num_list=cfg.cls_num_list, mode='worst')

    m_ce = compute_all_metrics(p_ce_test, labels_test, l_ce_test, cfg, train_dataset)
    m_la = compute_all_metrics(p_la_test, labels_test, l_la_test, cfg, train_dataset)
    m_bs = compute_all_metrics(p_bs_test, labels_test, l_bs_test, cfg, train_dataset)
    m_unif = compute_all_metrics(p_unif_test, labels_test, None, cfg, train_dataset)
    m_method = compute_all_metrics(p_mix_test, labels_test, None, cfg, train_dataset)

    print_uniform_comparison(m_unif, m_unif['nll'], m_unif['tail_ece'], m_unif['brier'], unif_bal['AURC'], unif_wst['AURC'])
    print_method_vs_uniform_comparison(m_method, m_unif, method_bal['AURC'], unif_bal['AURC'], method_wst['AURC'], unif_wst['AURC'])
    print_ce_comparison(m_ce, ce_bal['AURC'], ce_wst['AURC'])
    print_final_method_comparison(m_method, method_bal['AURC'], method_wst['AURC'])

    print("\n" + "="*140)
    print("TABLE 2: FULL DIAGNOSTIC BREAKDOWN (TEST SET) - PAPER BASELINES INCLUDED")
    print("="*140)
    print(f"{'Method':<15} | {'Bal Acc':<7} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<8} | {'Brier':<8} | {'ECE':<8} | {'Tail ECE':<8} | {'Mean Logit':<10} | {'%>10':<6} | {'%>20':<6}")
    print("-"*140)

    def print_row(name, m):
        print(f"{name:<15} | {m['bal_acc']:<7.2f} | {m['many']:<6.2f} | {m['med']:<6.2f} | {m['low']:<6.2f} | {m['nll']:<8.3f} | {m['brier']:<8.3f} | {m['ece']:<8.3f} | {m['tail_ece']:<8.3f} | {m.get('mean_logit', 0):<10.2f} | {m.get('sat_10', 0):<6.1f} | {m.get('sat_20', 0):<6.1f}")

    def print_paper_row(name, bal, many, med, low, nll, brier, ece, tail_ece):
        print(f"{name:<15} | {bal:<7} | {many:<6} | {med:<6} | {low:<6} | {nll:<8} | {brier:<8} | {ece:<8} | {tail_ece:<8} | {'N/A':<10} | {'N/A':<6} | {'N/A':<6}")

    print_paper_row("Paper's Method", "N/A", "N/A", "N/A", "N/A", "1.18", "0.403", "N/A", "0.088")
    print_paper_row("Paper Unif", "N/A", "N/A", "N/A", "N/A", "1.30", "0.442", "N/A", "0.171")
    print("-"*140)
    print_row("YOUR CE", m_ce)
    print_row("YOUR LA", m_la)
    print_row("YOUR BS", m_bs)
    print_row("YOUR Uniform", m_unif)
    print_row("My Method", m_method)
    print("="*140)


def run_temperature_comparison(recipe, l_ce_test, l_la_test, l_bs_test,
                               gate_logits_test, labels_test, cfg,
                               train_dataset, m_unif, m_method):
    """Recipe mixture vs the same mixture at global T=1.0.

    Both variants use the checkpoint's per-expert temperatures and the
    correct la_tau bias (fixing the previous la_tau omission). ``m_unif`` /
    ``m_method`` are the metrics of the recipe posteriors (computed by the
    caller from ``extract_data`` outputs).
    """
    T = recipe['T']
    print("\n" + "="*110)
    print("RAW T=1.0 vs RECIPE T={} COMPARISON".format(T))
    print("="*110)

    raw_logits = [l_ce_test, l_la_test, l_bs_test]
    g = gate_logits_test.detach().clone()

    def mix_with(Tg, gate_temp, mix_temp):
        weights = F.softmax(g / gate_temp, dim=1)
        return build_mixture(
            raw_logits, weights, recipe['cls_num_list'], recipe['la_tau'],
            T=Tg, per_expert_T=recipe['expert_temps'],
            k=recipe['k'], space=recipe['space'],
            weight_floor=recipe['weight_floor'],
            mix_temperature=mix_temp,
        )

    def unif_with(Tg):
        unif_w = uniform_weights(g.size(0), 3)
        return build_mixture(
            raw_logits, unif_w, recipe['cls_num_list'], recipe['la_tau'],
            T=Tg, per_expert_T=recipe['expert_temps'], k=None,
            space=recipe['space'], mix_temperature=1.0,
        )

    # T=1.0 variants: global T=1.0, gate temp and mixture temp left at 1.0
    # (LA bias is now correctly la_tau-scaled inside calibrate_expert_logits).
    p_unif_T1 = unif_with(1.0)
    p_mix_T1 = mix_with(1.0, 1.0, 1.0)

    m_unif_T1 = compute_all_metrics(p_unif_T1.numpy(), labels_test, None, cfg, train_dataset)
    m_method_T1 = compute_all_metrics(p_mix_T1.numpy(), labels_test, None, cfg, train_dataset)

    print(f"{'Metric':<35} | {'Unif @ T=1.0':<15} | {'Unif @ T={}'.format(T):<15} | {'Method @ T=1.0':<15} | {'Method @ T={}'.format(T):<15}")
    print("-"*110)

    def print_T_row(name, val_u1, val_uT, val_m1, val_mT):
        print(f"{name:<35} | {val_u1:<15.4f} | {val_uT:<15.4f} | {val_m1:<15.4f} | {val_mT:<15.4f}")

    print_T_row("NLL (lower is better)", m_unif_T1['nll'], m_unif['nll'], m_method_T1['nll'], m_method['nll'])
    print_T_row("Brier (lower is better)", m_unif_T1['brier'], m_unif['brier'], m_method_T1['brier'], m_method['brier'])
    print_T_row("ECE All (lower is better)", m_unif_T1['ece'], m_unif['ece'], m_method_T1['ece'], m_method['ece'])
    print_T_row("tail-ECE (lower is better)", m_unif_T1['tail_ece'], m_unif['tail_ece'], m_method_T1['tail_ece'], m_method['tail_ece'])
    print_T_row("Bal Acc (higher is better)", m_unif_T1['bal_acc'], m_unif['bal_acc'], m_method_T1['bal_acc'], m_method['bal_acc'])
    print("="*110)


def run_saves_the_day_checks(p_ce_test, p_la_test, p_bs_test, w_test, labels_test, label_groups_test, k):
    print("\n" + "="*100)
    print("LA 'SAVES THE DAY' ROUTING CHECK (Tail Samples where LA is right, CE & BS are wrong)")
    print("="*100)

    ce_preds_test = np.argmax(p_ce_test, axis=1)
    la_preds_test = np.argmax(p_la_test, axis=1)
    bs_preds_test = np.argmax(p_bs_test, axis=1)

    tail_mask_check = (label_groups_test == 1)
    la_correct_mask = (la_preds_test == labels_test)
    ce_bs_wrong_mask = (ce_preds_test != labels_test) & (bs_preds_test != labels_test)
    la_saves_day_mask = tail_mask_check & la_correct_mask & ce_bs_wrong_mask
    la_saves_day_indices = np.where(la_saves_day_mask)[0]

    total_la_saves_day = len(la_saves_day_indices)

    if total_la_saves_day == 0:
        print("[INFO] No samples found where LA was the sole correct expert on Tail classes.")
        return la_saves_day_indices

    print(f"[INFO] Found {total_la_saves_day} samples where LA was the sole correct expert on Tail classes.")
    avg_w_la_saves = np.mean(w_test[la_saves_day_indices], axis=0)
    print(f"Average Routing Weights for these samples: CE={avg_w_la_saves[0]:.4f} | LA={avg_w_la_saves[1]:.4f} | BS={avg_w_la_saves[2]:.4f}")
    topk_indices_la_saves = np.argsort(w_test[la_saves_day_indices], axis=1)[:, ::-1][:, :k]
    la_chosen_count = np.sum(topk_indices_la_saves == 1)
    print(f"LA was chosen in Top-{k} routing for {la_chosen_count}/{total_la_saves_day} of these samples ({la_chosen_count/total_la_saves_day*100:.1f}%)")
    print(f"\n{'Idx':<6} | {'True':<5} | {'CE_Pred':<8} | {'LA_Pred':<8} | {'BS_Pred':<8} | {'w_CE':<6} | {'w_LA':<6} | {'w_BS':<6} | Top-k Chosen")
    print("-"*100)
    for i in la_saves_day_indices[:15]:
        w = w_test[i]
        topk_idx = np.argsort(w)[::-1][:k]
        ce_pred = ce_preds_test[i]
        la_pred = la_preds_test[i]
        bs_pred = bs_preds_test[i]
        true_label = labels_test[i]
        experts_chosen = []
        if 0 in topk_idx: experts_chosen.append("CE")
        if 1 in topk_idx: experts_chosen.append("LA")
        if 2 in topk_idx: experts_chosen.append("BS")
        experts_chosen_str = ",".join(experts_chosen)
        print(f"{i:<6} | {true_label:<5} | {ce_pred:<8} | {la_pred:<8} | {bs_pred:<8} | {w[0]:<6.3f} | {w[1]:<6.3f} | {w[2]:<6.3f} | {experts_chosen_str}")
    print("="*100)
    return la_saves_day_indices


def run_raw_prob_inspection(la_saves_day_indices, p_ce_test, p_la_test, p_bs_test, w_test, labels_test):
    if len(la_saves_day_indices) == 0: return

    print("\n" + "="*100)
    print("RAW PROBABILITY INSPECTION FOR LA 'SAVES THE DAY' SAMPLES")
    print("="*100)
    print("[INFO] Inspecting up to 15 samples. Checking if BS/CE are overconfident when wrong, and LA is underconfident when right.")
    print("-"*100)

    def get_entropy(p):
        return -np.sum(p * np.log(p + 1e-8))

    print(f"{'Idx':<6} | {'Expert':<7} | {'True Prob':<10} | {'Pred':<8} | {'Max Prob':<10} | {'Entropy':<10} | {'w_Gate':<8}")
    print("-"*100)

    for i in la_saves_day_indices[:15]:
        true_label = labels_test[i]
        for exp_idx, exp_name in enumerate(["CE", "LA", "BS"]):
            if exp_name == "CE": probs = p_ce_test[i]
            elif exp_name == "LA": probs = p_la_test[i]
            else: probs = p_bs_test[i]
            true_prob = probs[true_label]
            pred = np.argmax(probs)
            max_prob = probs[pred]
            ent = get_entropy(probs)
            idx_str = str(i) if exp_idx == 0 else ""
            print(f"{idx_str:<6} | {exp_name:<7} | {true_prob:<10.4f} | {pred:<8} | {max_prob:<10.4f} | {ent:<10.4f} | {w_test[i, exp_idx]:<8.4f}")
        print("-"*100)
    print("="*100)


def run_oracle_diagnostic(p_ce_test, p_la_test, p_bs_test, p_mix_test, labels_test, head_mask, tail_mask, cfg, train_dataset):
    print("\n" + "="*100)
    print("ORACLE EXPERT DIAGNOSTIC TEST")
    print("="*100)

    true_probs = np.stack([
        p_ce_test[np.arange(len(labels_test)), labels_test],
        p_la_test[np.arange(len(labels_test)), labels_test],
        p_bs_test[np.arange(len(labels_test)), labels_test]
    ], axis=1)

    oracle_expert_indices = np.argmax(true_probs, axis=1)
    oracle_preds = np.zeros_like(labels_test)
    for i in range(len(labels_test)):
        exp_idx = oracle_expert_indices[i]
        if exp_idx == 0: oracle_preds[i] = np.argmax(p_ce_test[i])
        elif exp_idx == 1: oracle_preds[i] = np.argmax(p_la_test[i])
        else: oracle_preds[i] = np.argmax(p_bs_test[i])

    m_oracle = compute_all_metrics(p_mix_test, labels_test, None, cfg, train_dataset)
    oracle_bal_acc = np.mean([np.mean(oracle_preds[labels_test == c] == c) for c in range(cfg.num_classes) if np.sum(labels_test == c) > 0]) * 100

    oracle_many, oracle_med, oracle_low = shot_acc(cfg, oracle_preds, labels_test, train_dataset, acc_per_cls=False)

    print(f"Oracle Balanced Accuracy: {oracle_bal_acc:.2f}%")
    print(f"Oracle Many Acc:          {oracle_many*100:.2f}%")
    print(f"Oracle Med Acc:           {oracle_med*100:.2f}%")
    print(f"Oracle Low Acc:           {oracle_low*100:.2f}%")
    print("-"*100)

    ce_oracle_count = np.sum(oracle_expert_indices == 0)
    la_oracle_count = np.sum(oracle_expert_indices == 1)
    bs_oracle_count = np.sum(oracle_expert_indices == 2)
    total_samples = len(labels_test)

    print(f"Expert Chosen as Oracle:")
    print(f"  CE: {ce_oracle_count}/{total_samples} ({ce_oracle_count/total_samples*100:.1f}%)")
    print(f"  LA: {la_oracle_count}/{total_samples} ({la_oracle_count/total_samples*100:.1f}%)")
    print(f"  BS: {bs_oracle_count}/{total_samples} ({bs_oracle_count/total_samples*100:.1f}%)")

    print("\nOracle Choice by Group:")
    head_oracle_choices = oracle_expert_indices[head_mask]
    tail_oracle_choices = oracle_expert_indices[tail_mask]
    print(f"  Head ({len(head_oracle_choices)} samples): CE={np.sum(head_oracle_choices==0)} | LA={np.sum(head_oracle_choices==1)} | BS={np.sum(head_oracle_choices==2)}")
    print(f"  Tail ({len(tail_oracle_choices)} samples): CE={np.sum(tail_oracle_choices==0)} | LA={np.sum(tail_oracle_choices==1)} | BS={np.sum(tail_oracle_choices==2)}")
    print("="*100)

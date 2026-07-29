import torch
import numpy as np
from imbalanceddl.utils.plugin_rule import tune_plugin_for_rho

def print_gate_feature_importance(gate):
    print("\n" + "="*80)
    print("GATE MLP FEATURE IMPORTANCE (L1 NORM OF INPUT WEIGHTS)")
    print("="*80)
    
    fc1_weights = gate.fc1.weight.detach().cpu().numpy()
    feature_importance = np.sum(np.abs(fc1_weights), axis=0)
    
    feat_names = [
        "CE_Ent", "CE_Max", "CE_Marg", "CE_Top5", "CE_Tail", "CE_Cos", "CE_KL",
        "LA_Ent", "LA_Max", "LA_Marg", "LA_Top5", "LA_Tail", "LA_Cos", "LA_KL",
        "BS_Ent", "BS_Max", "BS_Marg", "BS_Top5", "BS_Tail", "BS_Cos", "BS_KL",
        "Glb_MeanEnt", "Glb_ClassVar", "Glb_ConfDisp"
    ]
    
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    print(f"{'Feature':<15} | {'Importance (L1 Norm)':<20}")
    print("-"*40)
    for idx in sorted_idx:
        print(f"{feat_names[idx]:<15} | {feature_importance[idx]:<20.6f}")
    print("="*80)

def print_expert_agreement(p_mix_test, ce_preds, la_preds, bs_preds, labels_test):
    print("\n" + "="*80)
    print("EXPERT AGREEMENT ON CORRECT VS. INCORRECT PREDICTIONS")
    print("="*80)
    
    method_preds = np.argmax(p_mix_test, axis=1)
    correct_mask = (method_preds == labels_test)
    incorrect_mask = ~correct_mask
    
    agree_correct = np.mean((ce_preds[correct_mask] == la_preds[correct_mask]) & (la_preds[correct_mask] == bs_preds[correct_mask]))
    agree_incorrect = np.mean((ce_preds[incorrect_mask] == la_preds[incorrect_mask]) & (la_preds[incorrect_mask] == bs_preds[incorrect_mask]))
    
    print(f"Total Correct: {np.sum(correct_mask)} | Total Incorrect: {np.sum(incorrect_mask)}")
    print(f"Expert Agreement when Mixture is CORRECT:   {agree_correct*100:.2f}%")
    print(f"Expert Agreement when Mixture is INCORRECT: {agree_incorrect*100:.2f}%")
    print("[INFO] If incorrect agreement is high, experts share the same blind spots.")
    print("[INFO] If incorrect agreement is low, gate is failing to pick the correct expert.")
    print("="*80)

def print_stage3_plugin_params(p_mix_tune, labels_tune, group_ids_2, cfg):
    print("\n" + "="*80)
    print("STAGE 3 PLUG-IN PARAMETERS (AT 50% REJECTION RATE)")
    print("="*80)
    
    alpha_bal, mu_bal = tune_plugin_for_rho(p_mix_tune, labels_tune, group_ids_2, rho=0.5, mode='bal', cls_num_list=cfg.cls_num_list)
    print(f"Plug-in [Bal] alpha: Head={alpha_bal[0]:.4f}, Tail={alpha_bal[1]:.4f}")
    print(f"Plug-in [Bal] mu:    Head={mu_bal[0]:.4f}, Tail={mu_bal[1]:.4f}")
    
    alpha_wst, mu_wst = tune_plugin_for_rho(p_mix_tune, labels_tune, group_ids_2, rho=0.5, mode='worst', cls_num_list=cfg.cls_num_list)
    print(f"\nPlug-in [Wst] alpha: Head={alpha_wst[0]:.4f}, Tail={alpha_wst[1]:.4f}")
    print(f"Plug-in [Wst] mu:    Head={mu_wst[0]:.4f}, Tail={mu_wst[1]:.4f}")
    print("[INFO] If Tail alpha is near 0, the rejector is killing the tail group to minimize risk.")
    print("="*80)

def print_per_class_extreme_routing(w_test, labels_test, cfg):
    print("\n" + "="*80)
    print("PER-CLASS EXTREME ROUTING (Top 5 Head vs Top 5 Tail Classes)")
    print("="*80)
    
    cls_num_list_np = np.array(cfg.cls_num_list)
    sorted_classes = np.argsort(cls_num_list_np)
    
    top5_head_cls = sorted_classes[-5:]
    top5_tail_cls = sorted_classes[:5]
    
    print(f"{'Class':<6} | {'Num Samples':<12} | {'Avg w_CE':<10} | {'Avg w_LA':<10} | {'Avg w_BS':<10}")
    print("-"*55)
    
    print("Top 5 Head Classes:")
    for c in top5_head_cls:
        mask = (labels_test == c)
        if np.sum(mask) > 0:
            avg_w = np.mean(w_test[mask], axis=0)
            print(f"{c:<6} | {cls_num_list_np[c]:<12} | {avg_w[0]:<10.4f} | {avg_w[1]:<10.4f} | {avg_w[2]:<10.4f}")
            
    print("\nTop 5 Tail Classes:")
    for c in top5_tail_cls:
        mask = (labels_test == c)
        if np.sum(mask) > 0:
            avg_w = np.mean(w_test[mask], axis=0)
            print(f"{c:<6} | {cls_num_list_np[c]:<12} | {avg_w[0]:<10.4f} | {avg_w[1]:<10.4f} | {avg_w[2]:<10.4f}")
    print("="*80)
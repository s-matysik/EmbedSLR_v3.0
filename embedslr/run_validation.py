#!/usr/bin/env python3
"""
run_validation.py -- Full MCDA validation suite for EmbedSLR 2.0
================================================================

Runs all 10 tests from mcda_validation.py on the provided Scopus CSV.
Generates a comprehensive text report and per-test CSV exports.

Usage:
    python run_validation.py --csv path/to/scopus_export.csv
    python run_validation.py --csv data.csv --output results/ --n-mc 2000

Requirements (minimal): pandas, numpy, scipy
  (sentence-transformers, openai etc. NOT needed for validation)
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
from pathlib import Path
from typing import Dict, Optional, Set
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedslr.advanced_scoring import (
    parse_keywords_cell, parse_references_cell,
    frequency_map, per_item_topk_sum,
)
from embedslr.mcda_validation import (
    weight_sensitivity_oat, criteria_removal_analysis,
    cross_method_correlation, parameter_sensitivity,
    precision_recall_at_k, bootstrap_stability,
    monte_carlo_weights, rank_reversal_analysis,
    normalization_comparison, compromise_ranking,
    generate_validation_report,
)
warnings.filterwarnings('ignore', category=FutureWarning)

# =====================================================================
#  Data preparation
# =====================================================================

def _prepare_criteria_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute raw criteria values from Scopus CSV columns."""
    out = df.copy()
    # Semantic
    if 'distance_cosine' in out.columns:
        out['semantic_sim'] = 1.0 - out['distance_cosine'].astype(float)
    elif 'semantic_similarity' in out.columns:
        out['semantic_sim'] = out['semantic_similarity'].astype(float)
    else:
        out['semantic_sim'] = np.random.RandomState(42).uniform(0.3, 0.9, len(out))
    # Keywords
    kw_col = next((c for c in ['Author Keywords','Authors Keywords',
                                'Keywords','Index Keywords'] if c in out.columns), None)
    if kw_col:
        kw_lists = [parse_keywords_cell(x) for x in out[kw_col]]
        kw_freq = frequency_map(kw_lists)
        out['kw_sum'] = [per_item_topk_sum(kws, kw_freq, 5)[0] or np.nan for kws in kw_lists]
    else:
        out['kw_sum'] = np.nan
    # References
    ref_col = next((c for c in ['References','Cited References',
                                 'Reference'] if c in out.columns), None)
    if ref_col:
        ref_lists = [parse_references_cell(x) for x in out[ref_col]]
        ref_freq = frequency_map(ref_lists)
        out['ref_sum'] = [per_item_topk_sum(refs, ref_freq, 15)[0] or np.nan for refs in ref_lists]
    else:
        out['ref_sum'] = np.nan
    # Citations
    cit_col = next((c for c in ['Cited by','Cited By',
                                 'Times Cited','Citations'] if c in out.columns), None)
    out['citation_count'] = (pd.to_numeric(out[cit_col], errors='coerce').fillna(0)
                             if cit_col else 0.0)
    # Fill NaN
    for col in ['kw_sum', 'ref_sum']:
        m = out[col].mean()
        out[col] = out[col].fillna(m if not np.isnan(m) else 0.0)
    return out

# =====================================================================
#  Scoring adapters: (df, weights_dict) -> pd.Series
# =====================================================================

def _make_l_scoring_func(prep):
    """L-Scoring: rank-based points, weighted sum."""
    def fn(df_in, w):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        P = len(src)
        w = {k: v/sum(w.values()) for k, v in w.items()}
        cmap = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
                'references': src['ref_sum'], 'citations': src['citation_count']}
        total = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cmap:
                total += (P - (cmap[c].rank(ascending=False, method='average') - 1.0)) * wt
        return total
    return fn

def _make_z_scoring_func(prep):
    """Z-Scoring: z-standardized weighted sum."""
    def fn(df_in, w):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        w = {k: v/sum(w.values()) for k, v in w.items()}
        cmap = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
                'references': src['ref_sum'], 'citations': src['citation_count']}
        total = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cmap:
                raw = cmap[c].astype(float)
                sd = raw.std(ddof=0)
                z = (raw - raw.mean()) / sd if sd > 1e-12 else pd.Series(0.0, index=raw.index)
                total += z * wt
        return total
    return fn

def _make_lp_func(prep, bsz=2.0, bfz=4.0):
    """L-Scoring+: L-Scoring base + outlier bonus."""
    def fn(df_in, w):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        P = len(src)
        w = {k: v/sum(w.values()) for k, v in w.items()}
        cmap = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
                'references': src['ref_sum'], 'citations': src['citation_count']}
        total = pd.Series(0.0, index=src.index)
        bonus = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cmap:
                total += (P - (cmap[c].rank(ascending=False, method='average') - 1.0)) * wt
                raw = cmap[c].astype(float)
                sd = raw.std(ddof=0)
                if sd > 1e-12:
                    z_med = (raw - raw.median()) / sd
                    frac = ((z_med - bsz) / (bfz - bsz)).clip(lower=0.0)
                    frac[z_med >= bfz] = 1.0
                    bonus += frac * float(P)
        return total + bonus.clip(upper=float(P))
    return fn

# =====================================================================
#  Main validation runner
# =====================================================================

def run_full_validation(csv_path, output_dir='validation_results',
                        base_weights=None, n_mc=1000, n_bs=500, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    if base_weights is None:
        base_weights = {'semantic': 0.40, 'keywords': 0.25,
                        'references': 0.25, 'citations': 0.10}
    criteria = list(base_weights.keys())

    if verbose: print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if verbose: print(f"  {len(df)} publications loaded")
    prep = _prepare_criteria_columns(df)

    l_fn = _make_l_scoring_func(prep)
    z_fn = _make_z_scoring_func(prep)
    lp_fn = _make_lp_func(prep)
    results = {}

    # TEST 1: Weight Sensitivity OAT
    if verbose: print("\n[TEST 1] Weight Sensitivity (OAT)...")
    r1 = weight_sensitivity_oat(
        l_fn, prep, base_weights,
        perturbations=(-0.50, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.50))
    results['weight_sensitivity'] = r1
    r1.correlations.to_csv(f'{output_dir}/test1_weight_sensitivity.csv', index=False)
    if verbose:
        print(f"  Mean WS={r1.summary['mean_ws']:.4f}  "
              f"Sensitive: {r1.summary['most_sensitive_criterion']}")

    # TEST 2: Criteria Removal
    if verbose: print("\n[TEST 2] Criteria Removal...")
    r2 = criteria_removal_analysis(l_fn, prep, base_weights)
    results['criteria_removal'] = r2
    r2.removal_steps.to_csv(f'{output_dir}/test2_criteria_removal.csv', index=False)
    if verbose:
        print(f"  Critical: {r2.critical_criterion}  "
              f"Resilient: {r2.most_resilient_criterion}")

    # TEST 3: Cross-Method Correlation
    if verbose: print("\n[TEST 3] Cross-Method Correlation...")
    rankings = {
        'L-Scoring': l_fn(prep, base_weights),
        'Z-Scoring': z_fn(prep, base_weights),
        'L-Scoring+': lp_fn(prep, base_weights),
    }
    cm_df, ws_mat = cross_method_correlation(rankings)
    results['cross_method'] = (cm_df, ws_mat)
    cm_df.to_csv(f'{output_dir}/test3_cross_method.csv', index=False)
    ws_mat.to_csv(f'{output_dir}/test3_ws_matrix.csv')
    if verbose:
        for _, r in cm_df.iterrows():
            print(f"  {r['method_1']} <-> {r['method_2']}: WS={r['ws']:.4f}")

    # TEST 4: Parameter Sensitivity
    if verbose: print("\n[TEST 4] Parameter Sensitivity...")
    p_results = {}

    def lp_bz(df_in, w, bz):
        return _make_lp_func(prep, bsz=bz, bfz=bz + 2.0)(df_in, w)
    p_results['bonus_start_z'] = parameter_sensitivity(
        lp_bz, prep, 'bonus_start_z',
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 2.0, base_weights)

    def l_topk(df_in, w, topk):
        kw_col = next((c for c in ['Author Keywords', 'Authors Keywords',
                                     'Keywords'] if c in prep.columns), None)
        if not kw_col:
            return l_fn(df_in, w)
        kw_lists = [parse_keywords_cell(x) for x in prep[kw_col]]
        kw_freq = frequency_map(kw_lists)
        mod = prep.copy()
        mod['kw_sum'] = [per_item_topk_sum(kws, kw_freq, topk)[0] or np.nan
                         for kws in kw_lists]
        mod['kw_sum'] = mod['kw_sum'].fillna(mod['kw_sum'].mean())
        return _make_l_scoring_func(mod)(df_in, w)
    p_results['top_keywords_k'] = parameter_sensitivity(
        l_topk, prep, 'top_keywords_k',
        [3, 5, 7, 10, 15], 5, base_weights)

    results['parameter_sensitivity'] = p_results
    for pn, pdf in p_results.items():
        pdf.to_csv(f'{output_dir}/test4_param_{pn}.csv', index=False)
    if verbose:
        for pn, pdf in p_results.items():
            print(f"  {pn}: min WS={pdf['ws'].min():.4f}")

    # TEST 5: Precision / Recall / F1
    if verbose: print("\n[TEST 5] Precision / Recall / F1...")
    combined = (prep['semantic_sim'].rank(pct=True) * 0.4
                + prep['kw_sum'].rank(pct=True) * 0.3
                + prep['ref_sum'].rank(pct=True) * 0.2
                + prep['citation_count'].rank(pct=True) * 0.1)
    relevant = set(combined[combined >= combined.quantile(0.80)].index.tolist())
    if verbose:
        print(f"  Simulated ground truth: {len(relevant)} relevant (top 20%)")
    r5 = precision_recall_at_k(rankings, relevant, k_values=(5, 10, 20, 50, 100))
    results['precision_recall'] = r5
    r5.per_k.to_csv(f'{output_dir}/test5_precision_recall.csv', index=False)
    if verbose:
        for m in ['L-Scoring', 'Z-Scoring', 'L-Scoring+', 'random_baseline']:
            p10 = r5.per_k[(r5.per_k['method'] == m) & (r5.per_k['k'] == 10)]['precision'].values
            print(f"  {m:20s} P@10={p10[0]:.3f}" if len(p10) else f"  {m}")

    # TEST 6: Bootstrap Stability
    if verbose: print(f"\n[TEST 6] Bootstrap Stability (n={n_bs})...")
    r6 = bootstrap_stability(l_fn, prep, base_weights,
                              n_bootstrap=n_bs, top_k_values=(5, 10, 20))
    results['bootstrap'] = r6
    r6.to_csv(f'{output_dir}/test6_bootstrap.csv', index=False)
    if verbose:
        for _, r in r6.iterrows():
            print(f"  Top-{int(r['top_k'])}: stability={r['mean_stability']:.3f}")

    # TEST 7: Monte Carlo Weight Space
    if verbose: print(f"\n[TEST 7] Monte Carlo Weights (n={n_mc})...")
    r7 = monte_carlo_weights(l_fn, prep, criteria,
                              n_samples=n_mc, reference_weights=base_weights)
    results['monte_carlo'] = r7
    r7.all_runs.to_csv(f'{output_dir}/test7_monte_carlo.csv', index=False)
    if verbose:
        s = r7.stability_summary
        print(f"  WS: mean={s['mean_ws']:.4f}  std={s['std_ws']:.4f}  "
              f"P5={s['pct5_ws']:.4f}")

    # TEST 8: Rank Reversal
    if verbose: print("\n[TEST 8] Rank Reversal...")
    r8 = rank_reversal_analysis(l_fn, prep, base_weights,
                                 n_removals=5, n_additions=3, top_k_monitor=20)
    results['rank_reversal'] = r8
    r8.removal_results.to_csv(f'{output_dir}/test8_rr_removal.csv', index=False)
    r8.addition_results.to_csv(f'{output_dir}/test8_rr_addition.csv', index=False)
    if verbose:
        print(f"  Reversal rate: {r8.reversal_rate:.4f} ({r8.reversal_count} total)")

    # TEST 9: Normalization Comparison
    if verbose: print("\n[TEST 9] Normalization Comparison...")
    dm = prep[['semantic_sim', 'kw_sum', 'ref_sum', 'citation_count']].copy()
    dm.columns = criteria
    norm_df, norm_scores = normalization_comparison(dm, base_weights)
    results['normalization'] = (norm_df, norm_scores)
    norm_df.to_csv(f'{output_dir}/test9_normalization.csv', index=False)
    if verbose:
        for _, r in norm_df.iterrows():
            print(f"  {r['norm_1']:8s} <-> {r['norm_2']:8s}: WS={r['ws']:.4f}")

    # TEST 10: Compromise Ranking
    if verbose: print("\n[TEST 10] Compromise Ranking...")
    r10 = compromise_ranking(rankings)
    results['compromise'] = r10
    r10.agreement_with_methods.to_csv(
        f'{output_dir}/test10_compromise.csv', index=False)
    if verbose:
        for _, r in r10.agreement_with_methods.iterrows():
            print(f"  {r['method']:15s} Borda WS={r['borda_ws']:.4f}  "
                  f"Copeland WS={r['copeland_ws']:.4f}")

    # Generate report
    report_path = f'{output_dir}/VALIDATION_REPORT.txt'
    report = generate_validation_report(results, report_path)
    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  All 10 tests completed in {elapsed:.1f}s")
        print(f"  Report: {report_path}")
        print(f"  CSVs:   {output_dir}/")
        print(f"{'=' * 60}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EmbedSLR 2.0 MCDA Validation Suite')
    parser.add_argument('--csv', required=True, help='Path to Scopus CSV')
    parser.add_argument('--output', default='validation_results',
                        help='Output directory')
    parser.add_argument('--n-mc', type=int, default=1000,
                        help='Monte Carlo samples')
    parser.add_argument('--n-bs', type=int, default=500,
                        help='Bootstrap iterations')
    args = parser.parse_args()
    run_full_validation(args.csv, args.output, n_mc=args.n_mc, n_bs=args.n_bs)

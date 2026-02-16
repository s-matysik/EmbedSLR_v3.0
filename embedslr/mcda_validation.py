"""
MCDA Validation & Sensitivity Analysis for EmbedSLR 2.0
=========================================================

Comprehensive framework for evaluating multi-criteria ranking methods
(L-Scoring, Z-Scoring, L-Scoring+, SMART) used in publication selection.

Tests implemented (based on MCDM literature):

  TEST 1  Weight Sensitivity OAT    — One-At-a-Time weight perturbation
  TEST 2  Criteria Removal          — Sequential & individual criterion exclusion
  TEST 3  Cross-Method Correlation  — Pairwise agreement between MCDA methods
  TEST 4  Parameter Sensitivity     — Internal method parameters (bonus_z, top_k, …)
  TEST 5  Precision / Recall / F1   — Against expert ground truth at multiple K
  TEST 6  Bootstrap Stability       — Resampling stability of rankings
  TEST 7  Monte Carlo Weight Space  — Random weight sampling (global sensitivity)
  TEST 8  Rank Reversal             — Robustness to alternative insertion/removal
  TEST 9  Normalization Comparison  — Effect of different normalization techniques
  TEST 10 Compromise Ranking        — Borda / Copeland aggregation across methods

Methodology references:
  [1] Wieckowski, Salabun (2023) Applied Soft Computing — SA taxonomy
  [2] Wieckowski, Kolodziejczyk, Salabun (2025) ISD2025 — criteria removal
  [3] Kizielewicz, Shekhovtsov, Salabun (2023) SoftwareX — pymcdm library
  [4] Nabavi, Wang, Rangaiah (2023) — rank reversal analysis
  [5] Pamucar, Bozanic, Randelovic (2017) — MSI/CFI consistency tests
"""
from __future__ import annotations

import copy
import itertools
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional,
    Sequence, Set, Tuple, Union,
)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# =========================================================================
#                       CORRELATION  COEFFICIENTS
# =========================================================================

def spearman_rho(r1: np.ndarray, r2: np.ndarray) -> float:
    """Spearman rank-order correlation."""
    rho, _ = sp_stats.spearmanr(r1, r2)
    return float(rho) if not np.isnan(rho) else 0.0


def kendall_tau(r1: np.ndarray, r2: np.ndarray) -> float:
    """Kendall tau-b correlation."""
    tau, _ = sp_stats.kendalltau(r1, r2)
    return float(tau) if not np.isnan(tau) else 0.0


def weighted_spearman_ws(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    WS coefficient -- emphasises top-rank agreement.

    Salabun & Urbaniak (2020), used extensively in Wieckowski et al. (2025).
    WS = 1 - (6 * sum |r1_i - r2_i| * (N - r1_i + 1)(N - r2_i + 1))
             / (N^4 + N^3 - N^2 - N)
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    n = len(r1)
    if n <= 1:
        return 1.0
    denom = n ** 4 + n ** 3 - n ** 2 - n
    if denom == 0:
        return 1.0
    num = np.sum(np.abs(r1 - r2) * (n - r1 + 1) * (n - r2 + 1))
    return float(1.0 - 6.0 * num / denom)


def _rank_array(scores: Union[pd.Series, np.ndarray],
                ascending: bool = False) -> np.ndarray:
    """Convert scores -> dense ranks (1 = best by default)."""
    return pd.Series(scores).rank(ascending=ascending, method='average').values


def _top_k_set(scores: Union[pd.Series, np.ndarray], k: int) -> Set[int]:
    """Indices of top-k items by score (descending)."""
    arr = np.asarray(scores)
    return set(np.argsort(-arr)[:k].tolist())


def _all_correlations(r1: np.ndarray, r2: np.ndarray) -> Dict[str, float]:
    return {
        'spearman': spearman_rho(r1, r2),
        'kendall': kendall_tau(r1, r2),
        'ws': weighted_spearman_ws(r1, r2),
    }


def _overlap_at_k(scores1, scores2, k_values: Sequence[int]) -> Dict[str, float]:
    out = {}
    for k in k_values:
        t1 = _top_k_set(scores1, k)
        t2 = _top_k_set(scores2, k)
        out[f'top{k}_overlap'] = len(t1 & t2) / k if k > 0 else 0.0
    return out


# =========================================================================
#   TEST 1 -- Weight Sensitivity (OAT)
# =========================================================================

@dataclass
class WeightSensitivityResult:
    """Results of one-at-a-time weight perturbation analysis."""
    scenarios: List[Dict[str, float]]
    correlations: pd.DataFrame
    reference_ranking: np.ndarray
    summary: Dict[str, Any]


def weight_sensitivity_oat(
    scoring_func: Callable,
    df: pd.DataFrame,
    base_weights: Dict[str, float],
    perturbations: Sequence[float] = (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30),
    top_k_values: Sequence[int] = (5, 10, 20),
) -> WeightSensitivityResult:
    """
    TEST 1: Perturb each criterion weight +/-delta (proportionally redistribute
    to remaining criteria). Measure ranking stability via WS, Spearman, Kendall.

    scoring_func(df, weights_dict) -> pd.Series of scores
    """
    criteria = list(base_weights.keys())
    ref_scores = scoring_func(df, base_weights)
    ref_ranks = _rank_array(ref_scores)

    scenarios, rows = [], []
    for crit in criteria:
        for delta in perturbations:
            new_w = copy.deepcopy(base_weights)
            orig = new_w[crit]
            mod = max(0.01, orig * (1.0 + delta))
            diff = mod - orig
            others = [c for c in criteria if c != crit]
            other_sum = sum(new_w[c] for c in others)
            if other_sum > 0:
                for c in others:
                    new_w[c] = max(0.01, new_w[c] - diff * (new_w[c] / other_sum))
            new_w[crit] = mod
            s = sum(new_w.values())
            new_w = {k: v / s for k, v in new_w.items()}

            new_scores = scoring_func(df, new_w)
            new_ranks = _rank_array(new_scores)

            row = {'criterion': crit, 'perturbation': delta,
                   **_all_correlations(ref_ranks, new_ranks),
                   **_overlap_at_k(ref_scores, new_scores, top_k_values)}
            scenarios.append(new_w)
            rows.append(row)

    corr_df = pd.DataFrame(rows)
    grp = corr_df.groupby('criterion')['ws']
    summary = {
        'mean_spearman': corr_df['spearman'].mean(),
        'min_spearman': corr_df['spearman'].min(),
        'mean_ws': corr_df['ws'].mean(),
        'min_ws': corr_df['ws'].min(),
        'mean_kendall': corr_df['kendall'].mean(),
        'most_sensitive_criterion': grp.mean().idxmin(),
        'most_stable_criterion': grp.mean().idxmax(),
    }
    return WeightSensitivityResult(scenarios, corr_df, ref_ranks, summary)


def weight_sensitivity_scenarios(
    scoring_func: Callable,
    df: pd.DataFrame,
    scenarios: List[Dict[str, float]],
    reference_weights: Optional[Dict[str, float]] = None,
    top_k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Compare ranking under a list of arbitrary weight scenarios
    against a reference weight configuration.
    """
    if reference_weights is None:
        reference_weights = scenarios[0]
    ref_scores = scoring_func(df, reference_weights)
    ref_ranks = _rank_array(ref_scores)

    rows = []
    for i, w in enumerate(scenarios):
        scores = scoring_func(df, w)
        ranks = _rank_array(scores)
        row = {'scenario': i, 'weights': str(w),
               **_all_correlations(ref_ranks, ranks),
               **_overlap_at_k(ref_scores, scores, top_k_values)}
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
#   TEST 2 -- Criteria Removal Analysis
# =========================================================================

@dataclass
class CriteriaRemovalResult:
    """Results of sequential and individual criteria removal."""
    removal_steps: pd.DataFrame
    critical_criterion: str
    most_resilient_criterion: str


def criteria_removal_analysis(
    scoring_func: Callable,
    df: pd.DataFrame,
    base_weights: Dict[str, float],
    removal_order: Literal['ascending', 'descending'] = 'ascending',
    top_k_values: Sequence[int] = (5, 10, 20),
) -> CriteriaRemovalResult:
    """
    TEST 2: Remove criteria one-by-one (least -> most important) and measure
    ranking degradation. Follows Wieckowski et al. (ISD2025) methodology.

    Two sub-analyses:
      a) Sequential removal (cumulative)
      b) Individual removal (OAT)
    """
    criteria = list(base_weights.keys())
    sorted_c = sorted(criteria, key=lambda c: base_weights[c],
                       reverse=(removal_order == 'descending'))
    ref_scores = scoring_func(df, base_weights)
    ref_ranks = _rank_array(ref_scores)

    rows = []

    # (a) Sequential removal
    remaining = list(criteria)
    for step, crit in enumerate(sorted_c[:-1], 1):
        remaining = [c for c in remaining if c != crit]
        sub_w = {c: base_weights[c] for c in remaining}
        s = sum(sub_w.values())
        sub_w = {k: v / s for k, v in sub_w.items()}
        new_scores = scoring_func(df, sub_w)
        new_ranks = _rank_array(new_scores)
        row = {'type': 'sequential', 'step': step, 'removed': crit,
               'remaining': ', '.join(remaining), 'n_remaining': len(remaining),
               **_all_correlations(ref_ranks, new_ranks),
               **_overlap_at_k(ref_scores, new_scores, top_k_values)}
        rows.append(row)

    # (b) Individual (OAT) removal
    oat_ws = {}
    for crit in criteria:
        rem = [c for c in criteria if c != crit]
        sub_w = {c: base_weights[c] for c in rem}
        s = sum(sub_w.values())
        sub_w = {k: v / s for k, v in sub_w.items()}
        new_scores = scoring_func(df, sub_w)
        new_ranks = _rank_array(new_scores)
        ws_val = weighted_spearman_ws(ref_ranks, new_ranks)
        oat_ws[crit] = ws_val
        row = {'type': 'individual_removal', 'step': 0, 'removed': crit,
               'remaining': ', '.join(rem), 'n_remaining': len(rem),
               **_all_correlations(ref_ranks, new_ranks),
               **_overlap_at_k(ref_scores, new_scores, top_k_values)}
        rows.append(row)

    return CriteriaRemovalResult(
        pd.DataFrame(rows),
        critical_criterion=min(oat_ws, key=oat_ws.get),
        most_resilient_criterion=max(oat_ws, key=oat_ws.get),
    )


# =========================================================================
#   TEST 3 -- Cross-Method Correlation
# =========================================================================

def cross_method_correlation(
    rankings: Dict[str, pd.Series],
    top_k_values: Sequence[int] = (5, 10, 20),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TEST 3: Pairwise correlation (WS, Spearman, Kendall) and top-K overlap
    between all MCDA methods. Returns (pairwise_df, ws_matrix).
    """
    methods = list(rankings.keys())
    rank_arrays = {m: _rank_array(s) for m, s in rankings.items()}
    rows = []
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            r1, r2 = rank_arrays[m1], rank_arrays[m2]
            row = {'method_1': m1, 'method_2': m2,
                   **_all_correlations(r1, r2),
                   **_overlap_at_k(rankings[m1], rankings[m2], top_k_values)}
            rows.append(row)

    corr_df = pd.DataFrame(rows)
    ws_matrix = pd.DataFrame(1.0, index=methods, columns=methods)
    for _, row in corr_df.iterrows():
        ws_matrix.loc[row['method_1'], row['method_2']] = row['ws']
        ws_matrix.loc[row['method_2'], row['method_1']] = row['ws']
    return corr_df, ws_matrix


# =========================================================================
#   TEST 4 -- Parameter Sensitivity
# =========================================================================

def parameter_sensitivity(
    scoring_func_with_params: Callable,
    df: pd.DataFrame,
    param_name: str,
    param_values: Sequence[Any],
    base_value: Any,
    weights: Dict[str, float],
    top_k_values: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """
    TEST 4: Vary a single internal method parameter (e.g. bonus_start_z,
    top_keywords) and measure ranking stability against the baseline.

    scoring_func_with_params(df, weights, param_value) -> pd.Series
    """
    ref_scores = scoring_func_with_params(df, weights, base_value)
    ref_ranks = _rank_array(ref_scores)
    rows = []
    for val in param_values:
        scores = scoring_func_with_params(df, weights, val)
        ranks = _rank_array(scores)
        row = {'parameter': param_name, 'value': val,
               'is_baseline': (val == base_value),
               **_all_correlations(ref_ranks, ranks),
               **_overlap_at_k(ref_scores, scores, top_k_values)}
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
#   TEST 5 -- Precision / Recall / F1 @ K
# =========================================================================

@dataclass
class PrecisionRecallResult:
    per_k: pd.DataFrame
    methods: List[str]


def precision_recall_at_k(
    rankings: Dict[str, pd.Series],
    relevant_indices: Set[int],
    k_values: Sequence[int] = (5, 10, 20, 50),
    total_relevant: Optional[int] = None,
) -> PrecisionRecallResult:
    """
    TEST 5: Precision, Recall, F1 at multiple K values for each method.
    Includes a random-selection baseline with analytical expectation.
    """
    if total_relevant is None:
        total_relevant = len(relevant_indices)
    n = len(next(iter(rankings.values())))
    rows = []
    for method, scores in rankings.items():
        sorted_idx = np.argsort(-np.asarray(scores))
        for k in k_values:
            if k > n:
                continue
            tp = len(set(sorted_idx[:k].tolist()) & relevant_indices)
            p = tp / k
            r = tp / total_relevant if total_relevant > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            rows.append({
                'method': method, 'k': k,
                'precision': p, 'recall': r, 'f1': f1,
                'tp': tp, 'fp': k - tp,
            })

    # Random baseline (analytical)
    base_rate = total_relevant / n if n > 0 else 0
    for k in k_values:
        if k > n:
            continue
        exp_tp = k * base_rate
        p = base_rate
        r = exp_tp / total_relevant if total_relevant > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        rows.append({
            'method': 'random_baseline', 'k': k,
            'precision': p, 'recall': r, 'f1': f1,
            'tp': exp_tp, 'fp': k - exp_tp,
        })

    methods_list = list(rankings.keys()) + ['random_baseline']
    return PrecisionRecallResult(pd.DataFrame(rows), methods_list)


# =========================================================================
#   TEST 6 -- Bootstrap Stability
# =========================================================================

def bootstrap_stability(
    scoring_func: Callable,
    df: pd.DataFrame,
    weights: Dict[str, float],
    n_bootstrap: int = 500,
    sample_frac: float = 0.8,
    seed: int = 42,
    top_k_values: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    """
    TEST 6: Resample 80% of publications N times, re-run scoring, measure
    how often each article stays in top-K. Reports stability and CI for
    Spearman rho.
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    sample_size = max(2, int(n * sample_frac))
    ref_scores = scoring_func(df, weights)
    top_k_counts = {k: np.zeros(n) for k in top_k_values}
    spearman_vals = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=False)
        sub_df = df.iloc[idx].reset_index(drop=True)
        try:
            sub_scores = scoring_func(sub_df, weights)
        except Exception:
            continue
        sub_len = len(sub_scores)
        for k in top_k_values:
            actual_k = min(k, sub_len)
            top_sub = np.argsort(-sub_scores.values)[:actual_k]
            # top_sub indexes into sub_df; map back to original via idx
            for t in top_sub:
                if t < len(idx):
                    top_k_counts[k][idx[t]] += 1
        ref_sub = ref_scores.values[idx]
        r1 = sp_stats.rankdata(-ref_sub)
        r2 = sp_stats.rankdata(-sub_scores.values)
        rho, _ = sp_stats.spearmanr(r1, r2)
        if not np.isnan(rho):
            spearman_vals.append(rho)

    rows = []
    for k in top_k_values:
        freq = top_k_counts[k] / n_bootstrap
        ref_top_idx = np.argsort(-ref_scores.values)[:k]
        rows.append({
            'top_k': k,
            'mean_stability': float(freq[ref_top_idx].mean()),
            'std_stability': float(freq[ref_top_idx].std()),
            'min_stability': float(freq[ref_top_idx].min()),
            'mean_spearman': float(np.mean(spearman_vals)) if spearman_vals else np.nan,
            'ci95_lo': float(np.percentile(spearman_vals, 2.5)) if spearman_vals else np.nan,
            'ci95_hi': float(np.percentile(spearman_vals, 97.5)) if spearman_vals else np.nan,
        })
    return pd.DataFrame(rows)


# =========================================================================
#   TEST 7 -- Monte Carlo Weight Space (Global Sensitivity)
# =========================================================================

@dataclass
class MonteCarloWeightResult:
    """Global sensitivity analysis via random Dirichlet weight sampling."""
    all_runs: pd.DataFrame          # each run: weights + correlations
    stability_summary: Dict[str, float]
    position_frequency: pd.DataFrame  # how often each article lands in top-K


def monte_carlo_weights(
    scoring_func: Callable,
    df: pd.DataFrame,
    criteria: List[str],
    n_samples: int = 1000,
    reference_weights: Optional[Dict[str, float]] = None,
    top_k_values: Sequence[int] = (5, 10, 20),
    seed: int = 42,
    alpha: Optional[List[float]] = None,
) -> MonteCarloWeightResult:
    """
    TEST 7: Sample N random weight vectors from Dirichlet distribution,
    compute rankings, measure correlation with reference.

    Probabilistic + Global + Crisp + Criteria weights
    (Class 2 in Wieckowski & Salabun 2023 taxonomy)
    """
    rng = np.random.RandomState(seed)
    n_crit = len(criteria)
    if alpha is None:
        alpha = [1.0] * n_crit  # uniform Dirichlet

    if reference_weights is None:
        equal = 1.0 / n_crit
        reference_weights = {c: equal for c in criteria}

    ref_scores = scoring_func(df, reference_weights)
    ref_ranks = _rank_array(ref_scores)
    n_items = len(df)

    top_k_counts = {k: np.zeros(n_items) for k in top_k_values}
    rows = []

    for i in range(n_samples):
        w_vec = rng.dirichlet(alpha)
        w_dict = {c: float(w_vec[j]) for j, c in enumerate(criteria)}
        scores = scoring_func(df, w_dict)
        ranks = _rank_array(scores)

        row = {f'w_{c}': w_dict[c] for c in criteria}
        row['run'] = i
        row.update(_all_correlations(ref_ranks, ranks))
        row.update(_overlap_at_k(ref_scores, scores, top_k_values))
        rows.append(row)

        for k in top_k_values:
            top_idx = np.argsort(-scores.values)[:k]
            for idx in top_idx:
                top_k_counts[k][idx] += 1

    all_runs = pd.DataFrame(rows)
    stability_summary = {
        'mean_ws': float(all_runs['ws'].mean()),
        'std_ws': float(all_runs['ws'].std()),
        'min_ws': float(all_runs['ws'].min()),
        'pct5_ws': float(all_runs['ws'].quantile(0.05)),
        'mean_spearman': float(all_runs['spearman'].mean()),
        'min_spearman': float(all_runs['spearman'].min()),
    }

    pos_rows = []
    ref_top = {k: set(np.argsort(-ref_scores.values)[:k].tolist()) for k in top_k_values}
    for k in top_k_values:
        freq = top_k_counts[k] / n_samples
        for idx in range(n_items):
            pos_rows.append({
                'article_idx': idx, 'top_k': k,
                'frequency': float(freq[idx]),
                'in_reference_topk': idx in ref_top[k],
            })
    pos_freq = pd.DataFrame(pos_rows)

    return MonteCarloWeightResult(all_runs, stability_summary, pos_freq)


# =========================================================================
#   TEST 8 -- Rank Reversal Analysis
# =========================================================================

@dataclass
class RankReversalResult:
    """Results of rank reversal testing."""
    removal_results: pd.DataFrame    # rankings after removing top articles
    addition_results: pd.DataFrame   # rankings after adding synthetic articles
    reversal_count: int              # total reversals detected
    reversal_rate: float             # fraction of pairwise comparisons reversed


def rank_reversal_analysis(
    scoring_func: Callable,
    df: pd.DataFrame,
    weights: Dict[str, float],
    n_removals: int = 5,
    n_additions: int = 3,
    top_k_monitor: int = 10,
    seed: int = 42,
) -> RankReversalResult:
    """
    TEST 8: Check if removing top-ranked or adding new (synthetic) articles
    reverses rankings among remaining publications.

    Based on Nabavi, Wang & Rangaiah (2023) -- Removal of Alternatives (RA).
    """
    rng = np.random.RandomState(seed)
    ref_scores = scoring_func(df, weights)
    ref_ranks = _rank_array(ref_scores)
    sorted_idx = np.argsort(-ref_scores.values)
    n = len(df)

    # --- Removal analysis ---
    removal_rows = []
    for n_rem in range(1, min(n_removals + 1, n)):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[sorted_idx[:n_rem]] = False
        sub_df = df.iloc[keep_mask].reset_index(drop=True)
        sub_scores = scoring_func(sub_df, weights)
        sub_ranks = _rank_array(sub_scores)

        # Compare pairwise ordering among remaining items
        original_remaining_idx = np.where(keep_mask)[0]
        orig_ranks_remaining = ref_ranks[original_remaining_idx]

        n_check = min(top_k_monitor, len(sub_ranks))
        n_pairs = 0
        n_reversals = 0
        for a in range(n_check):
            for b in range(a + 1, n_check):
                n_pairs += 1
                orig_a_better = orig_ranks_remaining[a] < orig_ranks_remaining[b]
                new_a_better = sub_ranks[a] < sub_ranks[b]
                if orig_a_better != new_a_better:
                    n_reversals += 1

        removal_rows.append({
            'n_removed': n_rem,
            'n_remaining': int(keep_mask.sum()),
            'n_pairs_checked': n_pairs,
            'n_reversals': n_reversals,
            'reversal_rate': n_reversals / n_pairs if n_pairs > 0 else 0.0,
            'spearman_remaining': spearman_rho(
                sp_stats.rankdata(ref_scores.values[keep_mask]),
                sp_stats.rankdata(sub_scores.values),
            ),
        })

    # --- Addition analysis ---
    addition_rows = []
    for n_add in range(1, n_additions + 1):
        add_idx = rng.choice(n, size=n_add, replace=True)
        new_rows = df.iloc[add_idx].copy().reset_index(drop=True)
        extended_df = pd.concat([df, new_rows], ignore_index=True)
        ext_scores = scoring_func(extended_df, weights)
        orig_sub = ext_scores.values[:n]
        orig_sub_ranks = _rank_array(pd.Series(orig_sub))

        n_check = min(top_k_monitor, n)
        n_pairs = 0
        n_reversals = 0
        for a in range(n_check):
            for b in range(a + 1, n_check):
                n_pairs += 1
                if (ref_ranks[a] < ref_ranks[b]) != (orig_sub_ranks[a] < orig_sub_ranks[b]):
                    n_reversals += 1

        addition_rows.append({
            'n_added': n_add,
            'n_total': len(extended_df),
            'n_pairs_checked': n_pairs,
            'n_reversals': n_reversals,
            'reversal_rate': n_reversals / n_pairs if n_pairs > 0 else 0.0,
        })

    rem_df = pd.DataFrame(removal_rows)
    add_df = pd.DataFrame(addition_rows)
    total_reversals = int(rem_df['n_reversals'].sum() + add_df['n_reversals'].sum())
    total_pairs = int(rem_df['n_pairs_checked'].sum() + add_df['n_pairs_checked'].sum())

    return RankReversalResult(
        removal_results=rem_df,
        addition_results=add_df,
        reversal_count=total_reversals,
        reversal_rate=total_reversals / total_pairs if total_pairs > 0 else 0.0,
    )


# =========================================================================
#   TEST 9 -- Normalization Comparison
# =========================================================================

def _normalize_minmax(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)

def _normalize_max(x):
    mx = np.nanmax(np.abs(x))
    return np.zeros_like(x) if mx < 1e-12 else x / mx

def _normalize_sum(x):
    s = np.nansum(x)
    return np.zeros_like(x) if abs(s) < 1e-12 else x / s

def _normalize_vector(x):
    norm = np.sqrt(np.nansum(x ** 2))
    return np.zeros_like(x) if norm < 1e-12 else x / norm

def _normalize_zscore_arr(x):
    mu, sd = np.nanmean(x), np.nanstd(x, ddof=0)
    return np.zeros_like(x) if sd < 1e-12 else (x - mu) / sd


NORMALIZATIONS = {
    'minmax': _normalize_minmax,
    'max': _normalize_max,
    'sum': _normalize_sum,
    'vector': _normalize_vector,
    'zscore': _normalize_zscore_arr,
}


def normalization_comparison(
    decision_matrix: pd.DataFrame,
    weights: Dict[str, float],
    higher_is_better: Optional[Dict[str, bool]] = None,
    norm_methods: Optional[List[str]] = None,
    top_k_values: Sequence[int] = (5, 10, 20),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TEST 9: Apply different normalization techniques to the same decision
    matrix and compare resulting rankings.

    Category 5 in Wieckowski & Salabun (2023) taxonomy.
    """
    if norm_methods is None:
        norm_methods = list(NORMALIZATIONS.keys())
    if higher_is_better is None:
        higher_is_better = {c: True for c in weights}

    criteria = list(weights.keys())
    w_arr = np.array([weights[c] for c in criteria])
    w_arr = w_arr / w_arr.sum()

    rankings_by_norm = {}
    for nm in norm_methods:
        norm_fn = NORMALIZATIONS[nm]
        normalized = pd.DataFrame(index=decision_matrix.index)
        for c in criteria:
            raw = decision_matrix[c].values.astype(float)
            if not higher_is_better.get(c, True):
                raw = -raw
            normalized[c] = norm_fn(raw)
        scores = (normalized[criteria].values * w_arr).sum(axis=1)
        rankings_by_norm[nm] = pd.Series(scores, index=decision_matrix.index)

    method_names = list(rankings_by_norm.keys())
    rank_arrays = {nm: _rank_array(s) for nm, s in rankings_by_norm.items()}
    rows = []
    for i, n1 in enumerate(method_names):
        for j, n2 in enumerate(method_names):
            if i >= j:
                continue
            row = {'norm_1': n1, 'norm_2': n2,
                   **_all_correlations(rank_arrays[n1], rank_arrays[n2]),
                   **_overlap_at_k(rankings_by_norm[n1], rankings_by_norm[n2], top_k_values)}
            rows.append(row)

    scores_df = pd.DataFrame({nm: s for nm, s in rankings_by_norm.items()})
    return pd.DataFrame(rows), scores_df


# =========================================================================
#   TEST 10 -- Compromise Ranking (Borda + Copeland)
# =========================================================================

@dataclass
class CompromiseRankingResult:
    borda_scores: pd.Series
    copeland_scores: pd.Series
    borda_ranking: np.ndarray
    copeland_ranking: np.ndarray
    agreement_with_methods: pd.DataFrame


def compromise_ranking(
    rankings: Dict[str, pd.Series],
    top_k_values: Sequence[int] = (5, 10, 20),
) -> CompromiseRankingResult:
    """
    TEST 10: Aggregate multiple MCDA method rankings into consensus
    via Borda count and Copeland's method.
    """
    methods = list(rankings.keys())
    n = len(next(iter(rankings.values())))
    rank_arrays = {m: _rank_array(s) for m, s in rankings.items()}

    # Borda: sum of (N - rank + 1)
    borda = np.zeros(n)
    for m in methods:
        borda += (n - rank_arrays[m] + 1)
    borda_series = pd.Series(borda)

    # Copeland: pairwise wins
    copeland = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            wins_i = sum(1 for m in methods if rank_arrays[m][i] < rank_arrays[m][j])
            wins_j = len(methods) - wins_i
            if wins_i > wins_j:
                copeland[i] += 1; copeland[j] -= 1
            elif wins_j > wins_i:
                copeland[j] += 1; copeland[i] -= 1
    copeland_series = pd.Series(copeland)

    borda_ranks = _rank_array(borda_series)
    copeland_ranks = _rank_array(copeland_series)

    rows = []
    for m in methods:
        m_ranks = rank_arrays[m]
        row = {
            'method': m,
            'borda_ws': weighted_spearman_ws(borda_ranks, m_ranks),
            'borda_spearman': spearman_rho(borda_ranks, m_ranks),
            'copeland_ws': weighted_spearman_ws(copeland_ranks, m_ranks),
            'copeland_spearman': spearman_rho(copeland_ranks, m_ranks),
        }
        for k in top_k_values:
            borda_top = _top_k_set(borda_series, k)
            copeland_top = _top_k_set(copeland_series, k)
            m_top = _top_k_set(rankings[m], k)
            row[f'borda_top{k}_overlap'] = len(borda_top & m_top) / k
            row[f'copeland_top{k}_overlap'] = len(copeland_top & m_top) / k
        rows.append(row)

    return CompromiseRankingResult(
        borda_scores=borda_series,
        copeland_scores=copeland_series,
        borda_ranking=borda_ranks,
        copeland_ranking=copeland_ranks,
        agreement_with_methods=pd.DataFrame(rows),
    )


# =========================================================================
#                       REPORT  GENERATOR
# =========================================================================

def generate_validation_report(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comprehensive text report from validation results dict.

    Expected keys (all optional):
      weight_sensitivity, criteria_removal, cross_method,
      parameter_sensitivity, precision_recall, bootstrap,
      monte_carlo, rank_reversal, normalization, compromise
    """
    lines = [
        "=" * 72,
        "  MCDA VALIDATION REPORT -- EmbedSLR 2.0",
        "  Methodology: Wieckowski & Salabun (2023); Wieckowski et al. (2025)",
        "=" * 72,
    ]

    if 'weight_sensitivity' in results:
        ws = results['weight_sensitivity']
        lines += [
            "\n-- TEST 1: Weight Sensitivity (OAT) --",
            f"  Mean WS: {ws.summary['mean_ws']:.4f}    Min WS: {ws.summary['min_ws']:.4f}",
            f"  Mean rho:  {ws.summary['mean_spearman']:.4f}    Mean tau: {ws.summary['mean_kendall']:.4f}",
            f"  Most sensitive criterion: {ws.summary['most_sensitive_criterion']}",
            f"  Most stable criterion:    {ws.summary['most_stable_criterion']}",
            "",
            "  Per-criterion breakdown:",
        ]
        grp = ws.correlations.groupby('criterion')['ws'].agg(['mean', 'min', 'std'])
        for c, r in grp.iterrows():
            lines.append(f"    {c:15s}  mean_WS={r['mean']:.4f}  min_WS={r['min']:.4f}  std={r['std']:.4f}")

    if 'criteria_removal' in results:
        cr = results['criteria_removal']
        lines += [
            "\n-- TEST 2: Criteria Removal --",
            f"  Critical criterion:  {cr.critical_criterion}",
            f"  Most resilient:      {cr.most_resilient_criterion}",
            "",
            "  Individual removal impact:",
        ]
        indiv = cr.removal_steps[cr.removal_steps['type'] == 'individual_removal']
        for _, r in indiv.iterrows():
            lines.append(f"    Remove {r['removed']:15s} -> WS={r['ws']:.4f}  rho={r['spearman']:.4f}")
        lines.append("\n  Sequential removal:")
        seq = cr.removal_steps[cr.removal_steps['type'] == 'sequential']
        for _, r in seq.iterrows():
            lines.append(f"    Step {int(r['step'])}: -{r['removed']:15s} -> WS={r['ws']:.4f}  "
                         f"remaining={int(r['n_remaining'])}")

    if 'cross_method' in results:
        cd, _ = results['cross_method']
        lines.append("\n-- TEST 3: Cross-Method Correlation --")
        for _, r in cd.iterrows():
            lines.append(f"  {r['method_1']:15s} <-> {r['method_2']:15s}  "
                         f"WS={r['ws']:.4f}  rho={r['spearman']:.4f}  tau={r['kendall']:.4f}")

    if 'parameter_sensitivity' in results:
        lines.append("\n-- TEST 4: Parameter Sensitivity --")
        for pn, pdf in results['parameter_sensitivity'].items():
            lines.append(f"  Parameter: {pn}")
            for _, r in pdf.iterrows():
                mark = " <- baseline" if r.get('is_baseline') else ""
                lines.append(f"    value={r['value']:>8}  WS={r['ws']:.4f}  rho={r['spearman']:.4f}{mark}")

    if 'precision_recall' in results:
        pr = results['precision_recall']
        lines.append("\n-- TEST 5: Precision / Recall / F1 @ K --")
        for m in pr.methods:
            md = pr.per_k[pr.per_k['method'] == m]
            parts = [f"K={int(r['k'])}:P={r['precision']:.3f}/R={r['recall']:.3f}/F1={r['f1']:.3f}"
                     for _, r in md.iterrows()]
            lines.append(f"  {m:20s}  {'  '.join(parts)}")

    if 'bootstrap' in results:
        bs = results['bootstrap']
        lines.append("\n-- TEST 6: Bootstrap Stability --")
        for _, r in bs.iterrows():
            lines.append(
                f"  Top-{int(r['top_k']):>2d}: stability={r['mean_stability']:.3f}+/-{r['std_stability']:.3f}  "
                f"rho mean={r['mean_spearman']:.3f}  95%CI=[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
            )

    if 'monte_carlo' in results:
        mc = results['monte_carlo']
        s = mc.stability_summary
        lines += [
            "\n-- TEST 7: Monte Carlo Weight Space --",
            f"  N samples: {len(mc.all_runs)}",
            f"  WS:  mean={s['mean_ws']:.4f}  std={s['std_ws']:.4f}  min={s['min_ws']:.4f}  P5={s['pct5_ws']:.4f}",
            f"  rho: mean={s['mean_spearman']:.4f}  min={s['min_spearman']:.4f}",
        ]

    if 'rank_reversal' in results:
        rr = results['rank_reversal']
        lines += [
            "\n-- TEST 8: Rank Reversal --",
            f"  Total reversals: {rr.reversal_count}  Overall rate: {rr.reversal_rate:.4f}",
            "  Removal of top alternatives:",
        ]
        for _, r in rr.removal_results.iterrows():
            lines.append(
                f"    Removed top {int(r['n_removed'])}: "
                f"reversals={int(r['n_reversals'])}/{int(r['n_pairs_checked'])} "
                f"({r['reversal_rate']:.2%})  rho_remaining={r['spearman_remaining']:.4f}"
            )
        lines.append("  Addition of alternatives:")
        for _, r in rr.addition_results.iterrows():
            lines.append(
                f"    Added {int(r['n_added'])}: "
                f"reversals={int(r['n_reversals'])}/{int(r['n_pairs_checked'])} ({r['reversal_rate']:.2%})"
            )

    if 'normalization' in results:
        norm_df, _ = results['normalization']
        lines.append("\n-- TEST 9: Normalization Comparison --")
        for _, r in norm_df.iterrows():
            lines.append(f"  {r['norm_1']:8s} <-> {r['norm_2']:8s}  WS={r['ws']:.4f}  rho={r['spearman']:.4f}")

    if 'compromise' in results:
        comp = results['compromise']
        lines.append("\n-- TEST 10: Compromise Ranking (Borda + Copeland) --")
        for _, r in comp.agreement_with_methods.iterrows():
            lines.append(
                f"  {r['method']:15s}  "
                f"Borda WS={r['borda_ws']:.4f} rho={r['borda_spearman']:.4f}  "
                f"Copeland WS={r['copeland_ws']:.4f} rho={r['copeland_spearman']:.4f}"
            )

    lines.append("\n" + "=" * 72)
    report = "\n".join(lines)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    return report

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Colab GUI
==========================================
Step 1: Run MCDA ranking (L / Z / L+) on Scopus CSV
Step 2: Validate that ranking with 10 sensitivity & robustness tests

Usage::
    !pip install -q git+https://github.com/s-matysik/EmbedSLR_v3.0.git
    from embedslr.colab_app import run
    run()
    run("/content/data.csv")   # skip upload
"""
from __future__ import annotations

import io, os, sys, time, warnings, zipfile
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import ScoringConfig
from .advanced_scoring import (
    rank_with_advanced_scoring, ScoringResult,
    parse_keywords_cell, parse_references_cell,
    frequency_map, per_item_topk_sum,
    _safe_rank_points, _zscore, _z_from_median,
)
from .mcda_validation import (
    weight_sensitivity_oat, criteria_removal_analysis,
    cross_method_correlation, parameter_sensitivity,
    precision_recall_at_k, bootstrap_stability,
    monte_carlo_weights, rank_reversal_analysis,
    normalization_comparison, compromise_ranking,
    generate_validation_report,
)

warnings.filterwarnings("ignore")

METHOD_NAMES = {"linear": "L-Scoring", "zscore": "Z-Scoring",
                "linear_plus": "L-Scoring+"}
CRITERIA = ["semantic", "keywords", "references", "citations"]
_COL = {"semantic": "_sem", "keywords": "_kw",
        "references": "_ref", "citations": "_cit"}


# =====================================================================
#  PRECOMPUTE — parse tokens once, reuse in all scorer calls
# =====================================================================

def _precompute(df, top_kw=5, top_ref=15):
    """Parse Scopus CSV → fast columns _sem/_kw/_ref/_cit + available list."""
    out = df.copy()
    avail = []

    # Semantic
    for c in ["semantic_similarity", "cosine_similarity", "similarity"]:
        if c in out.columns:
            out["_sem"] = pd.to_numeric(out[c], errors="coerce"); break
    else:
        for c in ["distance_cosine", "cosine_distance"]:
            if c in out.columns:
                out["_sem"] = 1.0 - pd.to_numeric(out[c], errors="coerce"); break
        else:
            out["_sem"] = np.nan
    if out["_sem"].notna().sum() > 0:
        out["_sem"] = out["_sem"].fillna(out["_sem"].mean())
        avail.append("semantic")
    else:
        out["_sem"] = 0.0

    # Keywords
    kw_col = next((c for c in ["Author Keywords", "Authors Keywords",
                                "Keywords", "Index Keywords"]
                   if c in out.columns), None)
    if kw_col:
        kls = [parse_keywords_cell(x) for x in out[kw_col]]
        kf = frequency_map(kls)
        out["_kw"] = pd.Series([per_item_topk_sum(k, kf, top_kw)[0] for k in kls],
                               dtype=float)
        out["_kw"] = out["_kw"].fillna(out["_kw"].mean())
        avail.append("keywords")
    else:
        out["_kw"] = 0.0

    # References
    ref_col = next((c for c in ["References", "Cited References"]
                    if c in out.columns), None)
    if ref_col:
        rls = [parse_references_cell(x) for x in out[ref_col]]
        rf = frequency_map(rls)
        out["_ref"] = pd.Series([per_item_topk_sum(r, rf, top_ref)[0] for r in rls],
                                dtype=float)
        out["_ref"] = out["_ref"].fillna(out["_ref"].mean())
        avail.append("references")
    else:
        out["_ref"] = 0.0

    # Citations
    cit_col = next((c for c in ["Cited by", "Cited By", "Times Cited", "Citations"]
                    if c in out.columns), None)
    if cit_col:
        out["_cit"] = pd.to_numeric(out[cit_col], errors="coerce").fillna(0)
        avail.append("citations")
    else:
        out["_cit"] = 0.0

    return out, avail


# =====================================================================
#  FAST SCORERS (operate on precomputed _sem/_kw/_ref/_cit)
# =====================================================================

def _l_score(prep, w):
    P = len(prep); s = sum(w.values()) or 1.0
    t = pd.Series(0.0, index=prep.index)
    for c, wt in w.items():
        if wt > 0 and c in _COL:
            t += _safe_rank_points(prep[_COL[c]], True, "average", P) * (wt/s)
    return t

def _z_score(prep, w):
    s = sum(w.values()) or 1.0
    t = pd.Series(0.0, index=prep.index)
    for c, wt in w.items():
        if wt > 0 and c in _COL:
            t += _zscore(prep[_COL[c]], True) * (wt/s)
    return t

def _lp_score(prep, w, bsz=2.0, bfz=4.0):
    P = len(prep)
    base = _l_score(prep, w)
    bonus = pd.Series(0.0, index=prep.index)
    for c in _COL:
        z = _z_from_median(prep[_COL[c]], True)
        frac = ((z - bsz) / (bfz - bsz)).clip(lower=0.0)
        frac[z >= bfz] = 1.0
        bonus += frac * float(P)
    return base + bonus.clip(upper=float(P))

_FNS = {"linear": _l_score, "zscore": _z_score, "linear_plus": _lp_score}


# =====================================================================
#  STEP 1: MCDA RANKING
# =====================================================================

def run_ranking(csv_path, method="linear_plus",
                w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                top_kw=5, top_ref=15,
                bonus_start_z=2.0, bonus_full_z=4.0,
                log_fn=None):
    """
    Step 1: Run MCDA ranking on Scopus CSV.
    Returns (ranked_df, prep, available_criteria, weights).
    """
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    _log(f"Loaded {len(df)} publications")

    _log("Parsing keywords & references...")
    prep, avail = _precompute(df, top_kw=top_kw, top_ref=top_ref)
    _log(f"Criteria found: {avail}")
    if "semantic" not in avail:
        _log("NOTE: No semantic_similarity column — criterion disabled.")

    raw_w = {"semantic": w_sem, "keywords": w_kw,
             "references": w_ref, "citations": w_cit}
    bw = {k: v for k, v in raw_w.items() if k in avail}
    s = sum(bw.values()); bw = {k: v/s for k, v in bw.items()}

    _log(f"Scoring with {METHOD_NAMES[method]}...")
    if method == "linear_plus":
        scores = _lp_score(prep, bw, bsz=bonus_start_z, bfz=bonus_full_z)
    else:
        scores = _FNS[method](prep, bw)

    # Also compute full ranking via rank_with_advanced_scoring for output
    cfg = ScoringConfig(method=method, weights=bw,
                         top_keywords=top_kw, top_references=top_ref,
                         bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)
    result = rank_with_advanced_scoring(df, cfg)
    ranked = result.df.copy()
    ranked["_fast_score"] = scores.values

    _log(f"Ranking complete. Top-10:")
    title_col = next((c for c in ["Title", "Article Title", "Document Title"]
                      if c in ranked.columns), ranked.columns[0])
    for i in range(min(10, len(ranked))):
        _log(f"  {i+1}. {str(ranked.iloc[i][title_col])[:80]}")

    return ranked, prep, avail, bw


# =====================================================================
#  STEP 2: VALIDATION  (fast scorers, tuned for 500-1000 articles)
# =====================================================================

def run_validation(prep, avail, bw, method="linear_plus",
                   n_mc=1000, n_bs=500,
                   bonus_start_z=2.0, bonus_full_z=4.0,
                   log_fn=None):
    """
    Step 2: Run 10 MCDA tests on precomputed data.
    Parameters tuned for datasets of 500-1000 publications.
    """
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    t0 = time.time()
    n = len(prep)
    criteria = list(bw.keys())

    # Build scorer
    if method == "linear_plus":
        def scorer(d, w): return _lp_score(d, w, bsz=bonus_start_z, bfz=bonus_full_z)
    else:
        scorer = _FNS[method]

    # Cross-method rankings (all 3)
    all_rankings = {}
    for m, name in METHOD_NAMES.items():
        if m == "linear_plus":
            all_rankings[name] = _lp_score(prep, bw, bsz=bonus_start_z, bfz=bonus_full_z)
        else:
            all_rankings[name] = _FNS[m](prep, bw)

    results = {}

    # T1 — Weight Sensitivity OAT
    _log("TEST  1/10: Weight Sensitivity OAT...")
    results["weight_sensitivity"] = weight_sensitivity_oat(
        lambda d, w: scorer(prep, w), prep, bw,
        perturbations=(-0.50, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.50))

    # T2 — Criteria Removal
    _log("TEST  2/10: Criteria Removal...")
    results["criteria_removal"] = criteria_removal_analysis(
        lambda d, w: scorer(prep, w), prep, bw)

    # T3 — Cross-Method Correlation
    _log("TEST  3/10: Cross-Method Correlation...")
    cm_df, ws_mat = cross_method_correlation(all_rankings)
    results["cross_method"] = (cm_df, ws_mat)

    # T4 — Parameter Sensitivity (L+ bonus_start_z)
    _log("TEST  4/10: Parameter Sensitivity...")
    def _lp_bz(d, w, bz):
        return _lp_score(prep, w, bsz=bz, bfz=bz + 2.0)
    results["parameter_sensitivity"] = {
        "bonus_start_z": parameter_sensitivity(
            _lp_bz, prep, "bonus_start_z",
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], bonus_start_z, bw)}

    # T5 — Precision / Recall
    _log("TEST  5/10: Precision / Recall / F1...")
    primary = scorer(prep, bw)
    thr = primary.quantile(0.80)
    relevant = set(primary[primary >= thr].index.tolist())
    # k_values scaled to dataset size
    max_k = min(200, n // 2)
    k_vals = tuple(k for k in (5, 10, 20, 50, 100, 200) if k <= max_k)
    _log(f"         Ground truth: {len(relevant)} articles (top 20%), K={k_vals}")
    results["precision_recall"] = precision_recall_at_k(
        all_rankings, relevant, k_values=k_vals)

    # T6 — Bootstrap Stability
    # For 500-1000 articles: 500 iterations, 80% sample, top_k up to 50
    top_k_bs = tuple(k for k in (5, 10, 20, 50) if k <= n // 5)
    _log(f"TEST  6/10: Bootstrap (n={n_bs}, top_k={top_k_bs})...")
    results["bootstrap"] = bootstrap_stability(
        scorer, prep, bw,
        n_bootstrap=n_bs, sample_frac=0.8, top_k_values=top_k_bs)

    # T7 — Monte Carlo Weights
    _log(f"TEST  7/10: Monte Carlo (n={n_mc})...")
    results["monte_carlo"] = monte_carlo_weights(
        lambda d, w: scorer(prep, w), prep, criteria,
        n_samples=n_mc, reference_weights=bw)

    # T8 — Rank Reversal
    # For 500-1000: monitor top-20, remove up to 10, add up to 5
    n_rem = min(10, n // 50)
    n_add = min(5, n // 100)
    top_k_rr = min(20, n // 25)
    _log(f"TEST  8/10: Rank Reversal (remove={n_rem}, add={n_add}, monitor top-{top_k_rr})...")
    results["rank_reversal"] = rank_reversal_analysis(
        scorer, prep, bw,
        n_removals=n_rem, n_additions=n_add, top_k_monitor=top_k_rr)

    # T9 — Normalization Comparison
    _log("TEST  9/10: Normalization Comparison...")
    dm = prep[[_COL[c] for c in criteria]].copy()
    dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results["normalization"] = (nd, ns)

    # T10 — Compromise Ranking
    _log("TEST 10/10: Compromise Ranking...")
    results["compromise"] = compromise_ranking(all_rankings)

    # Report
    report = generate_validation_report(results)
    header = (f"Primary method: {METHOD_NAMES[method]}\n"
              f"Weights: {bw}\n"
              f"Criteria: {criteria}\n"
              f"Publications: {n}\n"
              f"Params: n_mc={n_mc}, n_bs={n_bs}, "
              f"bonus_start_z={bonus_start_z}, bonus_full_z={bonus_full_z}\n\n")
    report = header + report

    # ZIP
    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    zname = METHOD_NAMES[method].replace("+", "plus")
    zip_path = f"{out_dir}/mcda_{zname}_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("VALIDATION_REPORT.txt", report)
        zf.writestr("t01_weight_sensitivity.csv",
                     results["weight_sensitivity"].correlations.to_csv(index=False))
        zf.writestr("t02_criteria_removal.csv",
                     results["criteria_removal"].removal_steps.to_csv(index=False))
        zf.writestr("t03_cross_method.csv", cm_df.to_csv(index=False))
        zf.writestr("t03_ws_matrix.csv", ws_mat.to_csv())
        for pn, pdf in results["parameter_sensitivity"].items():
            zf.writestr(f"t04_param_{pn}.csv", pdf.to_csv(index=False))
        zf.writestr("t05_precision_recall.csv",
                     results["precision_recall"].per_k.to_csv(index=False))
        zf.writestr("t06_bootstrap.csv",
                     results["bootstrap"].to_csv(index=False))
        zf.writestr("t07_monte_carlo.csv",
                     results["monte_carlo"].all_runs.to_csv(index=False))
        zf.writestr("t08_rr_removal.csv",
                     results["rank_reversal"].removal_results.to_csv(index=False))
        zf.writestr("t08_rr_addition.csv",
                     results["rank_reversal"].addition_results.to_csv(index=False))
        zf.writestr("t09_normalization.csv", nd.to_csv(index=False))
        zf.writestr("t10_compromise.csv",
                     results["compromise"].agreement_with_methods.to_csv(index=False))

    elapsed = time.time() - t0
    _log(f"All 10 tests completed in {elapsed:.1f}s")
    _log(f"ZIP: {zip_path}")
    return report, zip_path


# =====================================================================
#  COMBINED: ranking + validation (for programmatic use)
# =====================================================================

def run_mcda_validation(csv_path, method="linear_plus",
                        w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                        n_mc=1000, n_bs=500,
                        top_kw=5, top_ref=15,
                        bonus_start_z=2.0, bonus_full_z=4.0,
                        log_fn=None):
    """Run MCDA ranking then 10 validation tests. Returns (report, zip_path)."""
    ranked, prep, avail, bw = run_ranking(
        csv_path, method, w_sem, w_kw, w_ref, w_cit,
        top_kw, top_ref, bonus_start_z, bonus_full_z, log_fn)
    return run_validation(
        prep, avail, bw, method, n_mc, n_bs,
        bonus_start_z, bonus_full_z, log_fn)


# =====================================================================
#  COLAB GUI
# =====================================================================

def run(csv_path: Optional[str] = None):
    """
    Launch EmbedSLR GUI in Google Colab.
    Pass csv_path to skip upload dialog.
    """
    import ipywidgets as W
    from IPython.display import display, HTML

    # ── Get CSV ──────────────────────────────────────────────────────
    if not csv_path or not os.path.isfile(str(csv_path)):
        try:
            from google.colab import files
            print("Select your Scopus CSV file:")
            uploaded = files.upload()
            if not uploaded:
                print("No file. Use: run('/content/file.csv')")
                return
            csv_path = f"/content/{list(uploaded.keys())[0]}"
        except ImportError:
            print("Not in Colab. Use: run('/path/file.csv')")
            return

    try:
        n_total = len(pd.read_csv(csv_path, encoding="utf-8-sig"))
    except Exception as e:
        print(f"Cannot read CSV: {e}"); return

    # Quick preview of available criteria
    _prep_check, _avail_check = _precompute(
        pd.read_csv(csv_path, encoding="utf-8-sig", nrows=50))
    sem_note = ""
    if "semantic" not in _avail_check:
        sem_note = ("<br><span style='color:orange'>⚠ No semantic_similarity "
                    "column. Semantic criterion will be disabled.</span>")

    # ── Widgets ──────────────────────────────────────────────────────
    sty = {"description_width": "100px"}

    method_w = W.RadioButtons(
        options=[("L-Scoring  (rank-based weighted sum)", "linear"),
                 ("Z-Scoring  (z-standardized weighted sum)", "zscore"),
                 ("L-Scoring+ (L-Scoring + outlier bonus)", "linear_plus")],
        value="linear_plus", description="",
        layout=W.Layout(width="100%"))

    w_sem = W.FloatSlider(value=0.40, min=0.05, max=0.80, step=0.05,
                           description="semantic", readout_format=".2f", style=sty,
                           disabled="semantic" not in _avail_check)
    w_kw  = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                           description="keywords", readout_format=".2f", style=sty)
    w_ref = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                           description="references", readout_format=".2f", style=sty)
    w_cit = W.FloatSlider(value=0.10, min=0.05, max=0.80, step=0.05,
                           description="citations", readout_format=".2f", style=sty)

    mc_w  = W.IntSlider(value=1000, min=100, max=5000, step=100,
                         description="MC samples", style=sty)
    bs_w  = W.IntSlider(value=500, min=50, max=2000, step=50,
                         description="Bootstrap n", style=sty)
    bsz_w = W.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.5,
                           description="bonus_start_z", readout_format=".1f", style=sty)
    bfz_w = W.FloatSlider(value=4.0, min=2.0, max=6.0, step=0.5,
                           description="bonus_full_z", readout_format=".1f", style=sty)

    out_area = W.Output(layout=W.Layout(
        max_height="700px", overflow_y="auto",
        border="1px solid #ccc", padding="8px", width="100%"))

    btn_rank = W.Button(description="Step 1: Run MCDA Ranking",
                         button_style="primary", icon="sort-amount-desc",
                         layout=W.Layout(width="280px", height="42px"))
    btn_test = W.Button(description="Step 2: Run 10 Validation Tests",
                         button_style="success", icon="check-circle",
                         layout=W.Layout(width="280px", height="42px"),
                         disabled=True)

    # State
    state = {"prep": None, "avail": None, "bw": None}

    def _log(msg):
        with out_area: print(msg, flush=True)

    def _do_rank(b):
        out_area.clear_output(wait=True)
        btn_rank.disabled = True
        btn_rank.description = "Ranking..."
        btn_test.disabled = True
        try:
            ranked, prep, avail, bw = run_ranking(
                csv_path=csv_path,
                method=method_w.value,
                w_sem=w_sem.value, w_kw=w_kw.value,
                w_ref=w_ref.value, w_cit=w_cit.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                log_fn=_log)
            state["prep"] = prep
            state["avail"] = avail
            state["bw"] = bw

            # Save ranked CSV
            out_dir = "/content" if os.path.isdir("/content") else "/tmp"
            rank_path = f"{out_dir}/ranking_{METHOD_NAMES[method_w.value].replace('+','plus')}.csv"
            ranked.to_csv(rank_path, index=False)
            _log(f"\nRanking saved: {rank_path}")
            try:
                from google.colab import files as gf
                gf.download(rank_path)
            except: pass

            _log("\n→ Now click 'Step 2' to run validation tests on this ranking.")
            btn_test.disabled = False
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback; _log(traceback.format_exc())
        finally:
            btn_rank.disabled = False
            btn_rank.description = "Step 1: Run MCDA Ranking"

    def _do_test(b):
        if state["prep"] is None:
            _log("ERROR: Run Step 1 first."); return
        btn_test.disabled = True
        btn_test.description = "Testing..."
        with out_area: print("\n" + "="*60, flush=True)
        try:
            report, zip_path = run_validation(
                prep=state["prep"],
                avail=state["avail"],
                bw=state["bw"],
                method=method_w.value,
                n_mc=mc_w.value, n_bs=bs_w.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                log_fn=_log)
            with out_area:
                print("\n" + "="*60)
                print("VALIDATION REPORT")
                print("="*60)
                print(report, flush=True)
            try:
                from google.colab import files as gf
                gf.download(zip_path)
            except: pass
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback; _log(traceback.format_exc())
        finally:
            btn_test.disabled = False
            btn_test.description = "Step 2: Run 10 Validation Tests"

    btn_rank.on_click(_do_rank)
    btn_test.on_click(_do_test)

    # ── Render ───────────────────────────────────────────────────────
    display(HTML(f"""
    <h2>EmbedSLR v3.0</h2>
    <table style="border:none">
      <tr><td><b>File:</b></td><td><code>{os.path.basename(csv_path)}</code></td></tr>
      <tr><td><b>Publications:</b></td><td>{n_total}</td></tr>
      <tr><td><b>Criteria:</b></td><td>{', '.join(_avail_check)}</td></tr>
    </table>{sem_note}
    <hr><h4>MCDA method</h4>
    """))
    display(method_w)

    display(HTML("<h4>Criteria weights</h4><small>Auto-normalized to sum=1</small>"))
    display(w_sem, w_kw, w_ref, w_cit)

    display(HTML("<h4>Scoring parameters</h4>"))
    display(W.HBox([bsz_w, bfz_w]))

    display(HTML("<h4>Validation parameters</h4>"))
    display(W.HBox([mc_w, bs_w]))

    display(HTML("<hr>"))
    display(W.HBox([btn_rank, btn_test]))
    display(HTML("<h4>Output</h4>"))
    display(out_area)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Colab GUI
==========================================
Simple ipywidgets interface for Google Colab.

Usage::
    !pip install -q git+https://github.com/s-matysik/EmbedSLR_v3.0.git
    from embedslr.colab_app import run
    run()
"""
from __future__ import annotations

import io, json, os, time, zipfile
import itertools as it
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# lightweight internal imports
from .config import ScoringConfig
from .advanced_scoring import rank_with_advanced_scoring
from .mcda_validation import (
    weight_sensitivity_oat, criteria_removal_analysis,
    cross_method_correlation, parameter_sensitivity,
    precision_recall_at_k, bootstrap_stability,
    monte_carlo_weights, rank_reversal_analysis,
    normalization_comparison, compromise_ranking,
    generate_validation_report,
)


# =====================================================================
#  MCDA VALIDATION  (callable without GUI)
# =====================================================================

METHOD_NAMES = {"linear": "L-Scoring", "zscore": "Z-Scoring",
                "linear_plus": "L-Scoring+"}


def run_mcda_validation(csv_path, method="linear_plus",
                        w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                        n_mc=1000, n_bs=500,
                        top_kw=5, top_ref=15,
                        bonus_start_z=2.0, bonus_full_z=4.0,
                        log_fn=None):
    """
    Run 10 MCDA tests on the chosen scoring method.

    Parameters
    ----------
    method : str
        "linear" (L-Scoring), "zscore" (Z-Scoring), "linear_plus" (L-Scoring+)
    """
    def _log(msg):
        if log_fn: log_fn(msg)

    t0 = time.time()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    n = len(df)
    _log(f"Loaded {n} publications. Method: {METHOD_NAMES[method]}")

    raw_w = {'semantic': w_sem, 'keywords': w_kw,
             'references': w_ref, 'citations': w_cit}
    s = sum(raw_w.values())
    bw = {k: v/s for k, v in raw_w.items()}
    criteria = list(bw.keys())

    base_cfg = ScoringConfig(
        method=method, weights=bw,
        top_keywords=top_kw, top_references=top_ref,
        bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)

    # Build scorer wrapping rank_with_advanced_scoring
    method_col = {"linear": "score_linear", "zscore": "score_zscore",
                  "linear_plus": "score_linear_plus"}[method]

    def scorer(df_in, w):
        cfg = ScoringConfig(
            method=method, weights=w,
            top_keywords=top_kw, top_references=top_ref,
            bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)
        return rank_with_advanced_scoring(df_in, cfg).df[method_col]

    # Rankings for all 3 methods (for cross-method)
    all_rankings = {}
    for m in ["linear", "zscore", "linear_plus"]:
        cfg_m = ScoringConfig(method=m, weights=bw,
                              top_keywords=top_kw, top_references=top_ref,
                              bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)
        col = {"linear": "score_linear", "zscore": "score_zscore",
               "linear_plus": "score_linear_plus"}[m]
        all_rankings[METHOD_NAMES[m]] = rank_with_advanced_scoring(df, cfg_m).df[col]

    results = {}

    _log("TEST 1/10: Weight Sensitivity OAT...")
    results['weight_sensitivity'] = weight_sensitivity_oat(
        scorer, df, bw,
        perturbations=(-0.50,-0.30,-0.20,-0.10,0.10,0.20,0.30,0.50))

    _log("TEST 2/10: Criteria Removal...")
    results['criteria_removal'] = criteria_removal_analysis(scorer, df, bw)

    _log("TEST 3/10: Cross-Method Correlation...")
    cm_df, ws_mat = cross_method_correlation(all_rankings)
    results['cross_method'] = (cm_df, ws_mat)

    _log("TEST 4/10: Parameter Sensitivity...")
    def _lp_bz(df_in, w, bz):
        cfg = ScoringConfig(method="linear_plus", weights=w,
                            top_keywords=top_kw, top_references=top_ref,
                            bonus_start_z=bz, bonus_full_z=bz+2.0)
        return rank_with_advanced_scoring(df_in, cfg).df["score_linear_plus"]
    results['parameter_sensitivity'] = {
        'bonus_start_z': parameter_sensitivity(
            _lp_bz, df, 'bonus_start_z',
            [1.0,1.5,2.0,2.5,3.0,3.5], bonus_start_z, bw)}

    _log("TEST 5/10: Precision / Recall / F1...")
    res0 = rank_with_advanced_scoring(df, base_cfg)
    scores = res0.df[method_col]
    relevant = set(scores[scores >= scores.quantile(0.80)].index.tolist())
    _log(f"  Ground truth: {len(relevant)} articles (top 20%)")
    results['precision_recall'] = precision_recall_at_k(
        all_rankings, relevant, k_values=(5,10,20,50,100))

    _log(f"TEST 6/10: Bootstrap (n={n_bs})...")
    results['bootstrap'] = bootstrap_stability(
        scorer, df, bw, n_bootstrap=n_bs, top_k_values=(5,10,20))

    _log(f"TEST 7/10: Monte Carlo (n={n_mc})...")
    results['monte_carlo'] = monte_carlo_weights(
        scorer, df, criteria, n_samples=n_mc, reference_weights=bw)

    _log("TEST 8/10: Rank Reversal...")
    results['rank_reversal'] = rank_reversal_analysis(
        scorer, df, bw, n_removals=5, n_additions=3, top_k_monitor=20)

    _log("TEST 9/10: Normalization Comparison...")
    dm_cols = [res0.df.get(f"{c}_rank_pts", pd.Series(0.0, index=res0.df.index))
               for c in criteria]
    dm = pd.concat(dm_cols, axis=1); dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results['normalization'] = (nd, ns)

    _log("TEST 10/10: Compromise Ranking...")
    results['compromise'] = compromise_ranking(all_rankings)

    report = generate_validation_report(results)
    header = (f"Primary method: {METHOD_NAMES[method]}\n"
              f"Weights: {bw}\nPublications: {n}\n\n")
    report = header + report

    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    zname = METHOD_NAMES[method].replace("+","plus")
    zip_path = f"{out_dir}/mcda_{zname}_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("VALIDATION_REPORT.txt", report)
        zf.writestr("test1_weight_sensitivity.csv",
                     results['weight_sensitivity'].correlations.to_csv(index=False))
        zf.writestr("test2_criteria_removal.csv",
                     results['criteria_removal'].removal_steps.to_csv(index=False))
        zf.writestr("test3_cross_method.csv", cm_df.to_csv(index=False))
        zf.writestr("test3_ws_matrix.csv", ws_mat.to_csv())
        for pn, pdf in results['parameter_sensitivity'].items():
            zf.writestr(f"test4_param_{pn}.csv", pdf.to_csv(index=False))
        zf.writestr("test5_precision_recall.csv",
                     results['precision_recall'].per_k.to_csv(index=False))
        zf.writestr("test6_bootstrap.csv",
                     results['bootstrap'].to_csv(index=False))
        zf.writestr("test7_monte_carlo.csv",
                     results['monte_carlo'].all_runs.to_csv(index=False))
        zf.writestr("test8_rr_removal.csv",
                     results['rank_reversal'].removal_results.to_csv(index=False))
        zf.writestr("test8_rr_addition.csv",
                     results['rank_reversal'].addition_results.to_csv(index=False))
        zf.writestr("test9_normalization.csv", nd.to_csv(index=False))
        zf.writestr("test10_compromise.csv",
                     results['compromise'].agreement_with_methods.to_csv(index=False))

    elapsed = time.time() - t0
    _log(f"Done. {elapsed:.1f}s. ZIP: {zip_path}")
    return report, zip_path


# =====================================================================
#  COLAB GUI
# =====================================================================

def run():
    """Launch EmbedSLR GUI in Google Colab."""
    import ipywidgets as W
    from IPython.display import display, clear_output, HTML

    # ── state ────────────────────────────────────────────────────────
    state = {"csv_path": None}

    # ── output areas ─────────────────────────────────────────────────
    log_out = W.Output(layout=W.Layout(
        max_height="250px", overflow_y="auto",
        border="1px solid #ccc", padding="4px", width="100%"))
    report_out = W.Output(layout=W.Layout(
        max_height="500px", overflow_y="auto",
        border="1px solid #ddd", padding="4px", width="100%"))

    def _log(msg):
        with log_out: print(msg)

    # ── Step 1: Upload ───────────────────────────────────────────────
    btn_upload = W.Button(description="Upload Scopus CSV",
                           button_style="warning", icon="upload",
                           layout=W.Layout(width="220px", height="36px"))
    upload_label = W.HTML(value="<i>No file uploaded yet.</i>")

    def _do_upload(b):
        try:
            from google.colab import files
            uploaded = files.upload()
            if uploaded:
                name = list(uploaded.keys())[0]
                state["csv_path"] = f"/content/{name}" if not name.startswith("/") else name
                # save if needed
                if not os.path.exists(state["csv_path"]):
                    with open(state["csv_path"], "wb") as f:
                        f.write(uploaded[name])
                n = len(pd.read_csv(state["csv_path"], nrows=5, encoding="utf-8-sig"))
                total = len(pd.read_csv(state["csv_path"], encoding="utf-8-sig"))
                upload_label.value = (
                    f"<b style='color:green'>Uploaded: {name} "
                    f"({total} rows)</b>")
                _log(f"CSV loaded: {name} ({total} publications)")
        except ImportError:
            _log("google.colab not available. Set state['csv_path'] manually.")
        except Exception as e:
            _log(f"Upload error: {e}")

    btn_upload.on_click(_do_upload)

    # ── Step 2: MCDA Settings ────────────────────────────────────────
    method_w = W.RadioButtons(
        options=[("L-Scoring (rank-based weighted sum)", "linear"),
                 ("Z-Scoring (z-standardized weighted sum)", "zscore"),
                 ("L-Scoring+ (L-Scoring + outlier bonus)", "linear_plus")],
        value="linear_plus",
        description="Method:",
        style={"description_width": "70px"},
        layout=W.Layout(width="100%"))

    w_sem = W.FloatSlider(value=0.40, min=0.05, max=0.80, step=0.05,
                           description="semantic", readout_format=".2f",
                           style={"description_width": "80px"})
    w_kw = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                          description="keywords", readout_format=".2f",
                          style={"description_width": "80px"})
    w_ref = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                           description="references", readout_format=".2f",
                           style={"description_width": "80px"})
    w_cit = W.FloatSlider(value=0.10, min=0.05, max=0.80, step=0.05,
                           description="citations", readout_format=".2f",
                           style={"description_width": "80px"})

    mc_w = W.IntSlider(value=1000, min=100, max=5000, step=100,
                        description="MC samples", style={"description_width": "90px"})
    bs_w = W.IntSlider(value=500, min=50, max=2000, step=50,
                        description="Bootstrap n", style={"description_width": "90px"})
    bsz_w = W.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.5,
                           description="bonus_start_z", readout_format=".1f",
                           style={"description_width": "100px"})
    bfz_w = W.FloatSlider(value=4.0, min=2.0, max=6.0, step=0.5,
                           description="bonus_full_z", readout_format=".1f",
                           style={"description_width": "100px"})

    # ── Step 3: Run ──────────────────────────────────────────────────
    btn_run = W.Button(description="Run 10 MCDA Tests",
                        button_style="success", icon="play",
                        layout=W.Layout(width="220px", height="40px"))

    def _do_run(b):
        log_out.clear_output()
        report_out.clear_output()

        if not state["csv_path"]:
            _log("ERROR: Upload a CSV first.")
            return

        btn_run.disabled = True
        btn_run.description = "Running..."
        try:
            report, zip_path = run_mcda_validation(
                csv_path=state["csv_path"],
                method=method_w.value,
                w_sem=w_sem.value, w_kw=w_kw.value,
                w_ref=w_ref.value, w_cit=w_cit.value,
                n_mc=mc_w.value, n_bs=bs_w.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                log_fn=_log)
            with report_out:
                print(report)
            try:
                from google.colab import files as gfiles
                gfiles.download(zip_path)
            except:
                _log("ZIP saved locally (auto-download unavailable).")
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback
            _log(traceback.format_exc())
        finally:
            btn_run.disabled = False
            btn_run.description = "Run 10 MCDA Tests"

    btn_run.on_click(_do_run)

    # ── Layout ───────────────────────────────────────────────────────
    display(HTML("""
    <h2>EmbedSLR v3.0 — MCDA Sensitivity & Robustness</h2>
    <p>10 tests evaluating multi-criteria ranking stability.<br>
    <i>Methodology: Więckowski & Sałabun (2023), Więckowski et al. (2025)</i></p>
    <hr>
    """))

    display(HTML("<h4>Step 1: Upload data</h4>"))
    display(W.HBox([btn_upload, upload_label]))

    display(HTML("<h4>Step 2: Select MCDA method</h4>"))
    display(method_w)

    display(HTML("<h4>Step 3: Criteria weights</h4>"
                 "<p><i>Auto-normalized to sum=1.</i></p>"))
    display(W.VBox([w_sem, w_kw, w_ref, w_cit]))

    display(HTML("<h4>Step 4: Parameters</h4>"))
    display(W.HBox([mc_w, bs_w]))
    display(W.HBox([bsz_w, bfz_w]))

    display(HTML("<h4>Step 5: Run</h4>"))
    display(btn_run)

    display(HTML("<h4>Log</h4>"))
    display(log_out)

    display(HTML("<h4>Validation Report</h4>"))
    display(report_out)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Colab GUI (ipywidgets)
======================================================
Native Colab GUI using ipywidgets. No Gradio dependency.

Tab 1: Multi-Embedding Wizard   (consensus ranking)
Tab 2: MCDA Sensitivity & Robustness  (10 tests, selectable method)

Usage:
    from embedslr.colab_app import run
    run()
"""
from __future__ import annotations

import io, json, os, sys, time, textwrap, zipfile
import itertools as it
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# ── lightweight internal imports ─────────────────────────────────────
from .config import ScoringConfig, ColumnMap
from .advanced_scoring import (
    rank_with_advanced_scoring, ScoringResult,
    parse_keywords_cell, parse_references_cell,
    frequency_map, per_item_topk_sum,
)
from .mcda_validation import (
    weight_sensitivity_oat, criteria_removal_analysis,
    cross_method_correlation, parameter_sensitivity,
    precision_recall_at_k, bootstrap_stability,
    monte_carlo_weights, rank_reversal_analysis,
    normalization_comparison, compromise_ranking,
    generate_validation_report,
)


# =====================================================================
#  MODEL CATALOG  (Tab 1)
# =====================================================================
MODEL_CATALOG: Dict[str, str] = {
    "SBERT - all-MiniLM-L12-v2":       "sbert:sentence-transformers/all-MiniLM-L12-v2",
    "SBERT - all-mpnet-base-v2":        "sbert:sentence-transformers/all-mpnet-base-v2",
    "SBERT - all-distilroberta-v1":     "sbert:sentence-transformers/all-distilroberta-v1",
    "OpenAI - text-embedding-3-large":  "openai:text-embedding-3-large",
    "OpenAI - text-embedding-ada-002":  "openai:text-embedding-ada-002",
    "Nomic - nomic-embed-text-v1.5":    "nomic:nomic-embed-text-v1.5",
    "Jina - jina-embeddings-v3":        "jina:jina-embeddings-v3",
    "Cohere - embed-english-v3.0":      "cohere:embed-english-v3.0",
}
RECOMMENDED_DEFAULTS = [
    "SBERT - all-MiniLM-L12-v2",
    "SBERT - all-mpnet-base-v2",
    "OpenAI - text-embedding-ada-002",
    "Nomic - nomic-embed-text-v1.5",
]


# =====================================================================
#  Tab 1 — Multi-Embedding Wizard  (logic only, no UI)
# =====================================================================
def _set_keys(ok, ck, nk):
    if ok: os.environ["OPENAI_API_KEY"] = ok.strip()
    if ck: os.environ["COHERE_API_KEY"] = ck.strip()
    if nk: os.environ["NOMIC_API_KEY"]  = nk.strip()


def run_wizard(csv_path, query, model_labels, sizes, top_k, agg,
               ok="", ck="", nk="", log_fn=None):
    from .io import autodetect_columns, combine_title_abstract
    from .ensemble import ModelSpec, run_ensemble, per_group_bibliometrics, build_embeddings_cache
    from .bibliometrics import full_report

    def _log(msg):
        if log_fn: log_fn(msg)

    t0 = time.time()
    _set_keys(ok, ck, nk)

    df = pd.read_csv(csv_path)
    title_col, abs_col = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)

    models = []
    for l in model_labels:
        prov, mid = MODEL_CATALOG[l].split(":", 1)
        models.append(ModelSpec(prov, mid))

    sizes_int = sorted({int(s) for s in sizes})
    sizes_int = [k for k in sizes_int if 2 <= k <= min(5, len(models))]

    _log("Computing embeddings (cached)...")
    cache = build_embeddings_cache(df, "combined_text", query, models)

    combos = []
    for k in sizes_int:
        combos.extend(list(it.combinations(models, k)))

    rows = []
    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    zip_path = f"{out_dir}/embedslr_ensemble_{int(time.time())}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps({
            "query": query, "top_k": top_k, "aggregator": agg,
            "models": [m.__dict__ for m in models], "sizes": sizes_int,
        }, indent=2, ensure_ascii=False))

        for i, combo in enumerate(combos):
            cl = list(combo)
            tag = "__".join(m.label.replace("/","_") for m in cl)
            _log(f"Combination {i+1}/{len(combos)}: {' + '.join(m.label for m in cl)}")

            ranked = run_ensemble(df, "combined_text", query, cl,
                                  top_k_per_model=top_k, aggregator=agg,
                                  precomputed=cache)
            try:    report = full_report(ranked, path=None, top_n=top_k)
            except: report = "\n".join(str(t) for t in ranked[title_col].head(5))

            vc = ranked["hit_count"].value_counts().sort_index(ascending=False)
            rows.append({"k": len(cl),
                         "combination": " + ".join(m.label for m in cl),
                         "n_records": len(ranked),
                         "hit_dist": "; ".join(f"{int(k)}:{int(v)}" for k,v in vc.items() if k>0),
                         "top3": ", ".join(str(t) for t in ranked[title_col].head(3))})

            buf = io.StringIO(); ranked.to_csv(buf, index=False)
            zf.writestr(f"ranking_k{len(cl)}_{tag}.csv", buf.getvalue())
            zf.writestr(f"report_k{len(cl)}_{tag}.txt", report)

    summary = pd.DataFrame(rows).sort_values(["k","combination"]).reset_index(drop=True)
    elapsed = time.time() - t0
    _log(f"Done. {len(combos)} combinations in {elapsed:.1f}s. ZIP: {zip_path}")
    return summary, zip_path


# =====================================================================
#  Tab 2 — MCDA Validation  (uses rank_with_advanced_scoring directly)
# =====================================================================

def _build_scorer(df: pd.DataFrame, method: str, base_cfg: ScoringConfig):
    """
    Build a scoring function that wraps rank_with_advanced_scoring.
    Returns: fn(df_subset, weights_dict) -> pd.Series of scores.

    This ensures we validate the EXACT same code that produces the ranking.
    """
    method_col = {"linear": "score_linear",
                  "zscore": "score_zscore",
                  "linear_plus": "score_linear_plus"}[method]

    def scorer(df_in, w):
        cfg = ScoringConfig(
            method=method,
            weights=w,
            top_keywords=base_cfg.top_keywords,
            top_references=base_cfg.top_references,
            penalty_no_keywords=base_cfg.penalty_no_keywords,
            penalty_no_references=base_cfg.penalty_no_references,
            bonus_start_z=base_cfg.bonus_start_z,
            bonus_full_z=base_cfg.bonus_full_z,
            columns=base_cfg.columns,
        )
        result = rank_with_advanced_scoring(df_in, cfg)
        return result.df[method_col]

    return scorer, method_col


def run_mcda_validation(csv_path, method, w_sem, w_kw, w_ref, w_cit,
                        n_mc, n_bs, top_kw=5, top_ref=15,
                        bonus_start_z=2.0, bonus_full_z=4.0,
                        log_fn=None):
    """
    Run all 10 MCDA tests on selected method.

    method: "linear" (L-Scoring), "zscore" (Z-Scoring), "linear_plus" (L-Scoring+)
    Returns: (report_text, zip_path)
    """
    def _log(msg):
        if log_fn: log_fn(msg)

    t0 = time.time()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    n = len(df)
    _log(f"Loaded {n} publications.")

    raw_w = {'semantic': w_sem, 'keywords': w_kw,
             'references': w_ref, 'citations': w_cit}
    s = sum(raw_w.values())
    bw = {k: v/s for k,v in raw_w.items()}
    criteria = list(bw.keys())

    METHOD_NAMES = {"linear": "L-Scoring", "zscore": "Z-Scoring",
                    "linear_plus": "L-Scoring+"}
    _log(f"Primary method: {METHOD_NAMES[method]}")

    base_cfg = ScoringConfig(
        method=method, weights=bw,
        top_keywords=top_kw, top_references=top_ref,
        bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z,
    )

    # Build scorer for SELECTED method (uses rank_with_advanced_scoring)
    scorer, score_col = _build_scorer(df, method, base_cfg)

    # Build scorers for ALL 3 methods (for cross-method comparison)
    all_methods = ["linear", "zscore", "linear_plus"]
    all_rankings = {}
    for m in all_methods:
        cfg_m = ScoringConfig(method=m, weights=bw,
                              top_keywords=top_kw, top_references=top_ref,
                              bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)
        res = rank_with_advanced_scoring(df, cfg_m)
        col = {"linear": "score_linear", "zscore": "score_zscore",
               "linear_plus": "score_linear_plus"}[m]
        all_rankings[METHOD_NAMES[m]] = res.df[col]

    results = {}

    # T1
    _log("TEST 1/10: Weight Sensitivity OAT...")
    results['weight_sensitivity'] = weight_sensitivity_oat(
        scorer, df, bw,
        perturbations=(-0.50,-0.30,-0.20,-0.10,0.10,0.20,0.30,0.50))

    # T2
    _log("TEST 2/10: Criteria Removal...")
    results['criteria_removal'] = criteria_removal_analysis(scorer, df, bw)

    # T3
    _log("TEST 3/10: Cross-Method Correlation...")
    cm_df, ws_mat = cross_method_correlation(all_rankings)
    results['cross_method'] = (cm_df, ws_mat)

    # T4
    _log("TEST 4/10: Parameter Sensitivity...")
    def _param_bz(df_in, w, bz):
        cfg = ScoringConfig(method="linear_plus", weights=w,
                            top_keywords=top_kw, top_references=top_ref,
                            bonus_start_z=bz, bonus_full_z=bz+2.0)
        return rank_with_advanced_scoring(df_in, cfg).df["score_linear_plus"]
    results['parameter_sensitivity'] = {
        'bonus_start_z': parameter_sensitivity(
            _param_bz, df, 'bonus_start_z',
            [1.0,1.5,2.0,2.5,3.0,3.5], bonus_start_z, bw)}

    # T5
    _log("TEST 5/10: Precision / Recall / F1...")
    # Ground truth: top 20% by combined rank across all criteria
    result_all = rank_with_advanced_scoring(df, base_cfg)
    combined_score = result_all.df[score_col]
    threshold = combined_score.quantile(0.80)
    relevant = set(combined_score[combined_score >= threshold].index.tolist())
    _log(f"  Ground truth: {len(relevant)} articles (top 20%)")
    results['precision_recall'] = precision_recall_at_k(
        all_rankings, relevant, k_values=(5,10,20,50,100))

    # T6
    _log(f"TEST 6/10: Bootstrap Stability (n={n_bs})...")
    results['bootstrap'] = bootstrap_stability(
        scorer, df, bw, n_bootstrap=n_bs, top_k_values=(5,10,20))

    # T7
    _log(f"TEST 7/10: Monte Carlo Weights (n={n_mc})...")
    results['monte_carlo'] = monte_carlo_weights(
        scorer, df, criteria, n_samples=n_mc, reference_weights=bw)

    # T8
    _log("TEST 8/10: Rank Reversal...")
    results['rank_reversal'] = rank_reversal_analysis(
        scorer, df, bw, n_removals=5, n_additions=3, top_k_monitor=20)

    # T9
    _log("TEST 9/10: Normalization Comparison...")
    # Extract raw decision matrix from scored result
    res = rank_with_advanced_scoring(df, base_cfg)
    dm_cols = []
    for c in criteria:
        col = f"{c}_rank_pts"
        if col in res.df.columns:
            dm_cols.append(res.df[col])
        else:
            dm_cols.append(pd.Series(0.0, index=res.df.index))
    dm = pd.concat(dm_cols, axis=1)
    dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results['normalization'] = (nd, ns)

    # T10
    _log("TEST 10/10: Compromise Ranking...")
    results['compromise'] = compromise_ranking(all_rankings)

    # Generate report
    report = generate_validation_report(results)
    # Prepend method info
    header = (f"Primary validated method: {METHOD_NAMES[method]}\n"
              f"Weights: {bw}\n"
              f"Publications: {n}\n\n")
    report = header + report

    # Build ZIP
    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    zip_path = f"{out_dir}/mcda_validation_{METHOD_NAMES[method].replace('+','plus')}_{int(time.time())}.zip"
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
    _log(f"Done. 10 tests in {elapsed:.1f}s. ZIP: {zip_path}")
    return report, zip_path


# =====================================================================
#  COLAB GUI  (ipywidgets — no Gradio)
# =====================================================================
def run():
    """
    Launch EmbedSLR interactive GUI in Google Colab.

    Usage::
        from embedslr.colab_app import run
        run()
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML, FileLink
    except ImportError:
        print("ipywidgets or IPython not available. Use run_mcda_validation() directly.")
        return

    # ── common widgets ───────────────────────────────────────────────
    log_output = widgets.Output(layout=widgets.Layout(
        max_height='300px', overflow_y='auto',
        border='1px solid #ccc', padding='6px'))

    def _log(msg):
        with log_output:
            print(msg)

    # ── FILE UPLOAD ──────────────────────────────────────────────────
    upload_w = widgets.FileUpload(accept='.csv', multiple=False,
                                  description='Upload CSV')
    csv_path_state = [None]

    def _on_upload(change):
        if upload_w.value:
            item = list(upload_w.value.values())[0] if isinstance(upload_w.value, dict) else upload_w.value[0]
            content = item['content'] if isinstance(item, dict) else item.content
            name = item['name'] if isinstance(item, dict) else item.name
            out_dir = "/content" if os.path.isdir("/content") else "/tmp"
            path = f"{out_dir}/{name}"
            with open(path, 'wb') as f:
                f.write(content)
            csv_path_state[0] = path
            _log(f"Uploaded: {name} -> {path}")
    upload_w.observe(_on_upload, names='value')

    # ═════════════════════════════════════════════════════════════════
    #  TAB 2 — MCDA Validation
    # ═════════════════════════════════════════════════════════════════

    method_w = widgets.ToggleButtons(
        options=[('L-Scoring (linear)', 'linear'),
                 ('Z-Scoring (zscore)', 'zscore'),
                 ('L-Scoring+ (linear+bonus)', 'linear_plus')],
        value='linear_plus',
        description='Method:',
        style={'description_width': '80px'},
        button_style='info')

    w_sem = widgets.FloatSlider(value=0.40, min=0.05, max=0.80, step=0.05,
                                 description='W semantic', readout_format='.2f')
    w_kw  = widgets.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                                 description='W keywords', readout_format='.2f')
    w_ref = widgets.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                                 description='W references', readout_format='.2f')
    w_cit = widgets.FloatSlider(value=0.10, min=0.05, max=0.80, step=0.05,
                                 description='W citations', readout_format='.2f')
    mc_w = widgets.IntSlider(value=1000, min=100, max=5000, step=100,
                              description='Monte Carlo')
    bs_w = widgets.IntSlider(value=500, min=50, max=2000, step=50,
                              description='Bootstrap')
    bsz_w = widgets.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.5,
                                 description='bonus_start_z', readout_format='.1f')
    bfz_w = widgets.FloatSlider(value=4.0, min=2.0, max=6.0, step=0.5,
                                 description='bonus_full_z', readout_format='.1f')
    topkw_w = widgets.IntSlider(value=5, min=1, max=20, step=1,
                                 description='top_keywords')
    topref_w = widgets.IntSlider(value=15, min=5, max=50, step=5,
                                  description='top_references')

    report_output = widgets.Output(layout=widgets.Layout(
        max_height='500px', overflow_y='auto',
        border='1px solid #ddd', padding='6px'))

    btn_mcda = widgets.Button(description='Run 10 MCDA Tests',
                               button_style='success',
                               icon='play',
                               layout=widgets.Layout(width='250px', height='40px'))

    def _run_mcda(b):
        log_output.clear_output()
        report_output.clear_output()

        if not csv_path_state[0]:
            _log("ERROR: Upload a CSV first.")
            return

        btn_mcda.disabled = True
        btn_mcda.description = "Running..."
        try:
            report, zip_path = run_mcda_validation(
                csv_path=csv_path_state[0],
                method=method_w.value,
                w_sem=w_sem.value, w_kw=w_kw.value,
                w_ref=w_ref.value, w_cit=w_cit.value,
                n_mc=mc_w.value, n_bs=bs_w.value,
                top_kw=topkw_w.value, top_ref=topref_w.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                log_fn=_log)
            with report_output:
                print(report)
            _log(f"\nResults ZIP: {zip_path}")
            try:
                from google.colab import files
                files.download(zip_path)
            except:
                _log("(Auto-download not available. File saved locally.)")
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback
            _log(traceback.format_exc())
        finally:
            btn_mcda.disabled = False
            btn_mcda.description = "Run 10 MCDA Tests"

    btn_mcda.on_click(_run_mcda)

    mcda_panel = widgets.VBox([
        widgets.HTML("<h3>MCDA Sensitivity & Robustness Analysis</h3>"
                     "<p>Select which scoring method to validate, set weights and parameters.</p>"),
        method_w,
        widgets.HTML("<b>Criteria weights</b> (auto-normalized):"),
        widgets.HBox([w_sem, w_kw]),
        widgets.HBox([w_ref, w_cit]),
        widgets.HTML("<b>Scoring parameters:</b>"),
        widgets.HBox([topkw_w, topref_w]),
        widgets.HBox([bsz_w, bfz_w]),
        widgets.HTML("<b>Validation parameters:</b>"),
        widgets.HBox([mc_w, bs_w]),
        btn_mcda,
        widgets.HTML("<b>Report:</b>"),
        report_output,
    ])

    # ═════════════════════════════════════════════════════════════════
    #  TAB 1 — Multi-Embedding Wizard
    # ═════════════════════════════════════════════════════════════════

    query_w = widgets.Textarea(
        placeholder='e.g. Does blockchain affect customer loyalty?',
        description='Query:', layout=widgets.Layout(width='95%', height='60px'))
    models_w = widgets.SelectMultiple(
        options=list(MODEL_CATALOG.keys()),
        value=RECOMMENDED_DEFAULTS,
        description='Models:',
        layout=widgets.Layout(height='160px', width='95%'))
    sizes_w = widgets.SelectMultiple(
        options=['2','3','4','5'], value=['2','3','4'],
        description='Sizes:', layout=widgets.Layout(height='100px'))
    topk_w = widgets.IntSlider(value=50, min=10, max=200, step=1,
                                description='top-K')
    agg_w = widgets.ToggleButtons(options=['mean','min','median'], value='mean',
                                   description='Aggregation:')
    oai_w = widgets.Password(description='OpenAI key:',
                              layout=widgets.Layout(width='400px'))
    coh_w = widgets.Password(description='Cohere key:',
                              layout=widgets.Layout(width='400px'))
    nom_w = widgets.Password(description='Nomic key:',
                              layout=widgets.Layout(width='400px'))

    wizard_output = widgets.Output(layout=widgets.Layout(
        max_height='400px', overflow_y='auto',
        border='1px solid #ddd', padding='6px'))

    btn_wizard = widgets.Button(description='Run Wizard',
                                 button_style='primary', icon='play',
                                 layout=widgets.Layout(width='200px', height='40px'))

    def _run_wizard(b):
        log_output.clear_output()
        wizard_output.clear_output()

        if not csv_path_state[0]:
            _log("ERROR: Upload a CSV first.")
            return
        if not query_w.value.strip():
            _log("ERROR: Enter a research query.")
            return

        btn_wizard.disabled = True
        btn_wizard.description = "Running..."
        try:
            summary, zip_path = run_wizard(
                csv_path=csv_path_state[0],
                query=query_w.value.strip(),
                model_labels=list(models_w.value),
                sizes=list(sizes_w.value),
                top_k=topk_w.value,
                agg=agg_w.value,
                ok=oai_w.value, ck=coh_w.value, nk=nom_w.value,
                log_fn=_log)
            with wizard_output:
                display(summary)
            _log(f"\nResults ZIP: {zip_path}")
            try:
                from google.colab import files
                files.download(zip_path)
            except:
                _log("(Auto-download not available.)")
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback
            _log(traceback.format_exc())
        finally:
            btn_wizard.disabled = False
            btn_wizard.description = "Run Wizard"

    btn_wizard.on_click(_run_wizard)

    wizard_panel = widgets.VBox([
        widgets.HTML("<h3>Multi-Embedding Wizard</h3>"),
        query_w,
        widgets.HBox([models_w, sizes_w]),
        widgets.HBox([topk_w, agg_w]),
        widgets.HTML("<b>API keys</b> (only for cloud models):"),
        oai_w, coh_w, nom_w,
        btn_wizard,
        widgets.HTML("<b>Results:</b>"),
        wizard_output,
    ])

    # ═════════════════════════════════════════════════════════════════
    #  TABS
    # ═════════════════════════════════════════════════════════════════
    tabs = widgets.Tab(children=[wizard_panel, mcda_panel])
    tabs.set_title(0, 'Embedding Wizard')
    tabs.set_title(1, 'MCDA Validation')

    header = widgets.HTML(
        "<h2>EmbedSLR v3.0</h2>"
        "<p>Upload your Scopus CSV, then use either tab.</p>")

    display(header, upload_w, tabs,
            widgets.HTML("<b>Log:</b>"), log_output)


if __name__ == "__main__":
    run()

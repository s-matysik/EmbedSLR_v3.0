#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Colab GUI
==========================================
Step 1: MCDA ranking (L / Z / L+)
Step 2: 10 validation tests + plots

Usage::
    !pip install -q git+https://github.com/s-matysik/EmbedSLR_v3.0.git
    from embedslr.colab_app import run
    run()
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
#  PRECOMPUTE
# =====================================================================

def _precompute(df, top_kw=5, top_ref=15):
    """Parse once → _sem/_kw/_ref/_cit + available criteria list."""
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
                               index=out.index, dtype=float)
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
                                index=out.index, dtype=float)
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
#  FAST SCORERS
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
    P = len(prep); base = _l_score(prep, w)
    bonus = pd.Series(0.0, index=prep.index)
    for c in w:
        if c in _COL:
            z = _z_from_median(prep[_COL[c]], True)
            frac = ((z - bsz) / (bfz - bsz)).clip(lower=0.0)
            frac[z >= bfz] = 1.0
            bonus += frac * float(P)
    return base + bonus.clip(upper=float(P))

_FNS = {"linear": _l_score, "zscore": _z_score, "linear_plus": _lp_score}


# =====================================================================
#  PLOTS
# =====================================================================

def _generate_plots(results, method_name, out_dir):
    """Generate 4 matplotlib plots, save as PNGs. Returns list of paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MCDA Validation — {method_name}", fontsize=14, fontweight="bold")

    # 1) Weight Sensitivity per criterion
    ax = axes[0, 0]
    ws_df = results["weight_sensitivity"].correlations
    for crit in ws_df["criterion"].unique():
        sub = ws_df[ws_df["criterion"] == crit]
        ax.plot(sub["perturbation"], sub["ws"], "o-", label=crit, markersize=4)
    ax.set_xlabel("Perturbation (δ)")
    ax.set_ylabel("WS coefficient")
    ax.set_title("T1: Weight Sensitivity (OAT)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(0, ws_df["ws"].min() - 0.05), 1.01)

    # 2) Criteria Removal
    ax = axes[0, 1]
    cr = results["criteria_removal"].removal_steps
    ind = cr[cr["type"] == "individual_removal"]
    if len(ind) > 0:
        ax.barh(ind["removed"], ind["ws"], color="#e74c3c", alpha=0.8)
        ax.set_xlabel("WS vs baseline")
        ax.set_title("T2: Criteria Removal Impact")
        ax.set_xlim(max(0, float(ind["ws"].min()) - 0.1), 1.01)
    ax.grid(True, alpha=0.3, axis="x")

    # 3) Monte Carlo WS distribution
    ax = axes[1, 0]
    mc = results["monte_carlo"].all_runs
    if "ws" in mc.columns:
        ax.hist(mc["ws"], bins=40, color="#3498db", alpha=0.8, edgecolor="white")
        ax.axvline(mc["ws"].mean(), color="red", linestyle="--", label=f'mean={mc["ws"].mean():.4f}')
        ax.axvline(mc["ws"].quantile(0.05), color="orange", linestyle=":",
                   label=f'P5={mc["ws"].quantile(0.05):.4f}')
        ax.set_xlabel("Weighted Spearman (WS)")
        ax.set_ylabel("Count")
        ax.set_title("T7: Monte Carlo Weight Space")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Precision @ K
    ax = axes[1, 1]
    pr = results["precision_recall"].per_k
    for meth in pr["method"].unique():
        sub = pr[pr["method"] == meth]
        style = "--" if meth == "random_baseline" else "-"
        ax.plot(sub["k"], sub["precision"], f"o{style}", label=meth, markersize=4)
    ax.set_xlabel("K")
    ax.set_ylabel("Precision")
    ax.set_title("T5: Precision @ K")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = f"{out_dir}/validation_plots.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 5) Bootstrap stability bar chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bs = results["bootstrap"]
    ax2.bar([f"Top-{int(r['top_k'])}" for _, r in bs.iterrows()],
            bs["mean_stability"], yerr=bs["std_stability"],
            color="#2ecc71", alpha=0.8, capsize=5)
    ax2.set_ylabel("Stability (frequency)")
    ax2.set_title(f"T6: Bootstrap Stability ({method_name})")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p2 = f"{out_dir}/bootstrap_stability.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    paths.append(p2)

    return paths


# =====================================================================
#  STEP 1: RANKING
# =====================================================================

def run_ranking(csv_path, method="linear_plus",
                w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                top_kw=5, top_ref=15,
                bonus_start_z=2.0, bonus_full_z=4.0,
                log_fn=None):
    """Step 1: MCDA ranking. Returns (ranked_df, prep, avail, weights)."""
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    _log(f"Loaded {len(df)} publications")

    _log("Parsing keywords & references...")
    prep, avail = _precompute(df, top_kw=top_kw, top_ref=top_ref)
    _log(f"Criteria found: {avail}")
    if "semantic" not in avail:
        _log("⚠ No semantic_similarity / distance_cosine column — disabled.")

    raw_w = {"semantic": w_sem, "keywords": w_kw,
             "references": w_ref, "citations": w_cit}
    bw = {k: v for k, v in raw_w.items() if k in avail}
    s = sum(bw.values()); bw = {k: v/s for k, v in bw.items()}
    _log(f"Weights: {', '.join(f'{k}={v:.3f}' for k,v in bw.items())}")

    _log(f"Scoring: {METHOD_NAMES[method]}...")
    if method == "linear_plus":
        scores = _lp_score(prep, bw, bsz=bonus_start_z, bfz=bonus_full_z)
    else:
        scores = _FNS[method](prep, bw)

    # Full ranking via rank_with_advanced_scoring for CSV output
    cfg = ScoringConfig(method=method, weights=bw,
                         top_keywords=top_kw, top_references=top_ref,
                         bonus_start_z=bonus_start_z, bonus_full_z=bonus_full_z)
    result = rank_with_advanced_scoring(df, cfg)
    ranked = result.df

    title_col = next((c for c in ["Title", "Article Title", "Document Title"]
                      if c in ranked.columns), ranked.columns[0])
    _log("Top-10:")
    for i in range(min(10, len(ranked))):
        _log(f"  {i+1}. {str(ranked.iloc[i][title_col])[:80]}")

    return ranked, prep, avail, bw


# =====================================================================
#  STEP 2: VALIDATION
# =====================================================================

def run_validation(prep, avail, bw, method="linear_plus",
                   n_mc=1000, n_bs=500,
                   bonus_start_z=2.0, bonus_full_z=4.0,
                   log_fn=None):
    """Step 2: 10 tests + plots + ZIP. Returns (report, zip_path)."""
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    t0 = time.time()
    n = len(prep)
    criteria = list(bw.keys())

    if method == "linear_plus":
        def scorer(d, w): return _lp_score(d, w, bsz=bonus_start_z, bfz=bonus_full_z)
    else:
        scorer = _FNS[method]

    # All 3 rankings for cross-method
    all_rankings = {}
    for m, name in METHOD_NAMES.items():
        if m == "linear_plus":
            all_rankings[name] = _lp_score(prep, bw, bsz=bonus_start_z, bfz=bonus_full_z)
        else:
            all_rankings[name] = _FNS[m](prep, bw)

    results = {}

    _log("TEST  1/10: Weight Sensitivity OAT...")
    results["weight_sensitivity"] = weight_sensitivity_oat(
        lambda d, w: scorer(prep, w), prep, bw,
        perturbations=(-0.50, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.50))

    _log("TEST  2/10: Criteria Removal...")
    results["criteria_removal"] = criteria_removal_analysis(
        lambda d, w: scorer(prep, w), prep, bw)

    _log("TEST  3/10: Cross-Method Correlation...")
    cm_df, ws_mat = cross_method_correlation(all_rankings)
    results["cross_method"] = (cm_df, ws_mat)

    _log("TEST  4/10: Parameter Sensitivity...")
    def _lp_bz(d, w, bz):
        return _lp_score(prep, w, bsz=bz, bfz=bz + 2.0)
    results["parameter_sensitivity"] = {
        "bonus_start_z": parameter_sensitivity(
            _lp_bz, prep, "bonus_start_z",
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], bonus_start_z, bw)}

    _log("TEST  5/10: Precision / Recall / F1...")
    # Ground truth = consensus: article in top-20% of >= 2 out of 3 methods
    top_pct = 0.20
    votes = pd.Series(0, index=prep.index)
    for name, scores in all_rankings.items():
        thr = scores.quantile(1.0 - top_pct)
        votes += (scores >= thr).astype(int)
    relevant = set(votes[votes >= 2].index.tolist())
    max_k = min(200, n // 2)
    k_vals = tuple(k for k in (5, 10, 20, 50, 100, 200) if k <= max_k)
    _log(f"         Ground truth: {len(relevant)} articles (top 20%, consensus ≥2/3 methods), K={k_vals}")
    results["precision_recall"] = precision_recall_at_k(
        all_rankings, relevant, k_values=k_vals)

    top_k_bs = tuple(k for k in (5, 10, 20, 50) if k <= n // 5)
    _log(f"TEST  6/10: Bootstrap (n={n_bs}, top_k={top_k_bs})...")
    results["bootstrap"] = bootstrap_stability(
        scorer, prep, bw,
        n_bootstrap=n_bs, sample_frac=0.8, top_k_values=top_k_bs)

    _log(f"TEST  7/10: Monte Carlo (n={n_mc})...")
    results["monte_carlo"] = monte_carlo_weights(
        lambda d, w: scorer(prep, w), prep, criteria,
        n_samples=n_mc, reference_weights=bw)

    n_rem = min(10, max(5, n // 50))
    n_add = min(5, max(3, n // 100))
    top_k_rr = min(20, n // 25)
    _log(f"TEST  8/10: Rank Reversal (rem={n_rem}, add={n_add}, top-{top_k_rr})...")
    results["rank_reversal"] = rank_reversal_analysis(
        scorer, prep, bw,
        n_removals=n_rem, n_additions=n_add, top_k_monitor=top_k_rr)

    _log("TEST  9/10: Normalization Comparison...")
    dm = prep[[_COL[c] for c in criteria]].copy()
    dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results["normalization"] = (nd, ns)

    _log("TEST 10/10: Compromise Ranking...")
    results["compromise"] = compromise_ranking(all_rankings)

    # Report
    report = generate_validation_report(results)
    header = (f"Primary method: {METHOD_NAMES[method]}\n"
              f"Weights: {bw}\nCriteria: {criteria}\n"
              f"Publications: {n}\n"
              f"Params: n_mc={n_mc}, n_bs={n_bs}, "
              f"bonus_start_z={bonus_start_z}, bonus_full_z={bonus_full_z}\n\n")
    report = header + report

    # Plots
    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    _log("Generating plots...")
    plot_paths = _generate_plots(results, METHOD_NAMES[method], out_dir)

    # Show plots in Colab
    try:
        from IPython.display import display, Image
        for p in plot_paths:
            display(Image(filename=p))
    except Exception:
        pass

    # ZIP
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
        for pp in plot_paths:
            zf.write(pp, os.path.basename(pp))

    elapsed = time.time() - t0
    _log(f"Done in {elapsed:.1f}s — ZIP: {zip_path}")

    # Auto-download in Colab
    try:
        from google.colab import files as gf
        gf.download(zip_path)
    except Exception:
        pass

    return report, zip_path


# =====================================================================
#  COMBINED (programmatic)
# =====================================================================

def run_mcda_validation(csv_path, method="linear_plus",
                        w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                        n_mc=1000, n_bs=500,
                        top_kw=5, top_ref=15,
                        bonus_start_z=2.0, bonus_full_z=4.0,
                        log_fn=None):
    """Run ranking + validation. Returns (report, zip_path)."""
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
    """Launch EmbedSLR GUI in Colab."""
    import ipywidgets as W
    from IPython.display import display, HTML

    if not csv_path or not os.path.isfile(str(csv_path)):
        try:
            from google.colab import files
            print("Select your Scopus CSV file:")
            uploaded = files.upload()
            if not uploaded:
                print("Use: run('/content/file.csv')"); return
            csv_path = f"/content/{list(uploaded.keys())[0]}"
        except ImportError:
            print("Use: run('/path/file.csv')"); return

    try:
        n_total = len(pd.read_csv(csv_path, encoding="utf-8-sig"))
    except Exception as e:
        print(f"Cannot read: {e}"); return

    _, avail_check = _precompute(pd.read_csv(csv_path, encoding="utf-8-sig", nrows=50))
    sem_note = ("" if "semantic" in avail_check else
                "<br><span style='color:orange'>⚠ No semantic column — disabled</span>")

    sty = {"description_width": "100px"}

    method_w = W.RadioButtons(
        options=[("L-Scoring  (rank-based weighted sum)", "linear"),
                 ("Z-Scoring  (z-standardized weighted sum)", "zscore"),
                 ("L-Scoring+ (L-Scoring + outlier bonus)", "linear_plus")],
        value="linear_plus", description="", layout=W.Layout(width="100%"))

    w_sem = W.FloatSlider(value=0.40, min=0.05, max=0.80, step=0.05,
                           description="semantic", readout_format=".2f", style=sty,
                           disabled="semantic" not in avail_check)
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

    btn_rank = W.Button(description="Step 1: MCDA Ranking",
                         button_style="primary", icon="sort-amount-desc",
                         layout=W.Layout(width="280px", height="42px"))
    btn_test = W.Button(description="Step 2: Validation Tests",
                         button_style="success", icon="check-circle",
                         layout=W.Layout(width="280px", height="42px"),
                         disabled=True)

    state = {"prep": None, "avail": None, "bw": None}

    def _log(msg):
        with out_area: print(msg, flush=True)

    def _on_rank(b):
        out_area.clear_output(wait=True)
        btn_rank.disabled = True; btn_test.disabled = True
        btn_rank.description = "Ranking..."
        try:
            ranked, prep, avail, bw = run_ranking(
                csv_path, method_w.value,
                w_sem.value, w_kw.value, w_ref.value, w_cit.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                log_fn=_log)
            state.update(prep=prep, avail=avail, bw=bw)

            out_dir = "/content" if os.path.isdir("/content") else "/tmp"
            mn = METHOD_NAMES[method_w.value].replace("+", "plus")
            rp = f"{out_dir}/ranking_{mn}.csv"
            ranked.to_csv(rp, index=False)
            _log(f"\nRanking CSV: {rp}")
            try:
                from google.colab import files as gf; gf.download(rp)
            except: pass
            _log("\n→ Click 'Step 2' to validate this ranking.")
            btn_test.disabled = False
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback; _log(traceback.format_exc())
        finally:
            btn_rank.disabled = False
            btn_rank.description = "Step 1: MCDA Ranking"

    def _on_test(b):
        if state["prep"] is None: _log("Run Step 1 first."); return
        btn_test.disabled = True
        btn_test.description = "Testing..."
        with out_area: print("\n" + "="*60, flush=True)
        try:
            report, zip_path = run_validation(
                state["prep"], state["avail"], state["bw"],
                method_w.value, mc_w.value, bs_w.value,
                bsz_w.value, bfz_w.value, _log)
            with out_area:
                print("\n" + "="*60)
                print("VALIDATION REPORT")
                print("="*60)
                print(report, flush=True)
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback; _log(traceback.format_exc())
        finally:
            btn_test.disabled = False
            btn_test.description = "Step 2: Validation Tests"

    btn_rank.on_click(_on_rank)
    btn_test.on_click(_on_test)

    display(HTML(f"""
    <h2>EmbedSLR v3.0</h2>
    <table><tr><td><b>File:</b></td><td><code>{os.path.basename(csv_path)}</code></td></tr>
    <tr><td><b>Publications:</b></td><td>{n_total}</td></tr>
    <tr><td><b>Criteria:</b></td><td>{', '.join(avail_check)}</td></tr></table>{sem_note}
    <hr><h4>MCDA method</h4>"""))
    display(method_w)
    display(HTML("<h4>Criteria weights</h4><small>Auto-normalized</small>"))
    display(w_sem, w_kw, w_ref, w_cit)
    display(HTML("<h4>Parameters</h4>"))
    display(W.HBox([bsz_w, bfz_w]))
    display(W.HBox([mc_w, bs_w]))
    display(HTML("<hr>"))
    display(W.HBox([btn_rank, btn_test]))
    display(HTML("<h4>Output</h4>"))
    display(out_area)


if __name__ == "__main__":
    run()

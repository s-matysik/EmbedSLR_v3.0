#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Colab GUI
==========================================
Step 1: MCDA ranking (L / Z / L+)
  - Research problem → embeddings computed on the fly (sentence-transformers)
  - OR uses pre-computed distance_cosine / semantic_similarity from CSV
Step 2: Validation tests + plots
  - T3 Cross-Method & T10 Compromise: only when ≥2 methods selected
  - T4 Parameter Sensitivity: only for L-Scoring+
  - T5 Precision@K: primary method vs random baseline
  - T5b LLM Validation (optional): multi-provider article evaluation

Usage::
    !pip install -q git+https://github.com/s-matysik/EmbedSLR_v3.0.git
    from embedslr.colab_app import run
    run()
"""
from __future__ import annotations

import io, os, sys, time, warnings, zipfile
from typing import Dict, List, Optional, Set

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
from .llm_validation import (
    run_llm_validation, format_llm_report, LLMValidationResult,
)

warnings.filterwarnings("ignore")

METHOD_NAMES = {"linear": "L-Scoring", "zscore": "Z-Scoring",
                "linear_plus": "L-Scoring+"}
CRITERIA = ["semantic", "keywords", "references", "citations"]
_COL = {"semantic": "_sem", "keywords": "_kw",
        "references": "_ref", "citations": "_cit"}


# =====================================================================
#  EMBEDDING — compute semantic similarity on the fly
# =====================================================================

def _compute_embeddings(df, research_problem, model_name="all-MiniLM-L6-v2",
                        log_fn=None):
    """Compute cosine similarity between research_problem and each article.

    Uses sentence-transformers. Returns Series of similarity scores.
    """
    def _log(msg):
        if log_fn: log_fn(msg)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _log("⚠ sentence-transformers not installed. "
             "Run: pip install sentence-transformers")
        return None

    _log(f"Loading embedding model ({model_name})...")
    model = SentenceTransformer(model_name)

    # Build article text = title + abstract
    title_col = next((c for c in ["Title", "Article Title", "Document Title"]
                      if c in df.columns), None)
    abs_col = next((c for c in ["Abstract", "Description", "abstract"]
                    if c in df.columns), None)

    texts = []
    for _, row in df.iterrows():
        t = str(row[title_col]) if title_col and pd.notna(row.get(title_col)) else ""
        a = str(row[abs_col]) if abs_col and pd.notna(row.get(abs_col)) else ""
        texts.append(f"{t} {a}".strip() or "N/A")

    _log(f"Computing embeddings for {len(texts)} articles + research problem...")
    doc_embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    q_emb = model.encode([research_problem], normalize_embeddings=True)[0]

    # Cosine similarity (already normalized → dot product)
    sims = doc_embs @ q_emb
    _log(f"Semantic similarity: min={sims.min():.3f}, max={sims.max():.3f}, "
         f"mean={sims.mean():.3f}")
    return pd.Series(sims, index=df.index, dtype=float, name="_sem")


# =====================================================================
#  PRECOMPUTE
# =====================================================================

def _precompute(df, top_kw=5, top_ref=15, research_problem=None, log_fn=None):
    """Parse once → _sem/_kw/_ref/_cit + available criteria list.

    If research_problem is given and no semantic column exists in CSV,
    embeddings are computed on the fly using sentence-transformers.
    """
    out = df.copy()
    avail = []

    # --- Semantic ---
    has_sem_col = False
    for c in ["semantic_similarity", "cosine_similarity", "similarity"]:
        if c in out.columns:
            out["_sem"] = pd.to_numeric(out[c], errors="coerce")
            has_sem_col = True; break
    if not has_sem_col:
        for c in ["distance_cosine", "cosine_distance"]:
            if c in out.columns:
                out["_sem"] = 1.0 - pd.to_numeric(out[c], errors="coerce")
                has_sem_col = True; break

    # Compute from research problem if needed
    if not has_sem_col and research_problem and research_problem.strip():
        sims = _compute_embeddings(df, research_problem, log_fn=log_fn)
        if sims is not None:
            out["_sem"] = sims
            has_sem_col = True

    if has_sem_col and "_sem" in out.columns and out["_sem"].notna().sum() > 0:
        out["_sem"] = out["_sem"].fillna(out["_sem"].mean())
        avail.append("semantic")
    else:
        out["_sem"] = 0.0

    # --- Keywords ---
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

    # --- References ---
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

    # --- Citations ---
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
    """Generate matplotlib plots, save as PNGs. Returns list of paths."""
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
        ax.axvline(mc["ws"].mean(), color="red", linestyle="--",
                   label=f'mean={mc["ws"].mean():.4f}')
        ax.axvline(mc["ws"].quantile(0.05), color="orange", linestyle=":",
                   label=f'P5={mc["ws"].quantile(0.05):.4f}')
        ax.set_xlabel("Weighted Spearman (WS)")
        ax.set_ylabel("Count")
        ax.set_title("T7: Monte Carlo Weight Space")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Precision @ K (primary method vs random)
    ax = axes[1, 1]
    pr = results["precision_recall"].per_k
    for meth in pr["method"].unique():
        style = "--" if meth == "random_baseline" else "-"
        color = "#e74c3c" if meth == "random_baseline" else "#3498db"
        ax.plot(pr[pr["method"] == meth]["k"],
                pr[pr["method"] == meth]["precision"],
                f"o{style}", label=meth, markersize=5, color=color)
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
                research_problem=None, log_fn=None):
    """Step 1: MCDA ranking. Returns (ranked_df, prep, avail, weights)."""
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    _log(f"Loaded {len(df)} publications")

    _log("Parsing keywords & references...")
    prep, avail = _precompute(df, top_kw=top_kw, top_ref=top_ref,
                              research_problem=research_problem, log_fn=log_fn)
    _log(f"Criteria found: {avail}")
    if "semantic" not in avail:
        _log("⚠ No semantic column in CSV and no research problem → semantic disabled.")

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
                   methods_to_run=None,
                   n_mc=1000, n_bs=500,
                   bonus_start_z=2.0, bonus_full_z=4.0,
                   ranked_df=None, research_problem=None,
                   api_keys=None, llm_top_k=20,
                   log_fn=None):
    """Step 2: validation tests + plots + ZIP.

    Parameters
    ----------
    methods_to_run : list of method keys, e.g. ["linear"] or all 3.
        T3/T10 run only if len >= 2.
        T4 runs only if "linear_plus" in list.
    ranked_df : optional DataFrame for LLM validation (from Step 1)
    research_problem : optional str for LLM validation
    api_keys : optional dict of LLM API keys for T5b

    Returns (report, zip_path).
    """
    def _log(msg):
        if log_fn: log_fn(msg); sys.stdout.flush()

    t0 = time.time()
    n = len(prep)
    criteria = list(bw.keys())
    primary_name = METHOD_NAMES[method]

    if methods_to_run is None:
        methods_to_run = [method]

    multi_method = len(methods_to_run) >= 2

    if method == "linear_plus":
        def scorer(d, w): return _lp_score(d, w, bsz=bonus_start_z, bfz=bonus_full_z)
    else:
        scorer = _FNS[method]

    # Primary ranking scores
    primary_scores = scorer(prep, bw)

    # All selected methods
    all_rankings = {}
    for m in methods_to_run:
        name = METHOD_NAMES[m]
        if m == "linear_plus":
            all_rankings[name] = _lp_score(prep, bw, bsz=bonus_start_z, bfz=bonus_full_z)
        else:
            all_rankings[name] = _FNS[m](prep, bw)

    results = {}

    # ---- T1: Weight Sensitivity ----
    _log("TEST  1/10: Weight Sensitivity OAT...")
    results["weight_sensitivity"] = weight_sensitivity_oat(
        lambda d, w: scorer(prep, w), prep, bw,
        perturbations=(-0.50, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.50))

    # ---- T2: Criteria Removal ----
    _log("TEST  2/10: Criteria Removal...")
    results["criteria_removal"] = criteria_removal_analysis(
        lambda d, w: scorer(prep, w), prep, bw)

    # ---- T3: Cross-Method (≥2 methods only) ----
    if multi_method:
        _log("TEST  3/10: Cross-Method Correlation...")
        cm_df, ws_mat = cross_method_correlation(all_rankings)
        results["cross_method"] = (cm_df, ws_mat)
    else:
        _log("TEST  3/10: Cross-Method Correlation... SKIPPED (single method)")

    # ---- T4: Parameter Sensitivity (L-Scoring+ only) ----
    if "linear_plus" in methods_to_run:
        _log("TEST  4/10: Parameter Sensitivity...")
        def _lp_bz(d, w, bz):
            return _lp_score(prep, w, bsz=bz, bfz=bz + 2.0)
        results["parameter_sensitivity"] = {
            "bonus_start_z": parameter_sensitivity(
                _lp_bz, prep, "bonus_start_z",
                [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], bonus_start_z, bw)}
    else:
        _log("TEST  4/10: Parameter Sensitivity... SKIPPED (not L-Scoring+)")
        results["parameter_sensitivity"] = {}

    # ---- T5: Precision / Recall / F1 — primary vs random ----
    _log("TEST  5/10: Precision / Recall / F1...")
    top_pct = 0.20
    thr = primary_scores.quantile(1.0 - top_pct)
    relevant = set(primary_scores[primary_scores >= thr].index.tolist())
    max_k = min(200, n // 2)
    k_vals = tuple(k for k in (5, 10, 20, 50, 100, 200) if k <= max_k)
    pr_rankings = {primary_name: primary_scores}
    _log(f"         Ground truth: {len(relevant)} articles "
         f"(top {int(top_pct*100)}% of {primary_name}), K={k_vals}")
    results["precision_recall"] = precision_recall_at_k(
        pr_rankings, relevant, k_values=k_vals)

    # ---- T5b: LLM Validation (optional) ----
    llm_result = None
    if api_keys and research_problem and ranked_df is not None:
        _log("TEST 5b: LLM Validation...")
        llm_result = run_llm_validation(
            ranked_df, prep, research_problem,
            api_keys=api_keys, top_k=llm_top_k, log_fn=_log)
        results["llm_validation"] = llm_result

    # ---- T6: Bootstrap ----
    top_k_bs = tuple(k for k in (5, 10, 20, 50) if k <= n // 5)
    _log(f"TEST  6/10: Bootstrap (n={n_bs}, top_k={top_k_bs})...")
    results["bootstrap"] = bootstrap_stability(
        scorer, prep, bw,
        n_bootstrap=n_bs, sample_frac=0.8, top_k_values=top_k_bs)

    # ---- T7: Monte Carlo ----
    _log(f"TEST  7/10: Monte Carlo (n={n_mc})...")
    results["monte_carlo"] = monte_carlo_weights(
        lambda d, w: scorer(prep, w), prep, criteria,
        n_samples=n_mc, reference_weights=bw)

    # ---- T8: Rank Reversal ----
    n_rem = min(10, max(5, n // 50))
    n_add = min(5, max(3, n // 100))
    top_k_rr = min(20, n // 25)
    _log(f"TEST  8/10: Rank Reversal (rem={n_rem}, add={n_add}, top-{top_k_rr})...")
    results["rank_reversal"] = rank_reversal_analysis(
        scorer, prep, bw,
        n_removals=n_rem, n_additions=n_add, top_k_monitor=top_k_rr)

    # ---- T9: Normalization Comparison ----
    _log("TEST  9/10: Normalization Comparison...")
    dm = prep[[_COL[c] for c in criteria]].copy()
    dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results["normalization"] = (nd, ns)

    # ---- T10: Compromise Ranking (≥2 methods only) ----
    if multi_method:
        _log("TEST 10/10: Compromise Ranking...")
        results["compromise"] = compromise_ranking(all_rankings)
    else:
        _log("TEST 10/10: Compromise Ranking... SKIPPED (single method)")

    # ---- Report ----
    report = generate_validation_report(results)

    # Add LLM report section
    if llm_result is not None:
        report += "\n" + format_llm_report(llm_result)

    methods_str = ", ".join(METHOD_NAMES[m] for m in methods_to_run)
    header = (f"Primary method: {primary_name}\n"
              f"Methods evaluated: {methods_str}\n"
              f"Weights: {bw}\nCriteria: {criteria}\n"
              f"Publications: {n}\n"
              f"Params: n_mc={n_mc}, n_bs={n_bs}, "
              f"bonus_start_z={bonus_start_z}, bonus_full_z={bonus_full_z}\n")
    if research_problem:
        header += f"Research problem: {research_problem[:120]}...\n"
    header += "\n"
    report = header + report

    # ---- Plots ----
    out_dir = "/content" if os.path.isdir("/content") else "/tmp"
    _log("Generating plots...")
    plot_paths = _generate_plots(results, primary_name, out_dir)

    try:
        from IPython.display import display, Image
        for p in plot_paths:
            display(Image(filename=p))
    except Exception:
        pass

    # ---- ZIP ----
    zname = primary_name.replace("+", "plus")
    zip_path = f"{out_dir}/mcda_{zname}_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("VALIDATION_REPORT.txt", report)
        zf.writestr("t01_weight_sensitivity.csv",
                     results["weight_sensitivity"].correlations.to_csv(index=False))
        zf.writestr("t02_criteria_removal.csv",
                     results["criteria_removal"].removal_steps.to_csv(index=False))
        if "cross_method" in results:
            cm_df, ws_mat = results["cross_method"]
            zf.writestr("t03_cross_method.csv", cm_df.to_csv(index=False))
            zf.writestr("t03_ws_matrix.csv", ws_mat.to_csv())
        for pn, pdf in results["parameter_sensitivity"].items():
            zf.writestr(f"t04_param_{pn}.csv", pdf.to_csv(index=False))
        zf.writestr("t05_precision_recall.csv",
                     results["precision_recall"].per_k.to_csv(index=False))
        if llm_result and llm_result.consensus is not None:
            zf.writestr("t05b_llm_consensus.csv",
                         llm_result.consensus.to_csv(index=False))
            for model_name, model_df in llm_result.per_model.items():
                safe_name = model_name.lower().replace(" ", "_")
                zf.writestr(f"t05b_llm_{safe_name}.csv",
                             model_df.to_csv(index=False))
        zf.writestr("t06_bootstrap.csv",
                     results["bootstrap"].to_csv(index=False))
        zf.writestr("t07_monte_carlo.csv",
                     results["monte_carlo"].all_runs.to_csv(index=False))
        zf.writestr("t08_rr_removal.csv",
                     results["rank_reversal"].removal_results.to_csv(index=False))
        zf.writestr("t08_rr_addition.csv",
                     results["rank_reversal"].addition_results.to_csv(index=False))
        zf.writestr("t09_normalization.csv", nd.to_csv(index=False))
        if "compromise" in results:
            zf.writestr("t10_compromise.csv",
                         results["compromise"].agreement_with_methods.to_csv(index=False))
        for pp in plot_paths:
            zf.write(pp, os.path.basename(pp))

    elapsed = time.time() - t0
    _log(f"Done in {elapsed:.1f}s — ZIP: {zip_path}")

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
                        methods_to_run=None,
                        w_sem=0.40, w_kw=0.25, w_ref=0.25, w_cit=0.10,
                        n_mc=1000, n_bs=500,
                        top_kw=5, top_ref=15,
                        bonus_start_z=2.0, bonus_full_z=4.0,
                        research_problem=None,
                        api_keys=None, llm_top_k=20,
                        log_fn=None):
    """Run ranking + validation. Returns (report, zip_path)."""
    ranked, prep, avail, bw = run_ranking(
        csv_path, method, w_sem, w_kw, w_ref, w_cit,
        top_kw, top_ref, bonus_start_z, bonus_full_z,
        research_problem=research_problem, log_fn=log_fn)
    return run_validation(
        prep, avail, bw, method, methods_to_run, n_mc, n_bs,
        bonus_start_z, bonus_full_z,
        ranked_df=ranked, research_problem=research_problem,
        api_keys=api_keys, llm_top_k=llm_top_k,
        log_fn=log_fn)


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
                "<br><span style='color:orange'>⚠ No semantic column in CSV. "
                "Enter Research Problem below to compute embeddings.</span>")

    sty = {"description_width": "120px"}

    # --- Research problem ---
    rp_w = W.Textarea(
        value="",
        placeholder="e.g. How does AI impact customer experience and to what extent "
                    "does its use in marketing improve satisfaction and loyalty?",
        description="Research problem:",
        layout=W.Layout(width="100%", height="80px"),
        style=sty)

    # --- Primary MCDA method ---
    method_w = W.RadioButtons(
        options=[("L-Scoring  (rank-based weighted sum)", "linear"),
                 ("Z-Scoring  (z-standardized weighted sum)", "zscore"),
                 ("L-Scoring+ (L-Scoring + outlier bonus)", "linear_plus")],
        value="linear_plus", description="", layout=W.Layout(width="100%"))

    # --- Methods to validate (for T3/T10) ---
    methods_w = W.SelectMultiple(
        options=[("L-Scoring", "linear"),
                 ("Z-Scoring", "zscore"),
                 ("L-Scoring+", "linear_plus")],
        value=["linear_plus"],
        description="Validate methods:",
        layout=W.Layout(width="450px", height="80px"),
        style=sty)

    # --- Weights ---
    w_sem_slider = W.FloatSlider(value=0.40, min=0.05, max=0.80, step=0.05,
                                  description="semantic", readout_format=".2f",
                                  style=sty,
                                  disabled="semantic" not in avail_check)
    w_kw  = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                           description="keywords", readout_format=".2f", style=sty)
    w_ref = W.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.05,
                           description="references", readout_format=".2f", style=sty)
    w_cit = W.FloatSlider(value=0.10, min=0.05, max=0.80, step=0.05,
                           description="citations", readout_format=".2f", style=sty)

    # --- Params ---
    mc_w  = W.IntSlider(value=1000, min=100, max=5000, step=100,
                         description="MC samples", style=sty)
    bs_w  = W.IntSlider(value=500, min=50, max=2000, step=50,
                         description="Bootstrap n", style=sty)
    bsz_w = W.FloatSlider(value=2.0, min=0.5, max=4.0, step=0.5,
                           description="bonus_start_z", readout_format=".1f", style=sty)
    bfz_w = W.FloatSlider(value=4.0, min=2.0, max=6.0, step=0.5,
                           description="bonus_full_z", readout_format=".1f", style=sty)

    # --- LLM API keys ---
    llm_toggle = W.Checkbox(value=False, description="Enable LLM Validation (T5b)",
                             style=sty)
    llm_topk_w = W.IntSlider(value=20, min=5, max=50, step=5,
                              description="LLM top-K", style=sty)
    key_openai = W.Text(value="", placeholder="sk-...", description="OpenAI key:", style=sty,
                         layout=W.Layout(width="100%"))
    key_anthropic = W.Text(value="", placeholder="sk-ant-...", description="Anthropic key:", style=sty,
                            layout=W.Layout(width="100%"))
    key_google = W.Text(value="", placeholder="AIza...", description="Google key:", style=sty,
                         layout=W.Layout(width="100%"))
    key_deepseek = W.Text(value="", placeholder="sk-...", description="DeepSeek key:", style=sty,
                           layout=W.Layout(width="100%"))
    key_xai = W.Text(value="", placeholder="xai-...", description="xAI key:", style=sty,
                      layout=W.Layout(width="100%"))
    key_kimi = W.Text(value="", placeholder="sk-...", description="Kimi key:", style=sty,
                       layout=W.Layout(width="100%"))

    llm_box = W.VBox([llm_topk_w, key_openai, key_anthropic, key_google,
                       key_deepseek, key_xai, key_kimi],
                      layout=W.Layout(display="none"))

    def _toggle_llm(change):
        llm_box.layout.display = "" if change["new"] else "none"
    llm_toggle.observe(_toggle_llm, names="value")

    # --- Output ---
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

    state = {"prep": None, "avail": None, "bw": None, "ranked": None}

    def _log(msg):
        with out_area: print(msg, flush=True)

    # Enable semantic slider when research problem is typed
    def _on_rp_change(change):
        has_text = bool(change["new"].strip())
        if has_text and "semantic" not in avail_check:
            w_sem_slider.disabled = False
        elif "semantic" not in avail_check:
            w_sem_slider.disabled = True
    rp_w.observe(_on_rp_change, names="value")

    def _on_rank(b):
        out_area.clear_output(wait=True)
        btn_rank.disabled = True; btn_test.disabled = True
        btn_rank.description = "Ranking..."
        try:
            rp_text = rp_w.value.strip() or None
            ranked, prep, avail, bw = run_ranking(
                csv_path, method_w.value,
                w_sem_slider.value, w_kw.value, w_ref.value, w_cit.value,
                bonus_start_z=bsz_w.value, bonus_full_z=bfz_w.value,
                research_problem=rp_text, log_fn=_log)
            state.update(prep=prep, avail=avail, bw=bw, ranked=ranked)

            out_dir = "/content" if os.path.isdir("/content") else "/tmp"
            mn = METHOD_NAMES[method_w.value].replace("+", "plus")
            rp_path = f"{out_dir}/ranking_{mn}.csv"
            ranked.to_csv(rp_path, index=False)
            _log(f"\nRanking CSV: {rp_path}")
            try:
                from google.colab import files as gf; gf.download(rp_path)
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
        if state["prep"] is None:
            _log("Run Step 1 first."); return
        btn_test.disabled = True
        btn_test.description = "Testing..."
        with out_area: print("\n" + "="*60, flush=True)
        try:
            selected_methods = list(methods_w.value)
            if method_w.value not in selected_methods:
                selected_methods.insert(0, method_w.value)

            # Gather LLM API keys
            llm_keys = None
            rp_text = rp_w.value.strip() or None
            if llm_toggle.value and rp_text:
                llm_keys = {}
                if key_openai.value.strip(): llm_keys["openai"] = key_openai.value.strip()
                if key_anthropic.value.strip(): llm_keys["anthropic"] = key_anthropic.value.strip()
                if key_google.value.strip(): llm_keys["google"] = key_google.value.strip()
                if key_deepseek.value.strip(): llm_keys["deepseek"] = key_deepseek.value.strip()
                if key_xai.value.strip(): llm_keys["xai"] = key_xai.value.strip()
                if key_kimi.value.strip(): llm_keys["kimi"] = key_kimi.value.strip()
                if not llm_keys:
                    _log("⚠ LLM validation enabled but no API keys provided.")
                    llm_keys = None

            report, zip_path = run_validation(
                state["prep"], state["avail"], state["bw"],
                method_w.value, selected_methods,
                mc_w.value, bs_w.value,
                bsz_w.value, bfz_w.value,
                ranked_df=state["ranked"],
                research_problem=rp_text,
                api_keys=llm_keys,
                llm_top_k=llm_topk_w.value,
                log_fn=_log)
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

    # --- Layout ---
    display(HTML(f"""
    <h2>EmbedSLR v3.0</h2>
    <table><tr><td><b>File:</b></td><td><code>{os.path.basename(csv_path)}</code></td></tr>
    <tr><td><b>Publications:</b></td><td>{n_total}</td></tr>
    <tr><td><b>Criteria:</b></td><td>{', '.join(avail_check)}</td></tr></table>{sem_note}
    <hr>
    <h4>Research Problem</h4>
    <small>Required for semantic embeddings (if no distance_cosine in CSV) and for LLM validation.
    Embeddings are computed for the research problem AND each article (title + abstract).</small>"""))
    display(rp_w)

    display(HTML("<h4>Primary MCDA method</h4>"))
    display(method_w)

    display(HTML("<h4>Methods to validate</h4>"
                 "<small>Select 1 = single-method (T3/T10 skipped). "
                 "Select 2-3 = cross-method correlation (T3) + compromise ranking (T10).</small>"))
    display(methods_w)

    display(HTML("<h4>Criteria weights</h4><small>Auto-normalized</small>"))
    display(w_sem_slider, w_kw, w_ref, w_cit)

    display(HTML("<h4>Parameters</h4>"))
    display(W.HBox([bsz_w, bfz_w]))
    display(W.HBox([mc_w, bs_w]))

    display(HTML("<h4>LLM Validation (T5b)</h4>"
                 "<small>Optional: LLMs evaluate top-K articles for semantic relevance "
                 "and research importance. Requires research problem + API keys.</small>"))
    display(llm_toggle)
    display(llm_box)

    display(HTML("<hr>"))
    display(W.HBox([btn_rank, btn_test]))
    display(HTML("<h4>Output</h4>"))
    display(out_area)


if __name__ == "__main__":
    run()

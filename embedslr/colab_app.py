#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py  —  EmbedSLR v3.0 Wizard (Gradio)
================================================
Tab 1 : Multi-Embedding Wizard   (consensus ranking)
Tab 2 : MCDA Sensitivity & Robustness  (10 tests)

Usage (Colab):
    !pip install -q git+https://github.com/s-matysik/EmbedSLR_v3.0.git
    from embedslr.colab_app import run
    run()
"""
from __future__ import annotations

import io, json, os, time, textwrap, zipfile
import itertools as it
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ── lightweight internal imports (no GPU libs needed) ────────────────
from .io import autodetect_columns, combine_title_abstract
from .advanced_scoring import (
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

# ── heavy imports (ensemble, embeddings) — deferred to first call ────
# Gradio is imported only inside build_demo() / run().


# =====================================================================
#  MODEL CATALOG
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
#  TAB 1 — WIZARD helpers
# =====================================================================
def _to_models(labels):
    from .ensemble import ModelSpec
    return [ModelSpec(*MODEL_CATALOG[l].split(":", 1)) for l in labels]

def _combo_tag(specs):
    return "__".join(s.label.replace("/", "_") for s in specs)

def _hit_dist(df):
    vc = df["hit_count"].value_counts().sort_index(ascending=False)
    return "; ".join(f"{int(k)}:{int(v)}" for k, v in vc.items() if k > 0)

def _set_keys(ok, ck, nk):
    if ok: os.environ["OPENAI_API_KEY"] = ok.strip()
    if ck: os.environ["COHERE_API_KEY"] = ck.strip()
    if nk: os.environ["NOMIC_API_KEY"]  = nk.strip()


def run_wizard(csv_path, query, model_labels, sizes, top_k, agg,
               ok, ck, nk, progress=None):
    import gradio as gr
    from .ensemble import run_ensemble, per_group_bibliometrics, build_embeddings_cache
    from .bibliometrics import full_report

    t0 = time.time()
    _set_keys(ok, ck, nk)

    df = pd.read_csv(csv_path)
    title_col, abs_col = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)
    if df.empty:      raise gr.Error("CSV is empty.")
    if not model_labels or len(model_labels) < 2:
        raise gr.Error("Select at least 2 models.")
    models = _to_models(model_labels)

    sizes_int = sorted({int(s) for s in sizes})
    sizes_int = [k for k in sizes_int if 2 <= k <= min(5, len(models))]
    if not sizes_int:
        raise gr.Error("Combination sizes exceed number of selected models.")

    if progress: progress(0, desc="Computing embeddings...")
    cache = build_embeddings_cache(df, "combined_text", query, models,
                                    progress=progress)

    combos = []
    for k in sizes_int:
        combos.extend(list(it.combinations(models, k)))

    rows = []
    _out = "/content" if os.path.isdir("/content") else "/tmp"
    zip_path = f"{_out}/embedslr_ensemble_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps({
            "query": query, "top_k": top_k, "aggregator": agg,
            "models": [m.__dict__ for m in models], "sizes": sizes_int,
        }, indent=2, ensure_ascii=False))

        it_combo = progress.tqdm(combos, desc="Combinations") if progress else combos
        for combo in it_combo:
            cl = list(combo); tag = _combo_tag(cl)
            ranked = run_ensemble(df, "combined_text", query, cl,
                                  top_k_per_model=top_k, aggregator=agg,
                                  precomputed=cache)
            try:    groups = per_group_bibliometrics(ranked)
            except: groups = pd.DataFrame()
            try:    report = full_report(ranked, path=None, top_n=top_k)
            except: report = "\n".join(str(t) for t in ranked[title_col].head(5))

            rows.append({"k": len(cl),
                         "combination": " + ".join(m.label for m in cl),
                         "n_records": len(ranked),
                         "hit_dist": _hit_dist(ranked),
                         "top3": ", ".join(str(t) for t in ranked[title_col].head(3))})

            buf = io.StringIO(); ranked.to_csv(buf, index=False)
            zf.writestr(f"ranking_k{len(cl)}_{tag}.csv", buf.getvalue())
            if not groups.empty:
                buf = io.StringIO(); groups.to_csv(buf, index=False)
                zf.writestr(f"groups_k{len(cl)}_{tag}.csv", buf.getvalue())
            zf.writestr(f"report_k{len(cl)}_{tag}.txt", report)

    summary = pd.DataFrame(rows).sort_values(["k","combination"]).reset_index(drop=True)
    info = (f"**Done.** {len(combos)} combinations, sizes {sizes_int}, "
            f"{time.time()-t0:.1f}s.  \nZIP: `{os.path.basename(zip_path)}`")
    return summary, zip_path, info


# =====================================================================
#  TAB 2 — MCDA helpers
# =====================================================================
def _prepare_mcda(df):
    out = df.copy()
    if 'distance_cosine' in out.columns:
        out['semantic_sim'] = 1.0 - out['distance_cosine'].astype(float)
    elif 'semantic_similarity' in out.columns:
        out['semantic_sim'] = out['semantic_similarity'].astype(float)
    else:
        out['semantic_sim'] = np.random.RandomState(42).uniform(0.3, 0.9, len(out))

    kw_col = next((c for c in ['Author Keywords','Authors Keywords',
                                'Keywords','Index Keywords'] if c in out.columns), None)
    if kw_col:
        kls = [parse_keywords_cell(x) for x in out[kw_col]]
        kf  = frequency_map(kls)
        out['kw_sum'] = [per_item_topk_sum(k, kf, 5)[0] or np.nan for k in kls]
    else:
        out['kw_sum'] = np.nan

    ref_col = next((c for c in ['References','Cited References'] if c in out.columns), None)
    if ref_col:
        rls = [parse_references_cell(x) for x in out[ref_col]]
        rf  = frequency_map(rls)
        out['ref_sum'] = [per_item_topk_sum(r, rf, 15)[0] or np.nan for r in rls]
    else:
        out['ref_sum'] = np.nan

    cc = next((c for c in ['Cited by','Cited By','Times Cited','Citations']
               if c in out.columns), None)
    out['citation_count'] = pd.to_numeric(out[cc], errors='coerce').fillna(0) if cc else 0.0

    for col in ['kw_sum', 'ref_sum']:
        m = out[col].mean()
        out[col] = out[col].fillna(m if not np.isnan(m) else 0.0)
    return out


def _make_scorers(prep):
    def l_fn(df_in, w):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        P = len(src); w = {k: v/sum(w.values()) for k,v in w.items()}
        cm = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
              'references': src['ref_sum'], 'citations': src['citation_count']}
        t = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cm: t += (P - (cm[c].rank(ascending=False, method='average') - 1)) * wt
        return t

    def z_fn(df_in, w):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        w = {k: v/sum(w.values()) for k,v in w.items()}
        cm = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
              'references': src['ref_sum'], 'citations': src['citation_count']}
        t = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cm:
                r = cm[c].astype(float); sd = r.std(ddof=0)
                t += ((r - r.mean())/sd if sd > 1e-12 else 0.0) * wt
        return t

    def lp_fn(df_in, w, bsz=2.0, bfz=4.0):
        src = df_in if 'semantic_sim' in df_in.columns else prep
        P = len(src); w = {k: v/sum(w.values()) for k,v in w.items()}
        cm = {'semantic': src['semantic_sim'], 'keywords': src['kw_sum'],
              'references': src['ref_sum'], 'citations': src['citation_count']}
        t = pd.Series(0.0, index=src.index)
        b = pd.Series(0.0, index=src.index)
        for c, wt in w.items():
            if c in cm:
                t += (P - (cm[c].rank(ascending=False, method='average') - 1)) * wt
                r = cm[c].astype(float); sd = r.std(ddof=0)
                if sd > 1e-12:
                    z = (r - r.median())/sd
                    f = ((z - bsz)/(bfz - bsz)).clip(lower=0.0); f[z >= bfz] = 1.0
                    b += f * float(P)
        return t + b.clip(upper=float(P))

    return l_fn, z_fn, lp_fn


def run_mcda_validation(csv_path, w_sem, w_kw, w_ref, w_cit,
                        n_mc, n_bs, progress=None):
    t0 = time.time()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    prep = _prepare_mcda(df); n = len(prep)

    raw = {'semantic': w_sem, 'keywords': w_kw, 'references': w_ref, 'citations': w_cit}
    s = sum(raw.values()); bw = {k: v/s for k,v in raw.items()}
    criteria = list(bw.keys())

    l_fn, z_fn, lp_fn = _make_scorers(prep)
    lp_wrap = lambda di, w: lp_fn(di, w)

    rankings = {
        'L-Scoring':  l_fn(prep, bw),
        'Z-Scoring':  z_fn(prep, bw),
        'L-Scoring+': lp_wrap(prep, bw),
    }
    results = {}

    def _p(msg):
        if progress: progress(0, desc=msg)

    _p("TEST 1/10: Weight Sensitivity...")
    results['weight_sensitivity'] = weight_sensitivity_oat(
        l_fn, prep, bw,
        perturbations=(-0.50,-0.30,-0.20,-0.10,0.10,0.20,0.30,0.50))

    _p("TEST 2/10: Criteria Removal...")
    results['criteria_removal'] = criteria_removal_analysis(l_fn, prep, bw)

    _p("TEST 3/10: Cross-Method...")
    cm_df, ws_mat = cross_method_correlation(rankings)
    results['cross_method'] = (cm_df, ws_mat)

    _p("TEST 4/10: Parameter Sensitivity...")
    def _lp_bz(di, w, bz): return lp_fn(di, w, bsz=bz, bfz=bz+2.0)
    results['parameter_sensitivity'] = {
        'bonus_start_z': parameter_sensitivity(
            _lp_bz, prep, 'bonus_start_z',
            [1.0,1.5,2.0,2.5,3.0,3.5], 2.0, bw)}

    _p("TEST 5/10: Precision/Recall...")
    comb = (prep['semantic_sim'].rank(pct=True)*0.4
            + prep['kw_sum'].rank(pct=True)*0.3
            + prep['ref_sum'].rank(pct=True)*0.2
            + prep['citation_count'].rank(pct=True)*0.1)
    rel = set(comb[comb >= comb.quantile(0.80)].index.tolist())
    results['precision_recall'] = precision_recall_at_k(
        rankings, rel, k_values=(5,10,20,50,100))

    _p(f"TEST 6/10: Bootstrap (n={n_bs})...")
    results['bootstrap'] = bootstrap_stability(
        l_fn, prep, bw, n_bootstrap=n_bs, top_k_values=(5,10,20))

    _p(f"TEST 7/10: Monte Carlo (n={n_mc})...")
    results['monte_carlo'] = monte_carlo_weights(
        l_fn, prep, criteria, n_samples=n_mc, reference_weights=bw)

    _p("TEST 8/10: Rank Reversal...")
    results['rank_reversal'] = rank_reversal_analysis(
        l_fn, prep, bw, n_removals=5, n_additions=3, top_k_monitor=20)

    _p("TEST 9/10: Normalization...")
    dm = prep[['semantic_sim','kw_sum','ref_sum','citation_count']].copy()
    dm.columns = criteria
    nd, ns = normalization_comparison(dm, bw)
    results['normalization'] = (nd, ns)

    _p("TEST 10/10: Compromise Ranking...")
    results['compromise'] = compromise_ranking(rankings)

    report = generate_validation_report(results)

    _out = "/content" if os.path.isdir("/content") else "/tmp"
    zip_path = f"{_out}/mcda_validation_{int(time.time())}.zip"
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
    info = (f"**Done.** 10 tests on {n} articles in {elapsed:.1f}s.  \n"
            f"Weights: sem={bw['semantic']:.2f} kw={bw['keywords']:.2f} "
            f"ref={bw['references']:.2f} cit={bw['citations']:.2f}")
    return report, zip_path, info


# =====================================================================
#  GRADIO UI  — built only when run() is called, not at import time
# =====================================================================
def build_demo():
    import gradio as gr

    with gr.Blocks(title="EmbedSLR v3.0") as demo:
        gr.Markdown("# EmbedSLR v3.0 — Wizard + MCDA Validation")

        # ── TAB 1 ───────────────────────────────────────────────────
        with gr.Tab("Multi-Embedding Wizard"):
            gr.Markdown(
                "**Scopus CSV -> models -> all 2/3/4/5 combinations -> ZIP**\n\n"
                "Method: per-model cosine ranking -> top-K vote -> consensus "
                "(hit_count desc, agg_distance asc, mean_rank asc)")
            with gr.Row():
                csv1 = gr.File(label="Scopus CSV", file_types=[".csv"],
                               type="filepath")
                q1 = gr.Textbox(label="Research query", lines=3,
                    placeholder="e.g. Does blockchain affect customer loyalty?")
            with gr.Row():
                m1 = gr.CheckboxGroup(list(MODEL_CATALOG.keys()),
                                      value=RECOMMENDED_DEFAULTS,
                                      label="Models (min 2)")
                s1 = gr.CheckboxGroup(["2","3","4","5"], value=["2","3","4"],
                                      label="Combination sizes")
            with gr.Row():
                tk1 = gr.Slider(10, 200, 50, step=1, label="top-K per model")
                ag1 = gr.Radio(["mean","min","median"], value="mean",
                               label="Distance aggregation")
            with gr.Accordion("API keys (only for cloud models)", open=False):
                ok1 = gr.Textbox(label="OPENAI_API_KEY", type="password")
                ck1 = gr.Textbox(label="COHERE_API_KEY", type="password")
                nk1 = gr.Textbox(label="NOMIC_API_KEY", type="password")

            b1 = gr.Button("Run Wizard", variant="primary")
            out1_df  = gr.Dataframe(label="Summary", interactive=False)
            out1_zip = gr.File(label="Download ZIP")
            out1_md  = gr.Markdown()

            def _go1(c, q, m, s, t, a, ok, ck, nk):
                if not c: raise gr.Error("Upload CSV.")
                if not q or not q.strip(): raise gr.Error("Enter query.")
                return run_wizard(c, q.strip(), m, s, int(t), a, ok, ck, nk)

            b1.click(_go1,
                     [csv1,q1,m1,s1,tk1,ag1,ok1,ck1,nk1],
                     [out1_df, out1_zip, out1_md])

        # ── TAB 2 ───────────────────────────────────────────────────
        with gr.Tab("MCDA Sensitivity & Robustness"):
            gr.Markdown(
                "**10 tests** evaluating multi-criteria ranking stability.\n\n"
                "| # | Test | Category |\n"
                "|---|------|----------|\n"
                "| 1 | Weight Sensitivity OAT | Sensitivity |\n"
                "| 2 | Criteria Removal | Sensitivity |\n"
                "| 3 | Cross-Method Correlation | Agreement |\n"
                "| 4 | Parameter Sensitivity | Sensitivity |\n"
                "| 5 | Precision / Recall / F1 | Retrieval effectiveness |\n"
                "| 6 | Bootstrap Stability | Robustness |\n"
                "| 7 | Monte Carlo Weights | Global sensitivity |\n"
                "| 8 | Rank Reversal | Robustness |\n"
                "| 9 | Normalization Comparison | Robustness |\n"
                "| 10 | Compromise Ranking | Agreement |\n\n"
                "*Methodology: Wieckowski & Salabun (2023), "
                "Wieckowski et al. (2025)*")

            csv2 = gr.File(label="Scopus CSV (same or different)",
                           file_types=[".csv"], type="filepath")
            with gr.Row():
                ws2 = gr.Slider(0.05, 0.80, 0.40, step=0.05, label="W semantic")
                wk2 = gr.Slider(0.05, 0.80, 0.25, step=0.05, label="W keywords")
                wr2 = gr.Slider(0.05, 0.80, 0.25, step=0.05, label="W references")
                wc2 = gr.Slider(0.05, 0.80, 0.10, step=0.05, label="W citations")
            with gr.Row():
                mc2 = gr.Slider(100, 5000, 1000, step=100,
                                label="Monte Carlo samples")
                bs2 = gr.Slider(50, 2000, 500, step=50,
                                label="Bootstrap iterations")

            b2 = gr.Button("Run 10 MCDA Tests", variant="primary")
            out2_txt = gr.Textbox(label="Validation Report", lines=35,
                                  show_copy_button=True)
            out2_zip = gr.File(label="Download results ZIP")
            out2_md  = gr.Markdown()

            def _go2(c, w1, w2, w3, w4, mc, bs):
                if not c: raise gr.Error("Upload CSV.")
                return run_mcda_validation(c, w1, w2, w3, w4, int(mc), int(bs))

            b2.click(_go2,
                     [csv2, ws2, wk2, wr2, wc2, mc2, bs2],
                     [out2_txt, out2_zip, out2_md])

    return demo


# =====================================================================
#  PUBLIC API
# =====================================================================
def run(share: bool = True, server_name: str = "0.0.0.0",
        server_port: Optional[int] = None):
    """
    Launch EmbedSLR Wizard + MCDA in Colab::

        from embedslr.colab_app import run
        run()
    """
    demo = build_demo()
    return demo.launch(share=share, server_name=server_name,
                       server_port=server_port)


if __name__ == "__main__":
    run()

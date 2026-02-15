#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_app.py — Wizard do EmbedSLR z obsługą wielu modeli (2/3/4/5)
Autor: seba
"""

from __future__ import annotations

import os
import io
import json
import time
import itertools as it
import textwrap
import zipfile
from typing import List, Tuple, Dict

import pandas as pd
import gradio as gr

# ─ EmbedSLR: wewnętrzne moduły ─
from embedslr.io import autodetect_columns, combine_title_abstract
from embedslr.ensemble import (
    ModelSpec, run_ensemble, per_group_bibliometrics, build_embeddings_cache
)
from embedslr.bibliometrics import full_report


# ───────────────────────────────────────────────────────────────────────────────
# Katalog modeli (dodawaj własne "etykieta": "provider:model_id")
# ───────────────────────────────────────────────────────────────────────────────
MODEL_CATALOG: Dict[str, str] = {
    "SBERT • sentence-transformers/all-MiniLM-L12-v2": "sbert:sentence-transformers/all-MiniLM-L12-v2",
    "SBERT • sentence-transformers/all-mpnet-base-v2": "sbert:sentence-transformers/all-mpnet-base-v2",
    "SBERT • sentence-transformers/all-distilroberta-v1": "sbert:sentence-transformers/all-distilroberta-v1",
    "OpenAI • text-embedding-3-large": "openai:text-embedding-3-large",
    "OpenAI (legacy) • text-embedding-ada-002": "openai:text-embedding-ada-002",
    "Nomic • nomic-embed-text-v1.5": "nomic:nomic-embed-text-v1.5",
    "Jina • jina-embeddings-v3": "jina:jina-embeddings-v3",
    "Cohere • embed-english-v3.0": "cohere:embed-english-v3.0",
}

RECOMMENDED_DEFAULTS = [
    "SBERT • sentence-transformers/all-MiniLM-L12-v2",
    "SBERT • sentence-transformers/all-mpnet-base-v2",
    "OpenAI (legacy) • text-embedding-ada-002",
    "Nomic • nomic-embed-text-v1.5",
]


# ───────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ───────────────────────────────────────────────────────────────────────────────
def _to_models(selected_labels: List[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for label in selected_labels:
        raw = MODEL_CATALOG[label]
        if ":" not in raw:
            raise ValueError(f"Nieprawidłowy model: {raw}")
        prov, mid = raw.split(":", 1)
        specs.append(ModelSpec(prov, mid))
    return specs

def _short(ms: ModelSpec) -> str:
    return ms.label.replace("/", "_")

def _combo_tag(specs: List[ModelSpec]) -> str:
    return "__".join(_short(m) for m in specs)

def _hit_distribution(df: pd.DataFrame) -> str:
    vc = df["hit_count"].value_counts().sort_index(ascending=False)
    return "; ".join(f"{int(k)}:{int(v)}" for k, v in vc.items() if k > 0)

def _ensure_env(openai_key: str | None, cohere_key: str | None, nomic_key: str | None):
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if cohere_key: os.environ["COHERE_API_KEY"] = cohere_key.strip()
    if nomic_key: os.environ["NOMIC_API_KEY"]  = nomic_key.strip()

def _prepare_df(csv_path: str) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)
    title_col, abs_col = autodetect_columns(df)
    df["combined_text"] = combine_title_abstract(df, title_col, abs_col)
    return df, title_col, abs_col


# ───────────────────────────────────────────────────────────────────────────────
# Rdzeń przetwarzania — wszystkie kombinacje 2/3/4/5
# ───────────────────────────────────────────────────────────────────────────────
def run_wizard(
    scopus_csv: str,
    query: str,
    selected_model_labels: List[str],
    sizes: List[str],
    top_k: int,
    aggregator: str,
    openai_key: str | None,
    cohere_key: str | None,
    nomic_key: str | None,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, str, str]:
    t0 = time.time()
    _ensure_env(openai_key, cohere_key, nomic_key)

    # 1) Dane
    df, title_col, _abs_col = _prepare_df(scopus_csv)
    if df.empty:
        raise gr.Error("Plik CSV wydaje się pusty.")

    # 2) Modele i rozmiary kombinacji
    if not selected_model_labels or len(selected_model_labels) < 2:
        raise gr.Error("Wybierz co najmniej 2 modele.")
    base_models = _to_models(selected_model_labels)

    sizes_int = sorted({int(s) for s in sizes})
    sizes_int = [k for k in sizes_int if 2 <= k <= min(5, len(base_models))]
    if not sizes_int:
        raise gr.Error("Zaznaczone rozmiary kombinacji są większe niż liczba wybranych modeli.")

    # 3) Jednorazowe liczenie embeddings (cache)
    progress(0, desc="🔧 Inicjalizacja modeli i embeddings (pierwsze uruchomienie może chwilę potrwać)")
    cache = build_embeddings_cache(df, "combined_text", query, base_models, progress=progress)

    # 4) Wszystkie kombinacje
    all_combos: List[List[ModelSpec]] = []
    for k in sizes_int:
        all_combos.extend(list(it.combinations(base_models, k)))

    # 5) Wyniki: w pamięci + ZIP na dysku
    out_rows = []
    zip_name = f"/content/embedslr_ensemble_{int(time.time())}.zip"

    with zipfile.ZipFile(zip_name, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Metadane wsadu
        meta = {
            "query": query,
            "top_k_per_model": top_k,
            "aggregator": aggregator,
            "selected_models": [m.__dict__ for m in base_models],
            "sizes": sizes_int,
        }
        zf.writestr("meta.json", json.dumps(meta, indent=2, ensure_ascii=False))

        # 6) Kombinacje (szybkie — korzystają z cache)
        for combo in progress.tqdm(all_combos, desc="📦 Liczenie kombinacji (używam cache embeddings)"):
            combo_list = list(combo)
            tag = _combo_tag(combo_list)
            k = len(combo_list)

            # ranking konsensusu
            ranked = run_ensemble(
                df, "combined_text", query, combo_list,
                top_k_per_model=top_k, aggregator=aggregator,
                precomputed=cache,
            )

            # raporty
            try:
                groups_df = per_group_bibliometrics(ranked)
            except Exception:
                groups_df = pd.DataFrame({"group": [], "n": [], "A": [], "A′": [], "B": [], "B′": []})
            try:
                report_txt = full_report(ranked, path=None, top_n=top_k)
            except Exception:
                head = ranked.head(top_k)
                report_txt = "Bibliometric report (minimal)\nTop-N titles:\n" + "\n".join(map(str, head.iloc[:,0].tolist()))

            # wiersz podsumowania
            top_titles = ", ".join([str(t) for t in ranked[title_col].head(3).tolist()])
            out_rows.append({
                "k": k,
                "kombinacja": " + ".join(m.label for m in combo_list),
                "id_kombinacji": tag,
                "liczba_rekordow": len(ranked),
                "rozkład_hit_count": _hit_distribution(ranked),
                "top3_tytuly": top_titles,
            })

            # pliki w ZIP
            buf = io.StringIO(); ranked.to_csv(buf, index=False)
            zf.writestr(f"ranking__k{k}__{tag}.csv", buf.getvalue())
            buf = io.StringIO(); groups_df.to_csv(buf, index=False)
            zf.writestr(f"groups__k{k}__{tag}.csv", buf.getvalue())
            zf.writestr(f"report__k{k}__{tag}.txt", report_txt)

    # 7) Tabela zbiorcza + log
    summary_df = pd.DataFrame(out_rows).sort_values(["k", "id_kombinacji"]).reset_index(drop=True)
    info_md = textwrap.dedent(f"""
    **OK. Gotowe.**  
    • Plik ZIP: `{os.path.basename(zip_name)}`  
    • Kombinacji: **{len(all_combos)}**, rozmiary: **{sizes_int}**  
    • Czas: **{time.time()-t0:.1f} s**

    **Uwaga metodologiczna:** zgodnie z publikacją, zestaw wskazany jednocześnie przez **4 modele** 
    cechuje się najwyższą spójnością (najwięcej wspólnych referencji; por. tab. 6–7 i rys. 14–17), 
    a zbyt duża liczba modeli degraduje koszyk (efekt „zbyt dużej komisji”). 
    Dlatego domyślnie zaznaczono 2, 3 i 4 (4 jako preferowane). 
    """)
    return summary_df, zip_name, info_md


# ───────────────────────────────────────────────────────────────────────────────
# UI (Gradio)
# ───────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="EmbedSLR – Multi‑Embedding Wizard") as demo:
    gr.Markdown(
        """
        # 📚 EmbedSLR – Multi‑Embedding Wizard
        **Wgraj CSV ze Scopusa → wybierz modele → uruchom wszystkie kombinacje 2/3/4/5 modeli → pobierz komplet wyników (ZIP).**

        Metoda: ranking per‑model (kosinus) → głosowanie **top‑K** → konsensus i sortowanie wg `hit_count ↓`, `agg_distance ↑`, `mean_rank ↑`.  
        (Zgodnie z ECAI 2025: *Efficient AI‑Powered Decision‑Making in SLR Using Multi‑Embedding Models*).        
        """
    )

    with gr.Row():
        csv_in = gr.File(label="Plik CSV (export Scopus)", file_types=[".csv"], type="filepath")
        query_in = gr.Textbox(label="Opis problemu badawczego / query", placeholder="Np. Does blockchain affect customer loyalty?", lines=3)

    with gr.Row():
        models_in = gr.CheckboxGroup(
            choices=list(MODEL_CATALOG.keys()),
            value=RECOMMENDED_DEFAULTS,
            label="Wybierz modele (dowolna liczba ≥2)"
        )
        sizes_in = gr.CheckboxGroup(
            choices=["2", "3", "4", "5"],
            value=["2", "3", "4"],
            label="Rozmiary kombinacji do uruchomienia"
        )

    with gr.Row():
        topk_in = gr.Slider(10, 200, value=50, step=1, label="top‑K per model (głosowanie)")
        agg_in = gr.Radio(choices=["mean", "min", "median"], value="mean", label="Agregacja dystansów (konsensus)")

    with gr.Accordion("Klucze API (opcjonalnie – tylko jeśli wybierasz modele chmurowe)", open=False):
        openai_key_in = gr.Textbox(label="OPENAI_API_KEY", type="password")
        cohere_key_in = gr.Textbox(label="COHERE_API_KEY", type="password")
        nomic_key_in = gr.Textbox(label="NOMIC_API_KEY", type="password")
        gr.Markdown("> Nie podawaj kluczy, jeśli korzystasz wyłącznie z modeli lokalnych (SBERT).")

    run_btn = gr.Button("▶️ Uruchom wszystkie kombinacje")
    with gr.Row():
        summary_out = gr.Dataframe(label="Podsumowanie kombinacji", interactive=False)  # bez 'wrap'
    with gr.Row():
        zip_out = gr.File(label="Pobierz ZIP z wynikami")
    info_out = gr.Markdown()

    def _run(csv_file, q, models, sizes, topk, agg, openai_key, cohere_key, nomic_key):
        if not csv_file: raise gr.Error("Wgraj plik CSV.")
        if not q or not q.strip(): raise gr.Error("Uzupełnij problem badawczy (query).")
        return run_wizard(csv_file, q.strip(), models, sizes, int(topk), agg, openai_key, cohere_key, nomic_key)

    run_btn.click(
        _run,
        inputs=[csv_in, query_in, models_in, sizes_in, topk_in, agg_in, openai_key_in, cohere_key_in, nomic_key_in],
        outputs=[summary_out, zip_out, info_out]
    )

def run(share: bool = True, server_name: str = "0.0.0.0", server_port: int | None = None):
    """
    Uruchamia Wizard w trybie zgodnym z Colab:
        from embedslr.colab_app import run
        run()
    """
    return demo.launch(share=share, server_name=server_name, server_port=server_port)

if __name__ == "__main__":
    run()

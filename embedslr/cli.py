from __future__ import annotations
import argparse, json, sys, os
import pandas as pd

from .io import read_csv, autodetect_columns, combine_title_abstract
from .embeddings import get_embeddings
from .similarity import rank_by_cosine
from .bibliometrics import full_report
from .advanced_scoring import rank_with_advanced_scoring
from .config import ScoringConfig, ColumnMap
from .smart_mcdm_biblio import SMARTConfig as SMARTBiblioConfig, rank_with_smart_biblio, read_candidates


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="embedslr", description="SLR screening toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- SMART MCDM ---
    ap_smart = sub.add_parser("smart", help="SMART (MCDM) re-ranking using existing metrics")
    ap_smart.add_argument("--input", "-i", required=True, help="Input file (CSV/TSV/Excel/Parquet/Feather)")
    ap_smart.add_argument("--out", "-o", default="ranked_smart.csv", help="Output CSV (default: ranked_smart.csv)")
    ap_smart.add_argument("--top", type=int, default=None, help="Return only top-N rows")

    # Column mapping
    ap_smart.add_argument("--col-semantic", default=None, help="Column for semantic (similarity or distance)")
    ap_smart.add_argument("--col-keywords", default=None, help="Column for keyword similarity")
    ap_smart.add_argument("--col-references", default=None, help="Column for reference overlap / coupling")
    ap_smart.add_argument("--col-mutual", default=None, help="Column for mutual citations")
    ap_smart.add_argument("--semantic-is-distance", action="store_true", help="Treat semantic column as distance (invert)")

    # Weights: either direct or ranks (4..10)
    ap_smart.add_argument("--w-semantic", type=float, default=None, help="Direct weight for semantic")
    ap_smart.add_argument("--w-keywords", type=float, default=None, help="Direct weight for keywords")
    ap_smart.add_argument("--w-references", type=float, default=None, help="Direct weight for references")
    ap_smart.add_argument("--w-mutual", type=float, default=None, help="Direct weight for mutual")
    ap_smart.add_argument("--weights-json", type=str, default=None, help="Path to JSON with weights or ranks")

    ap_smart.add_argument("--rank-semantic", type=int, default=8, help="Rank 4..10 for semantic (SMART)")
    ap_smart.add_argument("--rank-keywords", type=int, default=7, help="Rank 4..10 for keywords")
    ap_smart.add_argument("--rank-references", type=int, default=7, help="Rank 4..10 for references")
    ap_smart.add_argument("--rank-mutual", type=int, default=6, help="Rank 4..10 for mutual")

    ap_smart.add_argument("--scale-4-10", action="store_true", help="Aggregate on 4..10 scale (g_ij=4+6u_ij)")
    ap_smart.add_argument("--norm", choices=["minmax", "max"], default="minmax", help="Normalization strategy")
    ap_smart.add_argument("--available-only", action="store_true", help="Skip missing criteria and renormalize weights")


    # --- embed & rank by cosine ---
    emb = sub.add_parser("embed", help="Compute embeddings & cosine distances")
    emb.add_argument("-i", "--input", required=True, help="CSV file exported from Scopus")
    emb.add_argument("-q", "--query", required=True, help="Research problem / query string")
    emb.add_argument("-p", "--provider", default="sbert",
                     choices=["sbert", "openai", "cohere", "nomic", "jina"])
    emb.add_argument("-m", "--model", help="Override default model name")
    emb.add_argument("--api_key", help="Pass API key via CLI (otherwise use env var)")
    emb.add_argument("-o", "--out", default="ranking.csv")
    emb.add_argument("--json_embs", action="store_true",
                     help="Store embeddings JSON in the output CSV")

    # --- advanced scoring ---
    sc = sub.add_parser("score", help="Run advanced scoring (L-Scoring / Z-Scoring / L-Scoring+)")
    sc.add_argument("-i", "--input", required=True, help="Input CSV with at least Title/Abstract")
    sc.add_argument("--method", default="linear_plus",
                    choices=["linear", "zscore", "linear_plus"])
    sc.add_argument("--top_keywords", type=int, default=5)
    sc.add_argument("--top_references", type=int, default=15)
    sc.add_argument("--penalty_no_keywords", type=float, default=0.10)
    sc.add_argument("--penalty_no_references", type=float, default=0.10)
    sc.add_argument("--weights", default=None,
                    help='JSON mapping of weights, e.g. {"semantic":0.4,"keywords":0.3,"references":0.2,"citations":0.1}')
    sc.add_argument("--bonus_start_z", type=float, default=2.0)
    sc.add_argument("--bonus_full_z", type=float, default=4.0)
    sc.add_argument("--bonus_cap_points", type=float, default=None)
    sc.add_argument("--save_frequencies", action="store_true")
    sc.add_argument("--out_dir", default=".")
    sc.add_argument("-o", "--out", default="advanced_ranking.csv")
    # optional column mappings
    sc.add_argument("--col_keywords", default=None)
    sc.add_argument("--col_references", default=None)
    sc.add_argument("--col_citations", default=None)
    sc.add_argument("--col_semantic_similarity", default=None)
    sc.add_argument("--col_distance_cosine", default=None)

    # optional convenience: recompute semantic similarity if query is provided
    sc.add_argument("-q", "--query", default=None,
                    help="If provided: compute (1 - cosine distance) to query into 'semantic_similarity'")

    return ap


def cmd_embed(args: argparse.Namespace) -> None:
    df = read_csv(args.input)
    title_col, abs_col = autodetect_columns(df)
    text = combine_title_abstract(df, title_col, abs_col).tolist()
    provider = args.provider
    model = args.model
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("COHERE_API_KEY") or None

    print(f"[i] Embedding {len(text)} documents using {provider} {model or ''}".strip())
    embs = get_embeddings(text, provider=provider, model=model, api_key=api_key)
    ranked = rank_by_cosine(embs['query'], embs['docs'], df)

    if args.json_embs:
        ranked["combined_embeddings"] = [json.dumps(e) for e in embs["docs"]]

    ranked.to_csv(args.out, index=False)
    print(f"[✓] Ranking saved -> {args.out}")


def cmd_score(args: argparse.Namespace) -> None:
    df = read_csv(args.input)

    # optional: compute semantic similarity from query if asked
    cols = ColumnMap(
        keywords=args.col_keywords,
        references=args.col_references,
        citations=args.col_citations,
        semantic_similarity=args.col_semantic_similarity,
        distance_cosine=args.col_distance_cosine,
    )

    cfg = ScoringConfig(
        method=args.method,
        top_keywords=args.top_keywords,
        top_references=args.top_references,
        penalty_no_keywords=args.penalty_no_keywords,
        penalty_no_references=args.penalty_no_references,
        bonus_start_z=args.bonus_start_z,
        bonus_full_z=args.bonus_full_z,
        bonus_cap_points=args.bonus_cap_points,
        save_frequencies=args.save_frequencies,
        out_dir=args.out_dir,
        columns=cols,
    )

    # Recompute semantic_similarity if a query was provided
    if args.query:
        title_col, abs_col = autodetect_columns(df)
        text = combine_title_abstract(df, title_col, abs_col).tolist()
        embs = get_embeddings([args.query] + text, provider="sbert", model=None, api_key=None)
        query_vec = embs["docs"][0]
        doc_vecs = embs["docs"][1:]
        # cosine similarity
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        q = np.asarray(query_vec).reshape(1, -1)
        d = np.asarray(doc_vecs)
        sim = cosine_similarity(q, d)[0]
        df["semantic_similarity"] = sim
        cfg.columns.semantic_similarity = "semantic_similarity"

    # parse weights from JSON string if provided
    if args.weights:
        cfg.weights = json.loads(args.weights)

    res = rank_with_advanced_scoring(df, cfg)
    res.df.to_csv(args.out, index=False)
    print(f"[✓] Advanced ranking saved -> {args.out}")
    if cfg.save_frequencies:
        print(f"[✓] Frequencies saved to {cfg.out_dir}")



def cmd_smart(args: argparse.Namespace) -> None:
    df = read_candidates(args.input)

    colmap = {}
    if args.col_semantic: colmap["semantic"] = args.col_semantic
    if args.col_keywords: colmap["keywords"] = args.col_keywords
    if args.col_references: colmap["references"] = args.col_references
    if args.col_mutual: colmap["mutual"] = args.col_mutual

    explicit = None

    # Load weights/ranks from JSON if provided
    if args.weights_json:
        import json, os
        with open(args.weights_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        # Allow either 'weights' or 'ranks' keys in JSON
        if isinstance(payload, dict):
            if 'weights' in payload and isinstance(payload['weights'], dict):
                explicit = payload['weights']
            if 'ranks' in payload and isinstance(payload['ranks'], dict):
                # override ranks only if explicit not set
                if explicit is None:
                    # ensure proper keys
                    pass  # ranks will be read below from args (we could merge but keep simple)
    if any(x is not None for x in [args.w_semantic, args.w_keywords, args.w_references, args.w_mutual]):
        explicit = {
            "semantic": args.w_semantic or 0.0,
            "keywords": args.w_keywords or 0.0,
            "references": args.w_references or 0.0,
            "mutual": args.w_mutual or 0.0
        }

    cfg = SMARTBiblioConfig(
        column_map=colmap,
        explicit_weights=explicit,
        importance_ranks={
            "semantic": args.rank_semantic,
            "keywords": args.rank_keywords,
            "references": args.rank_references,
            "mutual": args.rank_mutual
        },
        scale_4to10=args.scale_4_10,
        available_only=args.available_only,
        normalize_strategy=args.norm,
        semantic_is_distance=True if args.semantic_is_distance else None,
        verbose=True
    )

    res = rank_with_smart_biblio(df, cfg, top_n=args.top)
    res.df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] SMART ranking saved -> {args.out}")
    print("[i] weights:", res.weights)
    print("[i] used columns:", res.used_columns)
    if res.dropped_criteria:
        print("[i] dropped criteria:", res.dropped_criteria)

def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "smart":
        cmd_smart(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

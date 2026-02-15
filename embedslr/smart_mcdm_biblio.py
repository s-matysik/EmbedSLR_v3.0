
"""
SMART (Simple Multi-Attribute Rating Technique) re-ranking module
for bibliometric selection of scientific publications.

Key properties of this version:
 - Uses ONLY existing metrics/columns (no recomputation of bibliometrics).
 - Supports flexible column mapping + automatic synonyms.
 - Lets users set weights either as ranks (4..10) or direct weights.
 - Adds robust file loading helpers.
 - Provides a CLI for easy use from terminal/Colab.

Method notes (per SMART):
 - Weights from ranks h_j:  w_j ∝ (sqrt(2))**h_j  (then normalized to sum=1).
 - Aggregation: sum_j w_j * u_ij (utilities in [0,1]), or on 4–10 scale if enabled.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import math
import warnings
import argparse
import sys
from pathlib import Path


# ---------------------------- Configuration ----------------------------

@dataclass
class SMARTConfig:
    # map 'semantic' | 'keywords' | 'references' | 'mutual' -> column name
    column_map: Dict[str, str] = field(default_factory=dict)

    # Optional: supply weights directly (they will be normalized to sum=1).
    explicit_weights: Optional[Dict[str, float]] = None

    # Or provide ranks in [4..10]; weights will be derived as (sqrt(2))**h per SMART.
    importance_ranks: Dict[str, int] = field(
        default_factory=lambda: {'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6}
    )

    # Aggregate on 4–10 scale (g_ij) instead of 0–1 utilities; default False.
    scale_4to10: bool = False

    # If True: only use criteria whose columns exist. Missing ones are silently dropped (weights renormalized).
    # If False: raise if a requested criterion is missing.
    available_only: bool = True

    # How to normalize individual criterion columns to utilities in [0, 1].
    # Supported: 'minmax' or 'max' (divide by max). 'minmax' recommended.
    normalize_strategy: str = "minmax"

    # If your semantic column is a *distance*, set this to True or let the resolver infer by column name.
    semantic_is_distance: Optional[bool] = None

    # Column names to search automatically if column_map is partial.
    # (Order matters; first found will be used.)
    synonyms: Dict[str, List[str]] = field(default_factory=lambda: {
        'semantic': [
            'semantic_similarity', 'cosine_similarity', 'similarity', 'sim',
            'score_similarity', 'cos_sim', 'cosine_sim', 'distance_cosine',
            'cosine_distance', 'dist_cosine', 'distance', 'semantic_dist'
        ],
        'keywords': [
            'kw_similarity', 'keywords_similarity', 'author_keywords_similarity',
            'ak_similarity', 'kw_jaccard', 'keywords_jaccard', 'author_keywords_jaccard',
            'keyword_overlap_score', 'kw_overlap', 'kw_sim'
        ],
        'references': [
            'bibliographic_coupling', 'biblio_coupling', 'bc_score', 'bc_strength',
            'common_references', 'common_refs', 'common_refs_count',
            'co_citation_score', 'co_citation', 'ref_overlap_score'
        ],
        'mutual': [
            'mutual_citations', 'reciprocal_citations', 'reciprocal_citation',
            'two_way_citations', 'bidirectional_citations', 'mutual_cite_score',
            'mutual_citations_count'
        ]
    })

    # Diagnostic verbosity
    verbose: bool = True


@dataclass
class SMARTResult:
    df: pd.DataFrame                              # original df with SMART_score column appended
    utilities: pd.DataFrame                       # u_ij for used criteria
    contributions: pd.DataFrame                   # w_j * u_ij (or w_j * g_ij if scale_4to10)
    weights: Dict[str, float]                     # normalized weights actually used
    used_columns: Dict[str, str]                  # mapping criterion -> df column actually used
    dropped_criteria: List[str]                   # criteria requested but not used


# ---------------------------- File I/O helpers ----------------------------

READ_CANDIDATE_EXTS = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.parquet', '.pq', '.feather', '.ft'}

def read_candidates(path: str) -> pd.DataFrame:
    """Read a table of candidates from CSV/TSV/Excel/Parquet/Feather with best-effort encoding fallbacks."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = p.suffix.lower()
    if ext not in READ_CANDIDATE_EXTS:
        raise ValueError(f"Unsupported input extension {ext}. Supported: {sorted(READ_CANDIDATE_EXTS)}")
    if ext in {'.csv', '.txt'}:
        for enc in ('utf-8', 'utf-8-sig', 'ISO-8859-1'):
            try:
                return pd.read_csv(p, encoding=enc, on_bad_lines='skip', low_memory=False)
            except UnicodeDecodeError:
                continue
        # last attempt without encoding
        return pd.read_csv(p, on_bad_lines='skip', low_memory=False)
    if ext == '.tsv':
        for enc in ('utf-8', 'utf-8-sig', 'ISO-8859-1'):
            try:
                return pd.read_csv(p, sep='\t', encoding=enc, on_bad_lines='skip', low_memory=False)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(p, sep='\t', on_bad_lines='skip', low_memory=False)
    if ext in {'.xlsx', '.xls'}:
        return pd.read_excel(p)
    if ext in {'.parquet', '.pq'}:
        return pd.read_parquet(p)
    if ext in {'.feather', '.ft'}:
        return pd.read_feather(p)
    # fallback
    return pd.read_csv(p, encoding='utf-8', on_bad_lines='skip', low_memory=False)


# ---------------------------- Core utilities ----------------------------

def _normalize_series(s: pd.Series, method: str = "minmax") -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if method == "minmax":
        mn, mx = float(np.nanmin(s.values)), float(np.nanmax(s.values))
        if mx - mn == 0:
            # constant vector -> return zeros
            return pd.Series(np.zeros(len(s)), index=s.index, name=s.name)
        return (s - mn) / (mx - mn)
    elif method == "max":
        mx = float(np.nanmax(s.values))
        if mx == 0:
            return pd.Series(np.zeros(len(s)), index=s.index, name=s.name)
        return s / mx
    else:
        raise ValueError(f"Unknown normalize strategy: {method}")


def _is_distance_name(colname: str) -> bool:
    name = colname.lower()
    return 'dist' in name or 'distance' in name


def _derive_weights_from_ranks(ranks: Dict[str, int]) -> Dict[str, float]:
    # SMART: w_j ∝ (sqrt(2))**h_j  (Eq. 7), then normalized (Eq. 8).
    transformed = {k: (math.sqrt(2.0) ** int(v)) for k, v in ranks.items()}
    s = sum(transformed.values())
    if s == 0:
        # fallback equal weights
        n = len(transformed)
        return {k: 1.0/n for k in transformed}
    return {k: v / s for k, v in transformed.items()}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in weights.values())
    if s == 0:
        n = len(weights)
        return {k: 1.0/n for k in weights}
    return {k: float(v)/s for k, v in weights.items()}


def resolve_columns(df: pd.DataFrame, config: 'SMARTConfig') -> Tuple[Dict[str, str], List[str]]:
    """Return a mapping of used columns and list of dropped criteria."""
    used = {}
    dropped = []
    for crit in ['semantic', 'keywords', 'references', 'mutual']:
        # Prefer explicit mapping
        col = config.column_map.get(crit)
        if col and col in df.columns:
            used[crit] = col
            continue
        # Try synonyms
        for cand in config.synonyms.get(crit, []):
            if cand in df.columns:
                used[crit] = cand
                break
        if crit not in used:
            if config.available_only:
                dropped.append(crit)
            else:
                raise KeyError(
                    f"Missing column for criterion '{crit}'. Provide it in SMARTConfig.column_map or include a known synonym."
                )
    return used, dropped


def _criterion_utility(df: pd.DataFrame, col: str, config: 'SMARTConfig', crit: str) -> pd.Series:
    s = df[col]
    # Semantic may be a distance (lower is better) -> invert after scaling if needed
    if crit == 'semantic':
        is_distance = config.semantic_is_distance
        if is_distance is None:
            is_distance = _is_distance_name(col)
        if is_distance:
            # scale distance to [0,1], then invert to similarity utility
            u = _normalize_series(s, config.normalize_strategy)
            return 1.0 - u
    # All others: higher is better; scale to [0,1].
    return _normalize_series(s, config.normalize_strategy)


def _maybe_to_scale_4to10(u: pd.Series) -> pd.Series:
    # Map utility [0,1] -> [4,10] per SMART qualitative scale (Table 2).
    return 4.0 + 6.0 * u


def _safe_concat_cols(cols: List[pd.Series]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1)
    return df


# ---------------------------- Public API ----------------------------

def rank_with_smart_biblio(df: pd.DataFrame,
                           config: Optional['SMARTConfig'] = None,
                           top_n: Optional[int] = None) -> 'SMARTResult':
    """
    Re-rank publications using SMART with given bibliometric/semantic criteria.
    This function does NOT compute metrics; it only uses the columns already present.
    """
    if config is None:
        config = SMARTConfig()

    # Resolve usable columns
    used_cols, dropped = resolve_columns(df, config)

    # Compute utilities per used criterion
    utils_cols = []
    for crit, col in used_cols.items():
        try:
            u = _criterion_utility(df, col, config, crit)
        except Exception as e:
            if config.available_only:
                if config.verbose:
                    warnings.warn(f"Criterion '{crit}' dropped due to error: {e}")
                dropped.append(crit)
                continue
            raise
        utils_cols.append(u.rename(crit))

    utilities = _safe_concat_cols(utils_cols)

    if utilities.empty:
        raise ValueError("No usable criteria found. Check your column names or column_map.")

    # Prepare weights
    if config.explicit_weights is not None:
        # Use only weights for criteria actually present
        w = {crit: config.explicit_weights.get(crit, 0.0) for crit in utilities.columns}
        weights = _normalize_weights(w)
    else:
        # From ranks
        ranks_present = {crit: config.importance_ranks.get(crit, 6) for crit in utilities.columns}
        weights = _derive_weights_from_ranks(ranks_present)

    # Aggregate
    if config.scale_4to10:
        # Work on g_ij in [4,10]
        g = utilities.apply(_maybe_to_scale_4to10)
        contributions = g * pd.Series(weights)
        score = contributions.sum(axis=1)
    else:
        # Work on u_ij in [0,1]
        contributions = utilities * pd.Series(weights)
        score = contributions.sum(axis=1)

    out_df = df.copy()
    out_df['SMART_score'] = score
    out_df = out_df.sort_values('SMART_score', ascending=False)

    if top_n is not None:
        out_df = out_df.head(top_n)

    return SMARTResult(
        df=out_df.reset_index(drop=True),
        utilities=utilities.loc[out_df.index].reset_index(drop=True),
        contributions=contributions.loc[out_df.index].reset_index(drop=True),
        weights=weights,
        used_columns=used_cols,
        dropped_criteria=dropped
    )


# ---------------------------- Diagnostics ----------------------------

def quick_diagnose(df: pd.DataFrame, config: Optional['SMARTConfig'] = None) -> pd.DataFrame:
    """Report which columns were matched for each criterion and basic stats."""
    if config is None:
        config = SMARTConfig()
    used, dropped = resolve_columns(df, config)
    rows = []
    for crit in ['semantic', 'keywords', 'references', 'mutual']:
        col = used.get(crit, None)
        if col is None:
            rows.append({'criterion': crit, 'column': None, 'status': 'missing'})
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        rows.append({
            'criterion': crit, 'column': col, 'status': 'ok',
            'non_null': s.notna().sum(), 'min': float(np.nanmin(s)), 'max': float(np.nanmax(s)),
            'mean': float(np.nanmean(s))
        })
    diag = pd.DataFrame(rows)
    if config.verbose:
        print(diag)
        if dropped:
            print("Dropped criteria (not available):", dropped)
    return diag


# ---------------------------- CLI ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SMART (bibliometry) re-ranking CLI")
    p.add_argument('--input', '-i', required=True, help="Ścieżka do pliku z kandydatami (CSV/TSV/Excel/Parquet/Feather)")
    p.add_argument('--output', '-o', default=None, help="Ścieżka do wyjściowego CSV (domyślnie <input>_smart.csv)")
    p.add_argument('--top', type=int, default=None, help="Zwróć tylko TOP-N wierszy")

    # Column mapping
    p.add_argument('--col-semantic', help="Nazwa kolumny dla kryterium 'semantic'")
    p.add_argument('--col-keywords', help="Nazwa kolumny dla kryterium 'keywords'")
    p.add_argument('--col-references', help="Nazwa kolumny dla kryterium 'references'")
    p.add_argument('--col-mutual', help="Nazwa kolumny dla kryterium 'mutual'")
    p.add_argument('--semantic-is-distance', action='store_true', help="Wymuś interpretację kolumny semantycznej jako dystans (odwracanie skali)")

    # Weights
    p.add_argument('--w-semantic', type=float, help="Waga bezpośrednia dla 'semantic'")
    p.add_argument('--w-keywords', type=float, help="Waga bezpośrednia dla 'keywords'")
    p.add_argument('--w-references', type=float, help="Waga bezpośrednia dla 'references'")
    p.add_argument('--w-mutual', type=float, help="Waga bezpośrednia dla 'mutual'")
    p.add_argument('--rank-semantic', type=int, default=8, help="Ranga (4–10) dla 'semantic' (używana jeśli nie podano wag)")
    p.add_argument('--rank-keywords', type=int, default=7, help="Ranga (4–10) dla 'keywords'")
    p.add_argument('--rank-references', type=int, default=7, help="Ranga (4–10) dla 'references'")
    p.add_argument('--rank-mutual', type=int, default=6, help="Ranga (4–10) dla 'mutual'")

    # Other
    p.add_argument('--scale-4-10', action='store_true', help="Agreguj na skali 4–10 (g_ij = 4+6*u_ij)")
    p.add_argument('--norm', choices=['minmax', 'max'], default='minmax', help="Strategia normalizacji pojedynczych kryteriów")
    p.add_argument('--available-only', action='store_true', help="Pomiń brakujące kryteria i przeskaluj wagi")
    p.add_argument('--verbose', action='store_true', help="Więcej logów")
    return p


def cli_main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Build config
    column_map = {}
    if args.col_semantic: column_map['semantic'] = args.col_semantic
    if args.col_keywords: column_map['keywords'] = args.col_keywords
    if args.col_references: column_map['references'] = args.col_references
    if args.col_mutual: column_map['mutual'] = args.col_mutual

    explicit_weights = None
    # If any direct weights provided, use them; they will be normalized
    if any(v is not None for v in [args.w_semantic, args.w_keywords, args.w_references, args.w_mutual]):
        explicit_weights = {
            'semantic': args.w_semantic or 0.0,
            'keywords': args.w_keywords or 0.0,
            'references': args.w_references or 0.0,
            'mutual': args.w_mutual or 0.0
        }

    cfg = SMARTConfig(
        column_map=column_map,
        explicit_weights=explicit_weights,
        importance_ranks={
            'semantic': args.rank_semantic,
            'keywords': args.rank_keywords,
            'references': args.rank_references,
            'mutual': args.rank_mutual
        },
        scale_4to10=args.scale_4_10,
        available_only=args.available_only,
        normalize_strategy=args.norm,
        semantic_is_distance=True if args.semantic_is_distance else None,
        verbose=args.verbose
    )

    # Load data
    df = read_candidates(args.input)
    if args.verbose:
        print("[SMART] columns detected:", list(df.columns)[:60])

    # Diagnose
    if args.verbose:
        quick_diagnose(df, cfg)

    # Rank
    res = rank_with_smart_biblio(df, cfg, top_n=args.top)

    # Save
    out_path = args.output
    if out_path is None:
        p = Path(args.input)
        out_path = str(p.with_suffix('')) + "_smart.csv"
    res.df.to_csv(out_path, index=False, encoding='utf-8')
    if args.verbose:
        print("[SMART] saved:", out_path)
        print("[SMART] weights:", res.weights)
        print("[SMART] used_columns:", res.used_columns)
        if res.dropped_criteria:
            print("[SMART] dropped:", res.dropped_criteria)
    return 0


if __name__ == '__main__':
    sys.exit(cli_main())

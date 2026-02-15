from __future__ import annotations
"""
Advanced scoring for EmbedSLR.

Implements:
- Frequency-based keyword and reference contributions
- Rank-to-points conversion with tie averaging
- Linear weighted scoring (L-Scoring)
- Z-Scoring (standardized) aggregation
- L-Scoring+ (linear + outlier bonus)

References:
• User requirements "Update do SoftX 1" (Polish specification).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import re
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .config import ScoringConfig, ColumnMap, Criterion


# ----------------------------- parsing utils -----------------------------

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_WS_RE  = re.compile(r"\s+")

def _canon_kw(s: str) -> str:
    s = (s or "").strip().lower()
    # remove bordering punctuation and normalize dashes and whites
    s = re.sub(r"[\u2013\u2014\u2212]", "-", s)   # en/em/minus to hyphen
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    s = _WS_RE.sub(" ", s)
    return s

def parse_keywords_cell(cell: object) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    # split on ; , / | • and similar
    parts = re.split(r"[;\,\/\|•]+", cell)
    return [k for k in (_canon_kw(p) for p in parts) if k]

def _norm_ref_token(ref: str) -> str:
    """Return a stable key for a single reference entry."""
    m = _DOI_RE.search(ref or "")
    if m:
        return "doi:" + m.group(0).lower()
    # fallback: first ~12 words of a normalized title-like string
    s = (ref or "").strip().lower()
    s = re.sub(r"\(.*?\)", " ", s)      # drop parenthetical noise
    s = re.sub(r"[^a-z0-9\s\-:]", " ", s)
    s = _WS_RE.sub(" ", s)
    # try to skip author list "smith j., 2021." up to first colon/ dash
    s = re.sub(r"^.*?[:\-]\s*", "", s)
    return "t:" + " ".join(s.split()[:12])

def parse_references_cell(cell: object) -> List[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    # split on newlines or ; separators used by Scopus/CSV exports
    parts = re.split(r"[\n\r;]+", cell)
    toks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        toks.append(_norm_ref_token(p))
    return toks

# ----------------------- frequency / top-k helpers -----------------------

def frequency_map(rows: Iterable[List[str]]) -> Counter:
    c = Counter()
    for lst in rows:
        c.update(lst)
    return c

def per_item_topk_sum(tokens: List[str], freq: Counter, k: int) -> Tuple[Optional[float], int]:
    """
    Sum frequencies of the k most frequent tokens *among those present for the item*.
    Returns (sum_or_none, count_used).
    If the item has fewer than k tokens, we pad missing slots with the mean of existing values.
    If the item has zero tokens, returns (None, 0).
    """
    if not tokens:
        return None, 0
    vals = sorted((freq.get(t, 0) for t in set(tokens)), reverse=True)
    if not vals:
        return None, 0
    used = vals[:k]
    if len(used) < k:
        mean_v = float(np.mean(used)) if used else 0.0
        used = used + [mean_v] * (k - len(used))
    return float(sum(used)), min(len(vals), k)

def _safe_rank_points(values: pd.Series, higher_is_better: bool, tie_method: str, P: int) -> pd.Series:
    """
    Convert raw numeric values into 'points' where the best gets P, next P-1, etc.
    Ties receive the average of the points for the occupied positions.
    NaNs are ranked to the bottom.
    """
    if values.isna().all():
        return pd.Series([0.0] * len(values), index=values.index)
    asc = not higher_is_better
    ranks = values.rank(method=tie_method, ascending=asc)
    # When all equal, ranks == (n+1)/2; still fine.
    pts = P - (ranks - 1.0)
    return pts

def _zscore(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0]*len(s), index=s.index)
    if higher_is_better:
        return (s - mu) / sd
    else:
        return (mu - s) / sd

def _z_from_median(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = series.astype(float)
    med = s.median()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0]*len(s), index=s.index)
    if higher_is_better:
        return (s - med) / sd
    else:
        return (med - s) / sd


# ------------------------------ main API --------------------------------

@dataclass
class ScoringResult:
    df: pd.DataFrame
    keyword_freq: pd.DataFrame
    reference_freq: pd.DataFrame
    P: int
    config: ScoringConfig


def _autodetect_columns(df: pd.DataFrame, cm: ColumnMap) -> ColumnMap:
    cm = ColumnMap(**{**cm.__dict__})
    cols = {c.lower(): c for c in df.columns}
    def find(names: List[str]) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    cm.keywords   = cm.keywords   or find(["Author Keywords", "Authors Keywords", "Author Keywords (DE)", "DE", "Keywords", "Index Keywords"])
    cm.references = cm.references or find(["References", "Reference", "Cited References"])
    cm.citations  = cm.citations  or find(["Cited by", "Cited By", "Times Cited", "Citations"])
    cm.semantic_similarity = cm.semantic_similarity or find(["semantic_similarity", "cosine_similarity", "similarity"])
    cm.distance_cosine     = cm.distance_cosine     or find(["distance_cosine", "cosine_distance", "cos_distance"])
    cm.id_col    = cm.id_col    or find(["EID", "ID", "eid"])
    cm.title_col = cm.title_col or find(["Title", "Article Title", "Document Title", "TI"])
    return cm


def rank_with_advanced_scoring(df: pd.DataFrame, config: Optional[ScoringConfig] = None) -> ScoringResult:
    """
    Compute rankings and scores per the Polish specification.

    Returns a DataFrame with additional columns:
    - kw_sum, ref_sum: frequency-derived sums per item
    - *_rank_pts: points per criterion (P..1 with tie averaging)
    - score_linear, score_z, score_linear_plus
    """
    cfg = config or ScoringConfig()
    cm  = _autodetect_columns(df, cfg.columns)

    P = len(df)

    # --- parse tokens ---
    kw_lists  = [parse_keywords_cell(x) for x in df[cm.keywords]] if cm.keywords and cm.keywords in df else [[] for _ in range(P)]
    ref_lists = [parse_references_cell(x) for x in df[cm.references]] if cm.references and cm.references in df else [[] for _ in range(P)]

    kw_freq  = frequency_map(kw_lists)
    ref_freq = frequency_map(ref_lists)

    # Save frequencies if requested
    kw_freq_df  = pd.DataFrame(sorted(kw_freq.items(), key=lambda kv: (-kv[1], kv[0])), columns=["keyword", "count"])
    ref_freq_df = pd.DataFrame(sorted(ref_freq.items(), key=lambda kv: (-kv[1], kv[0])), columns=["reference_token", "count"])

    # --- per-item sums for tokens ---
    kw_sums = []
    ref_sums = []
    for kws in kw_lists:
        s, _ = per_item_topk_sum(kws, kw_freq, cfg.top_keywords)
        kw_sums.append(np.nan if s is None else s)
    for refs in ref_lists:
        s, _ = per_item_topk_sum(refs, ref_freq, cfg.top_references)
        ref_sums.append(np.nan if s is None else s)

    out = df.copy()
    out["kw_sum"]  = kw_sums
    out["ref_sum"] = ref_sums

    # --- semantic raw ---
    # prefer explicit similarity [0..1]; otherwise derive 1 - distance_cosine if present
    if cm.semantic_similarity and cm.semantic_similarity in out:
        semantic_raw = out[cm.semantic_similarity].astype(float)
        sem_hib = True
    elif cm.distance_cosine and cm.distance_cosine in out:
        semantic_raw = 1.0 - out[cm.distance_cosine].astype(float)
        sem_hib = True
    else:
        semantic_raw = pd.Series([np.nan]*P, index=out.index)
        sem_hib = True

    # --- citations raw ---
    if cm.citations and cm.citations in out:
        citations_raw = pd.to_numeric(out[cm.citations], errors="coerce")
    else:
        citations_raw = pd.Series([np.nan]*P, index=out.index)

    # Build chosen criterion list
    all_crit: List[Criterion] = ["semantic", "keywords", "references", "citations"]
    active_crit = cfg.use_criteria or all_crit

    # --- rank points per criterion ---
    tie = cfg.tie_method
    # semantic points
    sem_pts = _safe_rank_points(semantic_raw, higher_is_better=True, tie_method=tie, P=P)
    # keywords points
    kw_pts_native = _safe_rank_points(out["kw_sum"], higher_is_better=True, tie_method=tie, P=P)
    # references points
    ref_pts_native = _safe_rank_points(out["ref_sum"], higher_is_better=True, tie_method=tie, P=P)
    # citations points
    cit_pts = _safe_rank_points(citations_raw, higher_is_better=True, tie_method=tie, P=P)

    # --- fill missing token categories with average of other *similarity* categories and apply penalty ---
    # "similarity categories" per spec: semantic + references for keywords; semantic + keywords for references
    kw_missing = out["kw_sum"].isna()
    if kw_missing.any():
        fill_base = (sem_pts + ref_pts_native) / 2.0
        filled = fill_base.loc[kw_missing] * (1.0 - cfg.penalty_no_keywords)
        kw_pts_native.loc[kw_missing] = filled

    ref_missing = out["ref_sum"].isna()
    if ref_missing.any():
        fill_base = (sem_pts + kw_pts_native) / 2.0
        filled = fill_base.loc[ref_missing] * (1.0 - cfg.penalty_no_references)
        ref_pts_native.loc[ref_missing] = filled

    # Attach point columns
    out["semantic_rank_pts"]  = sem_pts
    out["keywords_rank_pts"]  = kw_pts_native
    out["references_rank_pts"]= ref_pts_native
    out["citations_rank_pts"] = cit_pts

    # ------------------- Aggregations -------------------

    weights = cfg.normalized_weights(active_crit)

    # L-Scoring (weighted sum of points)
    score_linear = sum(out[f"{c}_rank_pts"] * weights.get(c, 0.0) for c in active_crit)

    # Z-Scoring (standardize raw values per criterion)
    z_map: Dict[str, pd.Series] = {}
    raw_map: Dict[str, pd.Series] = {
        "semantic": semantic_raw,
        "keywords": out["kw_sum"],
        "references": out["ref_sum"],
        "citations": citations_raw
    }
    hib_map: Dict[str, bool] = {
        "semantic": True,
        "keywords": True,
        "references": True,
        "citations": True
    }
    for c in active_crit:
        z_map[c] = _zscore(raw_map[c], higher_is_better=hib_map[c])

    # For rows with NaN z in keywords/references (no tokens) → average of the other similarity z's minus penalty
    k_nan = z_map["keywords"].isna() if "keywords" in z_map else pd.Series(False, index=out.index)
    r_nan = z_map["references"].isna() if "references" in z_map else pd.Series(False, index=out.index)
    if k_nan.any() and "semantic" in z_map and "references" in z_map:
        fill = (z_map["semantic"] + z_map["references"]) / 2.0
        z_map["keywords"].loc[k_nan] = fill.loc[k_nan] * (1.0 - cfg.penalty_no_keywords)
    if r_nan.any() and "semantic" in z_map and "keywords" in z_map:
        fill = (z_map["semantic"] + z_map["keywords"]) / 2.0
        z_map["references"].loc[r_nan] = fill.loc[r_nan] * (1.0 - cfg.penalty_no_references)

    score_z = sum(z_map[c] * weights.get(c, 0.0) for c in active_crit)

    # L-Scoring+ (linear + bonus for outliers)
    start_z = cfg.bonus_start_z
    full_z  = cfg.bonus_full_z
    cap     = cfg.bonus_cap_points or float(P)

    bonus_total = pd.Series(0.0, index=out.index)
    # Use z from median for robustness
    zmed_map: Dict[str, pd.Series] = {
        c: _z_from_median(raw_map[c], hib_map[c]) for c in active_crit
    }
    for c in active_crit:
        zmed = zmed_map[c].fillna(0.0)
        # piecewise linear: 0 … P points per criterion
        frac = (zmed - start_z) / (full_z - start_z)
        frac = frac.clip(lower=0.0)
        # if above full_z → 1.0
        frac[zmed >= full_z] = 1.0
        bonus_total += frac * float(P)
    # cap bonus per article
    bonus_total = bonus_total.clip(upper=cap)

    score_linear_plus = score_linear + bonus_total

    out["score_linear"]      = score_linear
    out["score_zscore"]      = score_z
    out["score_linear_plus"] = score_linear_plus

    # Primary sort by the selected method
    method_to_col = {
        "linear": "score_linear",
        "zscore": "score_zscore",
        "linear_plus": "score_linear_plus"
    }
    sort_col = method_to_col.get(cfg.method, "score_linear_plus")
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out["rank_overall"] = np.arange(1, len(out) + 1, dtype=int)

    # Persist frequency CSVs if requested
    if cfg.save_frequencies:
        kw_freq_df.to_csv(f"{cfg.out_dir.rstrip('/')}/keyword_frequency.csv", index=False)
        ref_freq_df.to_csv(f"{cfg.out_dir.rstrip('/')}/reference_frequency.csv", index=False)

    return ScoringResult(
        df=out,
        keyword_freq=kw_freq_df,
        reference_freq=ref_freq_df,
        P=P,
        config=cfg
    )

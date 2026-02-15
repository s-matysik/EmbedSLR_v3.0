
"""
SMART-based MCDM module for EmbedSLR
====================================

Implements Simple Multi-Attribute Rating Technique (SMART)
for ranking candidate publications using four criteria:

1) semantic similarity (cosine similarity to the query)
2) topical similarity by authors' keywords
3) overlap of intellectual linkages (shared references)
4) mutual citations (bidirectional links within the seed/core set)

This follows the additive SMART model with weights derived
from importance ranks (the SMART "swing weights"). See Taherdoost & Mohebi (2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, List, Set
import math, json, re
import numpy as np
import pandas as pd

# ──────────────────────────────── Helpers ────────────────────────────────

KW_COLS = [
    "Author Keywords",
    "Authors Keywords",
    "Author Keywords Plus",
    "DE",
    "Index Keywords",
    "Keywords",
    "Author Keywords (DE)",
]

REF_COLS = [
    "References",
    "Cited References",
    "REF",
    "CR",
]

ID_COLS = [
    "DOI",
    "Identifier.Doi",
    "EID",
    "Scopus EID",
    "Unique ID",
    "Document ID",
    "UT",
    "Article Number",
    "Accession Number",
    "id",
]

TITLE_COLS = [
    "Article Title", "Title", "TI", "Document Title"
]


def _first_present(cols: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _norm_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_keywords(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    # split on ; , / | • and similar
    parts = re.split(r"[;\/\|,•]+", s)
    out = []
    for p in parts:
        p = _norm_text(p)
        p = re.sub(r"^[\s\-–—]+|[\s\-–—]+$", "", p)
        if p:
            out.append(p)
    return out


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


def _extract_dois(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return [m.group(0).lower() for m in _DOI_RE.finditer(s)]


def _hashish_title(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s.strip():
        return None
    s = _norm_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # take first 10 words
    words = s.split()[:10]
    if not words:
        return None
    return " ".join(words)


def _tokenize_references(s: str) -> Set[str]:
    """
    Returns a set of canonical tokens for references: DOIs if present,
    otherwise rough title hashes.
    """
    out: Set[str] = set()
    if not isinstance(s, str) or not s.strip():
        return out
    # Try DOIs first
    dois = _extract_dois(s)
    out.update(dois)
    # Split references heuristically and take a title-ish chunk
    if not dois:
        pieces = re.split(r"(?:(?:\.|;|\|\||\n)+\s*)", s)
        for p in pieces:
            t = _hashish_title(p)
            if t:
                out.add(t)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _safe_minmax(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    mn, mx = float(np.min(x)), float(np.max(x))
    if abs(mx - mn) < 1e-12:
        mx = mn + 1e-12
    return mn, mx


# ─────────────────────────────── SMART core ───────────────────────────────

@dataclass
class SMARTConfig:
    # Ranks on 4..10 (SMART scale) OR direct weights if provide_weights=True
    importance_ranks: Dict[str, int] = None  # e.g., {'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6}
    provide_weights: bool = False
    weights: Optional[Dict[str, float]] = None  # sum to 1 if provide_weights=True
    scale_4to10: bool = False  # if True, convert utilities u in [0,1] to g in [4,10]
    top_k_seed: int = 20       # number of top-by-cosine to define the "seed" profile for criteria 2..4
    # names of criteria
    c_semantic: str = "semantic"
    c_keywords: str = "keywords"
    c_references: str = "references"
    c_mutual: str = "mutual"


def _weights_from_ranks(ranks: Dict[str, int]) -> Dict[str, float]:
    """
    Taherdoost & Mohebi (2024), Eq. (7)-(8): w_j ∝ (√2)^(h_j), normalized to sum 1.
    h_j are ranks on 4..10.
    """
    base = math.sqrt(2.0)
    vals = {k: base ** v for k, v in ranks.items()}
    total = sum(vals.values())
    if total <= 0:
        # fallback equal
        n = len(ranks)
        return {k: 1.0 / n for k in ranks}
    return {k: v / total for k, v in vals.items()}


def _to_smart_scale(u: np.ndarray) -> np.ndarray:
    """
    Convert utilities u ∈ [0,1] to SMART's 7-point 4..10 scale (Table 2).
    Uses linear mapping: g = 4 + 6*u
    """
    return 4.0 + 6.0 * u


@dataclass
class SMARTResult:
    scores: pd.Series
    df: pd.DataFrame
    weights: Dict[str, float]
    utilities: pd.DataFrame
    contributions: pd.DataFrame


# ─────────────────────────────── Main pipeline ───────────────────────────────

def rank_with_smart(
    df: pd.DataFrame,
    query_vector: Optional[Sequence[float]] = None,
    doc_vectors: Optional[Sequence[Sequence[float]]] = None,
    config: Optional[SMARTConfig] = None,
    seed_keywords: Optional[Set[str]] = None,
    seed_core_idxs: Optional[Sequence[int]] = None,
    refs_column: Optional[str] = None,
    keywords_column: Optional[str] = None,
    doi_column: Optional[str] = None,
    title_column: Optional[str] = None,
) -> SMARTResult:
    """
    Compute SMART scores and sort df accordingly.

    Args:
        df: DataFrame with the candidate publications.
            Expected useful columns:
             - 'distance_cosine' or ('combined_embeddings' and query_vector)
             - a references column (heuristics used if not provided)
             - an authors' keywords column (heuristics used if not provided)
        query_vector: embedding of the query string (if we need to compute cosine)
        doc_vectors: embedding vectors for each row (if we need to compute cosine)
        config: SMART config (defaults to equal importance)
        seed_keywords: optional explicit set of target keywords
        seed_core_idxs: optional explicit indices for the core set (for criteria 2..4). If None,
                        we take top-K by semantic similarity.
        refs_column, keywords_column, doi_column, title_column: explicit column names (optional).

    Returns:
        SMARTResult with ranked df, weights, utilities, and contributions.
    """
    cfg = config or SMARTConfig(
        importance_ranks={
            "semantic": 7,
            "keywords": 7,
            "references": 7,
            "mutual": 7,
        },
        scale_4to10=False,
    )

    # ── determine columns ──
    kw_col = keywords_column or _first_present(KW_COLS, df)
    ref_col = refs_column or _first_present(REF_COLS, df)
    doi_col = doi_column or _first_present(ID_COLS, df)
    tit_col = title_column or _first_present(TITLE_COLS, df)

    # ── 1) semantic similarity ──
    if "distance_cosine" in df.columns:
        sim = 1.0 - df["distance_cosine"].astype(float).to_numpy()
    elif doc_vectors is not None and query_vector is not None:
        q = np.asarray(query_vector, dtype="float32")
        D = np.asarray(list(doc_vectors), dtype="float32")
        # vectorized cosine
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        sim = (D_norm @ q_norm.reshape(-1,1)).ravel()
    elif "combined_embeddings" in df.columns and query_vector is not None:
        # embeddings stored as JSON strings
        embs = df["combined_embeddings"].apply(lambda s: np.asarray(json.loads(s), dtype="float32"))
        D = np.stack(embs.values)
        q = np.asarray(query_vector, dtype="float32")
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        sim = (D_norm @ q_norm.reshape(-1,1)).ravel()
    else:
        raise ValueError("Need either 'distance_cosine' or (doc_vectors + query_vector) or ('combined_embeddings' + query_vector).")

    # normalize to [0,1]
    mn, mx = _safe_minmax(sim)
    u_semantic = (sim - mn) / (mx - mn)

    # ── build seed/core set ──
    if seed_core_idxs is None:
        top_k = min(cfg.top_k_seed, len(df))
        core_idxs = np.argsort(-u_semantic)[:top_k]  # top by similarity
    else:
        core_idxs = np.asarray(list(seed_core_idxs), dtype=int)

    # ── 2) keywords similarity ──
    # define seed keywords if not given: union over core set
    if seed_keywords is None:
        seed_kw: Set[str] = set()
        if kw_col:
            for i in core_idxs:
                kws = _parse_keywords(df.iloc[int(i)][kw_col])
                seed_kw.update(kws)
        else:
            seed_kw = set()
    else:
        seed_kw = { _norm_text(k) for k in seed_keywords }

    kw_scores = np.zeros(len(df), dtype="float32")
    if kw_col and seed_kw:
        for i, s in enumerate(df[kw_col].values):
            kws = set(_parse_keywords(s))
            inter = len(kws & seed_kw)
            union = len(kws | seed_kw) or 1
            kw_scores[i] = inter / union
    # else leave zeros
    mn, mx = _safe_minmax(kw_scores)
    u_keywords = (kw_scores - mn) / (mx - mn) if mx > mn else np.zeros_like(kw_scores)

    # ── 3) shared references (intellectual linkages) ──
    # seed reference multiset from core
    seed_refs: Set[str] = set()
    if ref_col:
        for i in core_idxs:
            rs = _tokenize_references(df.iloc[int(i)][ref_col])
            seed_refs.update(rs)
    ref_scores = np.zeros(len(df), dtype="float32")
    if ref_col and seed_refs:
        for i, s in enumerate(df[ref_col].values):
            rs = _tokenize_references(s)
            inter = len(rs & seed_refs)
            union = len(rs | seed_refs) or 1
            ref_scores[i] = inter / union
    mn, mx = _safe_minmax(ref_scores)
    u_references = (ref_scores - mn) / (mx - mn) if mx > mn else np.zeros_like(ref_scores)

    # ── 4) mutual citations ──
    # Build within-dataset citation adjacency via references resolving to DOIs or title-hashes
    # Map dataset items to canonical ids
    ids = None
    if doi_col and df[doi_col].notna().any():
        ids = [str(x).lower() if isinstance(x, str) else None for x in df[doi_col].values]
    else:
        # fallback on title hash
        if tit_col:
            ids = [_hashish_title(str(x)) if isinstance(x, str) else None for x in df[tit_col].values]
        else:
            ids = [None for _ in range(len(df))]

    id_to_idx: Dict[str, int] = {i: idx for idx, i in enumerate(ids) if i}
    cites = [set() for _ in range(len(df))]
    if ref_col:
        for i, s in enumerate(df[ref_col].values):
            rs = _tokenize_references(s)
            for r in rs:
                j = id_to_idx.get(r)
                if j is not None:
                    cites[i].add(j)

    # mutual with core set
    core_set = set(map(int, core_idxs))
    mutual_scores = np.zeros(len(df), dtype="float32")
    if len(core_set) > 0:
        denom = max(1, len(core_set))
        for i in range(len(df)):
            mutual_count = 0
            for j in cites[i]:
                if j in core_set and i in cites[j]:
                    mutual_count += 1
            mutual_scores[i] = mutual_count / denom  # fraction of core items with mutual links
        # re-normalize to [0,1]
        mn, mx = _safe_minmax(mutual_scores)
        u_mutual = (mutual_scores - mn) / (mx - mn) if mx > mn else np.zeros_like(mutual_scores)
    else:
        u_mutual = np.zeros(len(df), dtype="float32")

    # ── SMART aggregation ──
    if cfg.provide_weights and cfg.weights:
        weights = cfg.weights
    else:
        ranks = cfg.importance_ranks or {
            cfg.c_semantic: 7, cfg.c_keywords: 7, cfg.c_references: 7, cfg.c_mutual: 7
        }
        weights = _weights_from_ranks(ranks)

    # utilities matrix (0..1)
    U = pd.DataFrame({
        cfg.c_semantic: u_semantic,
        cfg.c_keywords: u_keywords,
        cfg.c_references: u_references,
        cfg.c_mutual: u_mutual,
    }, index=df.index)

    if cfg.scale_4to10:
        G = U.apply(_to_smart_scale)
        contrib = G.multiply(pd.Series(weights))
        scores = contrib.sum(axis=1)
    else:
        # directly aggregate utilities with weights
        contrib = U.multiply(pd.Series(weights))
        scores = contrib.sum(axis=1)

    out = df.copy()
    out["SMART_score"] = scores
    # sort descending by SMART_score
    out = out.sort_values("SMART_score", ascending=False)

    return SMARTResult(
        scores=scores,
        df=out,
        weights=weights,
        utilities=U,
        contributions=contrib
    )

# embedslr/ensemble.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Literal, Optional
import re, numpy as np, pandas as pd

from .embeddings import get_embeddings
from .similarity import rank_by_cosine

Agg = Literal["mean", "min", "median"]


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    @property
    def label(self) -> str:
        m = self.model.split("/")[-1]
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", m)
        return f"{self.provider}_{safe}"


# ------------------------
# Embedding precomputation
# ------------------------
def build_embeddings_cache(
    df: pd.DataFrame,
    combined_text_col: str,
    query: str,
    model_specs: List[ModelSpec],
    progress=None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute and cache embeddings for *all* selected models (once).
    Returns: mapping model_label -> (doc_embeddings, query_embedding).
    """
    texts = df[combined_text_col].astype(str).tolist()
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    iterator = model_specs
    if progress is not None:
        try:
            iterator = progress.tqdm(model_specs, desc="🔧 Pobieranie i inicjalizacja modeli / embeddings")
        except Exception:
            pass
    for ms in iterator:
        doc_vecs = get_embeddings(texts, provider=ms.provider, model=ms.model)
        q_vec    = get_embeddings([query], provider=ms.provider, model=ms.model)[0]
        cache[ms.label] = (np.asarray(doc_vecs), np.asarray(q_vec))
    return cache


def _rank_one_precomputed(
    df: pd.DataFrame,
    ms: ModelSpec,
    doc_vecs: np.ndarray,
    q_vec: np.ndarray,
) -> pd.DataFrame:
    """Return DataFrame with columns distance_{label}, rank_{label} (complete df length)."""
    ranked   = rank_by_cosine(q_vec, doc_vecs, df).reset_index(drop=False).rename(columns={"index":"_idx"})
    ranked[f"rank_{ms.label}"]     = np.arange(1, len(ranked) + 1, dtype=int)
    ranked[f"distance_{ms.label}"] = ranked["distance_cosine"].astype(float)
    ranked = ranked[["_idx", f"rank_{ms.label}", f"distance_{ms.label}"]]
    out = pd.DataFrame(index=df.index)
    out = out.join(ranked.set_index("_idx"), how="left")
    return out


def run_ensemble(
    df: pd.DataFrame,
    combined_text_col: str,
    query: str,
    model_specs: List[ModelSpec],
    *,
    top_k_per_model: int = 50,
    aggregator: Agg = "mean",
    precomputed: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Consensus ranking across multiple embedding models.

    If `precomputed` is provided (label -> (doc_embs, q_emb)), heavy embedding
    computation is *skipped* and those arrays are reused. This is critical when
    the same dataset is ranked for many combinations of the same models.
    """
    base = df.copy()

    if precomputed is None:
        # Fallback: compute on the fly (kept for backward compatibility)
        texts = df[combined_text_col].astype(str).tolist()

    for ms in model_specs:
        if precomputed is not None and ms.label in precomputed:
            doc_vecs, q_vec = precomputed[ms.label]
            part = _rank_one_precomputed(df, ms, doc_vecs, q_vec)
        else:
            doc_vecs = get_embeddings(texts, provider=ms.provider, model=ms.model)
            q_vec    = get_embeddings([query], provider=ms.provider, model=ms.model)[0]
            part = _rank_one_precomputed(df, ms, np.asarray(doc_vecs), np.asarray(q_vec))

        base = base.join(part)

    rank_cols     = [c for c in base.columns if c.startswith("rank_")]
    distance_cols = [c for c in base.columns if c.startswith("distance_")]

    # Determine which rows are in top-K for which models
    hit_mask = np.zeros((len(base), len(rank_cols)), dtype=bool)
    for j, c in enumerate(rank_cols):
        vals = base[c].values
        hit_mask[:, j] = (vals <= top_k_per_model)

    base["hit_count"]  = hit_mask.sum(axis=1).astype(int)
    base["hit_models"] = [
        ";".join([rank_cols[j].replace("rank_", "") for j, ok in enumerate(row) if ok])
        for row in hit_mask
    ]

    # Aggregate distances/ranks across the subset of models that voted (top-K)
    dist_arr = np.column_stack([base[c].values for c in distance_cols])
    rank_arr = np.column_stack([base[c].values for c in rank_cols])
    dist_arr = np.where(hit_mask, dist_arr, np.nan)
    rank_arr = np.where(hit_mask, rank_arr, np.nan)

    if aggregator == "mean":
        agg_dist = np.nanmean(dist_arr, axis=1)
    elif aggregator == "min":
        agg_dist = np.nanmin(dist_arr, axis=1)
    else:
        agg_dist = np.nanmedian(dist_arr, axis=1)

    base["agg_distance"] = agg_dist
    base["mean_rank"]    = np.nanmean(rank_arr, axis=1)

    base = base.sort_values(
        by=["hit_count", "agg_distance", "mean_rank"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    return base


def per_group_bibliometrics(ranked_df: pd.DataFrame, groups=(5,4,3,2,1)) -> pd.DataFrame:
    """
    Compute bibliometric indicators A, A′, B, B′ per hit_count group.
    Falls back to NaN if bibliometrics module isn't available.
    """
    try:
        from .bibliometrics import indicator_a, indicator_a_prime, indicator_b, indicator_b_prime
    except Exception:
        import numpy as _np
        rows = [{"group": g, "n": int((ranked_df["hit_count"]==g).sum()),
                 "A": _np.nan, "A′": _np.nan, "B": _np.nan, "B′": _np.nan} for g in groups]
        return pd.DataFrame(rows)

    rows = []
    for k in groups:
        g = ranked_df[ranked_df["hit_count"] == k]
        if g.empty:
            rows.append({"group": k, "n": 0, "A": np.nan, "A′": np.nan, "B": np.nan, "B′": np.nan})
            continue
        A   = indicator_a(g); Ap = indicator_a_prime(g)
        B   = indicator_b(g); Bp = indicator_b_prime(g)
        rows.append({"group": k, "n": len(g), "A": A, "A′": Ap, "B": B, "B′": Bp})
    return pd.DataFrame(rows)

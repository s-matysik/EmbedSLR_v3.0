from __future__ import annotations

"""
Configuration objects for advanced scoring in EmbedSLR.

You can instantiate ScoringConfig programmatically or load it from a dict.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal


Criterion = Literal["semantic", "keywords", "references", "citations"]


@dataclass
class ColumnMap:
    """
    Maps semantic/keyword/reference/citations to column names present in your DataFrame.
    We perform light autodetection if unspecified.
    """
    # semantic similarity (prefer "semantic_similarity" in [0,1]; otherwise provide "distance_cosine")
    semantic_similarity: Optional[str] = None
    distance_cosine: Optional[str] = None

    # metadata columns
    keywords: Optional[str] = None
    references: Optional[str] = None
    citations: Optional[str] = None

    # Optional identifier/title to keep in outputs
    id_col: Optional[str] = None
    title_col: Optional[str] = None


@dataclass
class ScoringConfig:
    method: Literal["linear", "zscore", "linear_plus"] = "linear_plus"

    # How many top tokens to use when summing per-item contributions
    top_keywords: int = 5         # sk
    top_references: int = 15      # b

    # Penalties applied when item has no tokens in a category (expressed as 0.10 == 10%)
    penalty_no_keywords: float = 0.10   # ksk
    penalty_no_references: float = 0.10 # kr

    # Criteria weights for aggregation (sum does not have to be 1; we will normalize).
    # Only criteria that exist in the data are used.
    weights: Dict[Criterion, float] = field(default_factory=lambda: {
        "semantic": 0.40, "keywords": 0.25, "references": 0.25, "citations": 0.10
    })

    # Outlier bonus (used by method == "linear_plus")
    bonus_start_z: float = 2.0   # start awarding at >= 2 sd from median
    bonus_full_z: float  = 4.0   # full bonus at >= 4 sd from median
    bonus_cap_points: Optional[float] = None  # default == P (#items)

    # Column mapping / autodetection hints
    columns: ColumnMap = field(default_factory=ColumnMap)

    # Which criteria to use (order does not matter). Defaults to all.
    use_criteria: Optional[List[Criterion]] = None

    # Whether higher values in a criterion are better. We auto-set for "distance_cosine".
    higher_is_better: Dict[Criterion, bool] = field(default_factory=lambda: {
        "semantic": True, "keywords": True, "references": True, "citations": True
    })

    # Persist auxiliary CSVs with token frequencies
    save_frequencies: bool = False
    out_dir: str = "."

    # Deterministic tie handling
    tie_method: Literal["average", "min", "max", "dense", "ordinal"] = "average"

    def normalized_weights(self, active: List[Criterion]) -> Dict[Criterion, float]:
        w = {k: v for k, v in self.weights.items() if k in active}
        s = sum(w.values()) or 1.0
        return {k: v/s for k, v in w.items()}

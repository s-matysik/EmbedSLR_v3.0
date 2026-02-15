from importlib import metadata as _m

# -- Lazy imports --
# Heavy dependencies (sentence-transformers, openai, cohere, gradio)
# are loaded only when the corresponding functions are first called.
# This allows lightweight modules (config, advanced_scoring,
# mcda_validation) to be imported without installing GPU/API libs.

def __getattr__(name):
    _lazy = {
        "get_embeddings": (".embeddings", "get_embeddings"),
        "list_models":    (".embeddings", "list_models"),
        "rank_by_cosine": (".similarity", "rank_by_cosine"),
        "full_report":    (".bibliometrics", "full_report"),
        "indicators":     (".bibliometrics", "indicators"),
        "colab_run":      (".colab_app", "run"),
    }
    if name in _lazy:
        mod_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(mod_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# -- Eager (lightweight) imports --
from .advanced_scoring import rank_with_advanced_scoring
from .config import ScoringConfig, ColumnMap
from .mcda_validation import (
    weight_sensitivity_oat,
    weight_sensitivity_scenarios,
    criteria_removal_analysis,
    cross_method_correlation,
    parameter_sensitivity,
    precision_recall_at_k,
    bootstrap_stability,
    monte_carlo_weights,
    rank_reversal_analysis,
    normalization_comparison,
    compromise_ranking,
    generate_validation_report,
)

try:
    __version__ = _m.version(__name__)
except _m.PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = [
    "get_embeddings", "list_models",
    "rank_by_cosine",
    "full_report", "indicators",
    "colab_run",
    "rank_with_advanced_scoring", "ScoringConfig", "ColumnMap",
    "weight_sensitivity_oat", "weight_sensitivity_scenarios",
    "criteria_removal_analysis", "cross_method_correlation",
    "parameter_sensitivity", "precision_recall_at_k",
    "bootstrap_stability", "monte_carlo_weights",
    "rank_reversal_analysis", "normalization_comparison",
    "compromise_ranking", "generate_validation_report",
]

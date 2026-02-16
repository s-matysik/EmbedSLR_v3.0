#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_validation.py — LLM-based validation for EmbedSLR v3.0
============================================================
Evaluates top-ranked articles against the research problem using
multiple LLM providers. Each LLM scores articles on:
  - semantic_relevance (1-10): how closely the article relates to the research problem
  - research_importance (1-10): scientific value based on MCDA parameters
  - classification: GAP / EXPLORED / LOW IMPORTANCE / METHOD / NOISE

Supports: OpenAI, Anthropic, Google, DeepSeek, xAI, Kimi (Moonshot)
All providers use OpenAI-compatible interface except Anthropic and Google.
"""
from __future__ import annotations

import json
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd

# =====================================================================
#  PROMPT TEMPLATE
# =====================================================================

ARTICLE_EVAL_PROMPT = """Role: You are a senior academic researcher with 20+ years of experience in systematic literature reviews and evaluating research relevance.

Research Problem: {research_problem}

Task: Evaluate each article below for its relevance to the above research problem and its scientific importance. Consider the article's title, abstract (if available), and its MCDA ranking parameters.

EVALUATION FRAMEWORK:

1. Semantic Relevance (1-10):
   - 1-2: Article has minimal or no connection to the research problem
   - 3-5: Partial overlap — covers related but tangential topics
   - 6-8: Strong thematic alignment with the research problem
   - 9-10: Directly addresses core aspects of the research problem

2. Research Importance (1-10):
   - 1-2: Low scientific value — trivial, redundant, or methodologically weak
   - 3-5: Moderate value — contributes incrementally to the field
   - 6-8: High value — addresses meaningful questions with solid approach
   - 9-10: Critical contribution — fills a genuine gap or provides breakthrough insight

Articles to evaluate (format: rank, title, semantic_similarity, keyword_score, reference_score, citations):
{articles_block}

Output Format: Return ONLY a JSON array, one object per article:
[
  {{
    "rank": int,
    "title": "string",
    "semantic_relevance": int,
    "research_importance": int
  }}
]
"""


# =====================================================================
#  PROVIDER FUNCTIONS
# =====================================================================

def _parse_json_response(text: str) -> List[Dict]:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # Sometimes there's a trailing ```
    if "```" in text:
        text = text[:text.index("```")]
    return json.loads(text.strip())


def _call_openai_compatible(client, model: str, prompt: str,
                            provider_name: str) -> List[Dict]:
    """Generic OpenAI-compatible API call."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4000,
    )
    text = response.choices[0].message.content.strip()
    return _parse_json_response(text)


def _call_anthropic(client, prompt: str) -> List[Dict]:
    """Anthropic API call."""
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    return _parse_json_response(text)


def _call_google(client, prompt: str) -> List[Dict]:
    """Google Gemini API call."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.0},
    )
    text = response.text.strip()
    return _parse_json_response(text)


# =====================================================================
#  MAIN EVALUATOR
# =====================================================================

@dataclass
class LLMValidationResult:
    """Results from LLM-based article evaluation."""
    per_model: Dict[str, pd.DataFrame] = field(default_factory=dict)
    consensus: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = field(default_factory=dict)


def _build_articles_block(ranked_df: pd.DataFrame, prep: pd.DataFrame,
                          top_k: int = 20) -> str:
    """Build text block describing top-K articles for LLM evaluation."""
    title_col = next((c for c in ["Title", "Article Title", "Document Title"]
                      if c in ranked_df.columns), ranked_df.columns[0])
    abs_col = next((c for c in ["Abstract", "Description", "abstract"]
                    if c in ranked_df.columns), None)

    lines = []
    for i in range(min(top_k, len(ranked_df))):
        row = ranked_df.iloc[i]
        title = str(row[title_col])[:120]
        abstract = ""
        if abs_col and pd.notna(row.get(abs_col)):
            abstract = str(row[abs_col])[:200] + "..."

        # Get MCDA parameters from prep if available
        sem = f"{prep.iloc[i].get('_sem', 0):.3f}" if '_sem' in prep.columns else "N/A"
        kw = f"{prep.iloc[i].get('_kw', 0):.1f}" if '_kw' in prep.columns else "N/A"
        ref = f"{prep.iloc[i].get('_ref', 0):.1f}" if '_ref' in prep.columns else "N/A"
        cit = f"{prep.iloc[i].get('_cit', 0):.0f}" if '_cit' in prep.columns else "N/A"

        line = f"{i+1}. \"{title}\""
        if abstract:
            line += f"\n   Abstract: {abstract}"
        line += f"\n   [semantic={sem}, keywords={kw}, references={ref}, citations={cit}]"
        lines.append(line)

    return "\n\n".join(lines)


def run_llm_validation(
    ranked_df: pd.DataFrame,
    prep: pd.DataFrame,
    research_problem: str,
    api_keys: Dict[str, str],
    top_k: int = 20,
    batch_size: int = 20,
    log_fn=None,
) -> LLMValidationResult:
    """
    Evaluate top-K articles via multiple LLM providers.

    Parameters
    ----------
    ranked_df : DataFrame with ranked articles (from Step 1)
    prep : DataFrame with precomputed MCDA values (_sem, _kw, _ref, _cit)
    research_problem : str — the research problem text
    api_keys : dict with keys like 'openai', 'anthropic', 'google', 'deepseek', 'xai', 'kimi'
    top_k : number of top articles to evaluate
    batch_size : articles per LLM call
    log_fn : optional logging function

    Returns
    -------
    LLMValidationResult with per-model DataFrames and consensus
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    if not research_problem or not research_problem.strip():
        _log("⚠ No research problem provided — LLM validation skipped.")
        return LLMValidationResult(summary={"error": "no_research_problem"})

    articles_block = _build_articles_block(ranked_df, prep, top_k)
    prompt = ARTICLE_EVAL_PROMPT.format(
        research_problem=research_problem,
        articles_block=articles_block,
    )

    # Build provider callables
    providers = {}

    if api_keys.get("openai"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["openai"])
            providers["OpenAI"] = lambda p=prompt, c=client: _call_openai_compatible(
                c, "gpt-4o-mini", p, "OpenAI")
        except ImportError:
            _log("⚠ openai package not installed — skipping OpenAI")

    if api_keys.get("anthropic"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_keys["anthropic"])
            providers["Anthropic"] = lambda p=prompt, c=client: _call_anthropic(c, p)
        except ImportError:
            _log("⚠ anthropic package not installed — skipping Anthropic")

    if api_keys.get("google"):
        try:
            from google import genai
            client = genai.Client(api_key=api_keys["google"])
            providers["Google"] = lambda p=prompt, c=client: _call_google(c, p)
        except ImportError:
            _log("⚠ google-genai package not installed — skipping Google")

    if api_keys.get("deepseek"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["deepseek"],
                          base_url="https://api.deepseek.com")
            providers["DeepSeek"] = lambda p=prompt, c=client: _call_openai_compatible(
                c, "deepseek-chat", p, "DeepSeek")
        except ImportError:
            pass

    if api_keys.get("xai"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["xai"],
                          base_url="https://api.x.ai/v1")
            providers["xAI"] = lambda p=prompt, c=client: _call_openai_compatible(
                c, "grok-3-mini-fast", p, "xAI")
        except ImportError:
            pass

    if api_keys.get("kimi"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["kimi"],
                          base_url="https://api.moonshot.ai/v1")
            providers["Kimi"] = lambda p=prompt, c=client: _call_openai_compatible(
                c, "kimi-k2-turbo-preview", p, "Kimi")
        except ImportError:
            pass

    if not providers:
        _log("⚠ No LLM API keys provided — LLM validation skipped.")
        return LLMValidationResult(summary={"error": "no_api_keys"})

    _log(f"LLM validation: {len(providers)} providers, top-{top_k} articles...")

    # Run providers in parallel
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(providers))) as executor:
        future_to_name = {
            executor.submit(fn): name
            for name, fn in providers.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                data = future.result()
                results_dict[name] = pd.DataFrame(data)
                _log(f"  [{name}] ✓ {len(data)} articles evaluated")
            except Exception as e:
                _log(f"  [{name}] ✗ Error: {e}")

    if not results_dict:
        _log("⚠ All LLM providers failed.")
        return LLMValidationResult(summary={"error": "all_providers_failed"})

    # Build consensus
    result = LLMValidationResult(per_model=results_dict)

    # Compute consensus scores
    all_relevance = []
    all_importance = []

    for name, df in results_dict.items():
        if "semantic_relevance" in df.columns:
            all_relevance.append(df.set_index("rank")["semantic_relevance"].rename(name))
        if "research_importance" in df.columns:
            all_importance.append(df.set_index("rank")["research_importance"].rename(name))

    if all_relevance:
        rel_df = pd.concat(all_relevance, axis=1)
        imp_df = pd.concat(all_importance, axis=1) if all_importance else rel_df * 0

        consensus = pd.DataFrame({
            "rank": rel_df.index,
            "mean_relevance": rel_df.mean(axis=1).values,
            "std_relevance": rel_df.std(axis=1).values,
            "mean_importance": imp_df.mean(axis=1).values,
            "std_importance": imp_df.std(axis=1).values,
            "n_providers": rel_df.notna().sum(axis=1).values,
        })

        result.consensus = consensus.reset_index(drop=True)

        result.summary = {
            "n_articles": len(consensus),
            "n_providers": len(results_dict),
            "providers": list(results_dict.keys()),
            "mean_relevance": float(consensus["mean_relevance"].mean()),
            "mean_importance": float(consensus["mean_importance"].mean()),
        }

        _log(f"  Consensus: mean_relevance={result.summary['mean_relevance']:.2f}, "
             f"mean_importance={result.summary['mean_importance']:.2f}")

    return result


def format_llm_report(result: LLMValidationResult) -> str:
    """Generate text report section for LLM validation."""
    lines = ["\n-- TEST 5b: LLM Validation --"]

    if result.summary.get("error"):
        lines.append(f"  Skipped: {result.summary['error']}")
        return "\n".join(lines)

    s = result.summary
    lines.append(f"  Providers: {', '.join(s.get('providers', []))}")
    lines.append(f"  Articles evaluated: {s.get('n_articles', 0)}")
    lines.append(f"  Mean semantic relevance: {s.get('mean_relevance', 0):.2f}")
    lines.append(f"  Mean research importance: {s.get('mean_importance', 0):.2f}")

    if result.consensus is not None and len(result.consensus) > 0:
        lines.append("\n  Top articles by consensus relevance:")
        top = result.consensus.sort_values("mean_relevance", ascending=False).head(10)
        for _, r in top.iterrows():
            lines.append(
                f"    #{int(r['rank']):>3d}  rel={r['mean_relevance']:.1f}±{r['std_relevance']:.1f}  "
                f"imp={r['mean_importance']:.1f}±{r['std_importance']:.1f}"
            )

    return "\n".join(lines)

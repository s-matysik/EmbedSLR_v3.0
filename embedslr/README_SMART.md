
# SMART (bibliometry) re-ranking module

**Cel:** Re-ranking publikacji z użyciem metody **SMART** na czterech kryteriach:
1) podobieństwo semantyczne (z gotowej kolumny `*similarity*` albo odwrócona `*distance*`),
2) podobieństwo tematyczne po słowach kluczowych autorów (gotowa kolumna, np. `kw_similarity`),
3) zbieżność połączeń intelektualnych – np. **bibliographic coupling** / wspólne referencje (gotowa kolumna),
4) wzajemne cytowania (gotowa kolumna).

> Moduł **nie przelicza** mierników bibliometrycznych od zera. Wykorzystuje tylko to, co jest już w danych i **jedynie normalizuje** do [0,1], a potem agreguje metodą SMART.

### Instalacja / włączenie w Colab
```python
# w Colab: wgraj plik smart_mcdm_biblio.py do /content/
import sys
sys.path.append('/content')
from smart_mcdm_biblio import SMARTConfig, rank_with_smart_biblio, quick_diagnose
```

### Minimalny przykład
```python
import pandas as pd
from smart_mcdm_biblio import SMARTConfig, rank_with_smart_biblio

df = pd.read_csv("ranked.csv")  # Twoje dane z istniejącymi kolumnami

cfg = SMARTConfig(
    column_map={
        # ustaw tylko jeśli nazwy w Twoim pliku są specyficzne
        # 'semantic': 'cosine_distance',       # zostanie zinterpretowane jako dystans -> invert
        # 'keywords': 'kw_similarity',
        # 'references': 'bibliographic_coupling',
        # 'mutual': 'mutual_citations'
    },
    # wariant A: wagi bezpośrednio (zostaną znormalizowane)
    # explicit_weights={'semantic': 0.4, 'keywords': 0.25, 'references': 0.2, 'mutual': 0.15},

    # wariant B: rangi 4–10 (domyślnie jak poniżej); wagi liczone jako (sqrt(2))**h i normalizowane
    importance_ranks={'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6},

    scale_4to10=False,          # jeżeli chcesz agregować na skali 4–10 (g_ij), ustaw True
    available_only=True,        # brakujące kryteria są pomijane, a wagi przeskalowane
    normalize_strategy="minmax" # 'minmax' albo 'max'
)

res = rank_with_smart_biblio(df, cfg, top_n=50)
res.df.to_csv("ranked_smart.csv", index=False)

print("Wagi:", res.weights)
print("Użyte kolumny:", res.used_columns)
print(res.utilities.head())
```

### Interaktywna zmiana wag (Colab)
```python
!pip -q install ipywidgets
from ipywidgets import interact, IntSlider
from smart_mcdm_biblio import SMARTConfig, rank_with_smart_biblio

def rerank(sem=8, kw=7, ref=7, mut=6):
    cfg = SMARTConfig(importance_ranks={'semantic': sem, 'keywords': kw, 'references': ref, 'mutual': mut})
    res = rank_with_smart_biblio(df, cfg, top_n=20)
    display(res.df[['SMART_score'] + [c for c in df.columns if c.lower() in ('title','doc_title','article_title','doi')]].head(20))
    print("Wagi:", res.weights, "| Kolumny:", res.used_columns)

interact(
    rerank,
    sem=IntSlider(min=4, max=10, step=1, value=8, description='semantic'),
    kw=IntSlider(min=4, max=10, step=1, value=7, description='keywords'),
    ref=IntSlider(min=4, max=10, step=1, value=7, description='references'),
    mut=IntSlider(min=4, max=10, step=1, value=6, description='mutual')
);
```

### Mapowanie nazw kolumn
Moduł potrafi sam wykryć typowe nazwy. Jeśli Twoje kolumny mają inne nazwy – podaj je jawnie w `SMARTConfig.column_map`:
- **semantic**: `semantic_similarity`, `cosine_similarity`, `similarity`, `cos_sim`, `distance_cosine`, `cosine_distance`, `semantic_dist`…  
  (jeśli nazwa zawiera `distance`, moduł automatycznie odwróci skalę).
- **keywords**: `kw_similarity`, `author_keywords_similarity`, `kw_jaccard`, `keyword_overlap_score`…
- **references**: `bibliographic_coupling`, `biblio_coupling`, `bc_score`, `common_references`, `co_citation_score`…
- **mutual**: `mutual_citations`, `reciprocal_citations`, `two_way_citations`, `mutual_citations_count`…

### Diagnoza problemów
```python
from smart_mcdm_biblio import quick_diagnose
quick_diagnose(df)  # pokaże, które kryteria zostały rozpoznane i podstawowe statystyki kolumn
```

### Jak działają wagi – SMART
- Wagi z rang: \( w_j \propto (\sqrt{2})^{h_j} \), następnie normalizacja do sumy 1 (wzory (7)–(8)).
- Agregacja: \( f_i = \sum_j w_j \, u_{ij} \) (wzór (9)).
- Opcjonalnie można użyć skali 4–10: \( g_{ij} = 4 + 6\,u_{ij} \) i sumować \( \sum_j w_j g_{ij} \).

Źródło metody i kroków procesu SMART – patrz **rysunek 2** (str. 6) oraz wzory (7)–(9) w załączonym artykule.

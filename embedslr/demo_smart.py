
# === demo_smart.py ===
# Minimalny przykład uruchomienia modułu SMART w trybie skryptowym.
import pandas as pd
from smart_mcdm_biblio import SMARTConfig, rank_with_smart_biblio, quick_diagnose

def main():
    # Zmień ścieżkę do swojego CSV
    csv_path = "ranked.csv"
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
    except Exception:
        # ostateczny fallback
        df = pd.read_csv(csv_path, encoding="ISO-8859-1", on_bad_lines="skip", low_memory=False)

    print("Kolumny:", list(df.columns)[:50])

    cfg = SMARTConfig(
        # Podaj swoje nazwy kolumn, jeśli automatyczne nie zadziała
        # column_map={'semantic': 'cosine_distance', 'keywords':'kw_similarity',
        #             'references':'bibliographic_coupling', 'mutual':'mutual_citations'},
        importance_ranks={'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6},
        available_only=True,
    )

    quick_diagnose(df, cfg)
    res = rank_with_smart_biblio(df, cfg, top_n=100)
    res.df.to_csv("ranked_smart.csv", index=False)
    print("Wagi:", res.weights)
    print("Użyte kolumny:", res.used_columns)
    print(res.df.head(5))

if __name__ == "__main__":
    main()

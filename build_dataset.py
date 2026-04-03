"""
Constrói um dataset unificado a partir de 3 fontes:
  1. Hello-SimpleAI/HC3            (HuggingFace)
  2. andythetechnerd03/AI-human-text (HuggingFace)
  3. AI_Human.csv.zip               (arquivo local)

Schema final (Parquet + CSV):
  text           : str   — texto bruto
  label          : int   — 0 = humano | 1 = IA/GPT
  source_dataset : str   — 'hc3' | 'ai_human_hf' | 'ai_human_csv'
  domain         : str   — domínio do HC3 (reddit_eli5, …) ou 'unknown'

Uso:
    python build_dataset.py               # todas as fontes, sem limite
    python build_dataset.py --no-hf       # pula downloads do HuggingFace
    python build_dataset.py -n 50000      # limita cada fonte a N amostras
"""
import argparse
import io
import pathlib
import zipfile

import pandas as pd
from datasets import load_dataset

MIN_CHARS  = 50
DATA_DIR   = pathlib.Path("data")
OUT_PARQUET = DATA_DIR / "unified_dataset.parquet"
OUT_CSV     = DATA_DIR / "unified_dataset.csv"
ZIP_PATH    = pathlib.Path("AI_Human.csv.zip")


# ── Loaders ───────────────────────────────────────────────────────────────────

RANDOM_SEED = 42


def _sample_balanced(df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    """Amostra n_per_class exemplos de cada label (shuffle, sem reposição)."""
    return (df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(min(n_per_class, len(x)),
                                        random_state=RANDOM_SEED))
              .reset_index(drop=True))


def load_hc3() -> pd.DataFrame:
    print("\n[1/3] Carregando HC3 (HuggingFace) — dataset de referência...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
        split="train",
    )
    rows = []
    for sample in ds:
        human  = (sample.get("human_answers") or [""])[0]
        gpt    = (sample.get("chatgpt_answers") or [""])[0]
        domain = sample.get("source", "unknown")
        if len(human) >= MIN_CHARS:
            rows.append({"text": human, "label": 0,
                         "source_dataset": "hc3", "domain": domain})
        if len(gpt) >= MIN_CHARS:
            rows.append({"text": gpt, "label": 1,
                         "source_dataset": "hc3", "domain": domain})

    df = pd.DataFrame(rows)
    # Balanceia o próprio HC3 pelo menor lado
    n_per_class = min((df.label == 0).sum(), (df.label == 1).sum())
    df = _sample_balanced(df, n_per_class)
    print(f"  HC3: {len(df):,} linhas  "
          f"(humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    return df


def load_ai_human_hf(n_per_class: int) -> pd.DataFrame:
    print(f"\n[2/3] Carregando andythetechnerd03/AI-human-text "
          f"(HuggingFace) — {n_per_class:,} por classe...")
    ds = load_dataset("andythetechnerd03/AI-human-text", split="train")
    df = pd.DataFrame({"text": ds["text"], "label": [int(v) for v in ds["generated"]]})
    df = df[df["text"].str.len() >= MIN_CHARS].copy()
    df["source_dataset"] = "ai_human_hf"
    df["domain"]         = "unknown"
    df = _sample_balanced(df, n_per_class)
    print(f"  AI-human-hf: {len(df):,} linhas  "
          f"(humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    return df


def load_ai_human_csv(n_per_class: int) -> pd.DataFrame:
    print(f"\n[3/3] Carregando AI_Human.csv.zip (local) "
          f"— {n_per_class:,} por classe...")
    if not ZIP_PATH.exists():
        print(f"  AVISO: {ZIP_PATH} não encontrado — fonte ignorada.")
        return pd.DataFrame(columns=["text", "label", "source_dataset", "domain"])

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
        with zf.open(csv_name) as f:
            df_raw = pd.read_csv(
                io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            )

    df = pd.DataFrame({
        "text":  df_raw["text"],
        "label": df_raw["generated"].astype(int),
    })
    df = df[df["text"].str.len() >= MIN_CHARS].copy()
    df["source_dataset"] = "ai_human_csv"
    df["domain"]         = "unknown"
    df = _sample_balanced(df, n_per_class)
    print(f"  AI-human-csv: {len(df):,} linhas  "
          f"(humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build(no_hf: bool) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # HC3 carregado primeiro — define o tamanho de referência
    df_hc3 = load_hc3()
    n_per_class = (df_hc3.label == 0).sum()   # HC3 já está balanceado
    print(f"\nReferência: {n_per_class:,} amostras por classe por fonte.")

    parts = [df_hc3]

    if not no_hf:
        parts.append(load_ai_human_hf(n_per_class))
    else:
        print("Pulando fontes HuggingFace (--no-hf).")

    parts.append(load_ai_human_csv(n_per_class))

    df = pd.concat(parts, ignore_index=True)

    # ── Deduplicação por texto exato ──────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    after  = len(df)
    print(f"\nDeduplicação: {before:,} → {after:,}  ({before - after:,} duplicatas removidas)")

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMO DO DATASET UNIFICADO")
    print("=" * 60)
    print(f"\nTotal de amostras : {len(df):,}")
    print(f"  label=0 (humano): {(df.label==0).sum():,}")
    print(f"  label=1 (IA/GPT): {(df.label==1).sum():,}")
    print(f"\nPor fonte:")
    for src, grp in df.groupby("source_dataset"):
        h = (grp.label==0).sum()
        a = (grp.label==1).sum()
        print(f"  {src:20s}: {len(grp):7,}  (humano={h:,} | IA={a:,})")
    print(f"\nPor domínio (top 10):")
    top = df.groupby("domain").size().sort_values(ascending=False).head(10)
    for dom, cnt in top.items():
        print(f"  {dom:35s}: {cnt:,}")
    lens = df["text"].str.len()
    print(f"\nComprimento de texto — min:{lens.min()} / média:{lens.mean():.0f} / max:{lens.max()}")

    # ── Salva ─────────────────────────────────────────────────────────────────
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSalvo em:")
    print(f"  {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {OUT_CSV}      ({OUT_CSV.stat().st_size / 1024 / 1024:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrói dataset unificado")
    parser.add_argument("--no-hf", action="store_true",
                        help="Pula downloads do HuggingFace")
    args = parser.parse_args()
    build(args.no_hf)

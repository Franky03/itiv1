"""
Exploração do dataset AI_Human.csv.zip (arquivo local).
Mostra estrutura, colunas, exemplos e distribuições.
"""
import zipfile
import pathlib
import io
import pandas as pd
from collections import Counter

ZIP_PATH = pathlib.Path(__file__).parent / "AI_Human.csv.zip"

print("=" * 60)
print("DATASET: AI_Human.csv.zip (local)")
print("=" * 60)

# ── Inspeciona o ZIP ──────────────────────────────────────────────────────────
print(f"\nArquivo: {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = zf.namelist()
    print(f"Arquivos dentro do ZIP: {names}")

    # Carrega o primeiro CSV encontrado
    csv_name = next(n for n in names if n.endswith('.csv'))
    print(f"\nCarregando: {csv_name}")
    with zf.open(csv_name) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8', errors='replace'))

print(f"\nShape: {df.shape}  ({df.shape[0]} linhas × {df.shape[1]} colunas)")
print(f"Colunas: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")

print("\n--- Exemplo (primeiras 3 linhas) ---")
for i, row in df.head(3).iterrows():
    print(f"\n  [idx={i}]")
    for col, val in row.items():
        print(f"    {col}: {repr(str(val))[:120]}")

print("\n--- Valores nulos ---")
print(df.isnull().sum().to_string())

print("\n--- Valores únicos por coluna categórica ---")
for col in df.columns:
    n_unique = df[col].nunique()
    if n_unique <= 20:
        cnt = Counter(str(v) for v in df[col])
        print(f"\n  {col}  ({n_unique} únicos):")
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"    {repr(k):30s}: {v:6d}")
    else:
        print(f"\n  {col}: {n_unique} valores únicos")
        print(f"    exemplos: {list(df[col].dropna().unique()[:5])}")

print("\n--- Comprimentos de texto por coluna string ---")
for col in df.columns:
    if df[col].dtype == object:
        lens = df[col].dropna().str.len()
        if lens.mean() > 20:
            print(f"  {col} — min:{lens.min()} / média:{lens.mean():.0f} / max:{lens.max()}")

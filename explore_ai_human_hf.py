"""
Exploração do dataset andythetechnerd03/AI-human-text (Hugging Face).
Mostra estrutura, colunas, exemplos e distribuições.
"""
from datasets import load_dataset
from collections import Counter

print("=" * 60)
print("DATASET: andythetechnerd03/AI-human-text")
print("=" * 60)

ds = load_dataset("andythetechnerd03/AI-human-text", split="train")

print(f"\nTotal de registros: {len(ds)}")
print(f"Colunas: {ds.column_names}")
print(f"Features: {ds.features}")

print("\n--- Exemplo (índice 0) ---")
sample = ds[0]
for k, v in sample.items():
    print(f"  {k}: {repr(str(v))[:120]}")

print("\n--- Mais exemplos ---")
for i in [1, 2, 100, 1000]:
    s = ds[i]
    print(f"\n  [idx={i}]")
    for k, v in s.items():
        print(f"    {k}: {repr(str(v))[:100]}")

print("\n--- Valores únicos por coluna categórica ---")
for col in ds.column_names:
    vals = ds[col]
    unique = set(str(v) for v in vals)
    if len(unique) <= 20:
        print(f"  {col}: {sorted(unique)}")
    else:
        print(f"  {col}: {len(unique)} valores únicos")
        # mostra os primeiros 5
        sample_vals = list(unique)[:5]
        print(f"    exemplos: {sample_vals}")

print("\n--- Distribuição da coluna de label (se existir) ---")
for col in ds.column_names:
    vals = ds[col]
    unique = set(str(v) for v in vals)
    if len(unique) <= 10:
        cnt = Counter(str(v) for v in vals)
        print(f"\n  {col}:")
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"    {repr(k):30s}: {v:6d}")

print("\n--- Comprimentos de texto ---")
for col in ds.column_names:
    sample_val = ds[0][col]
    if isinstance(sample_val, str) and len(sample_val) > 20:
        lens = [len(str(r[col])) for r in ds]
        print(f"  {col} — min:{min(lens)} / média:{sum(lens)//len(lens)} / max:{max(lens)}")

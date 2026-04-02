"""
Exploração do dataset HC3 (Hello-SimpleAI/HC3).
Mostra estrutura, colunas, exemplos e distribuições.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from datasets import load_dataset
from collections import Counter

print("=" * 60)
print("DATASET: Hello-SimpleAI/HC3")
print("=" * 60)

ds = load_dataset(
    "json",
    data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
    split="train",
)

print(f"\nTotal de registros: {len(ds)}")
print(f"Colunas: {ds.column_names}")
print(f"Features: {ds.features}")

print("\n--- Exemplo (índice 0) ---")
sample = ds[0]
for k, v in sample.items():
    val = v[:2] if isinstance(v, list) else v
    print(f"  {k}: {repr(val)[:120]}")

print("\n--- Distribuição por source ---")
sources = Counter(ds['source'])
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {src:35s}: {cnt:5d} registros")

print("\n--- Estrutura das respostas ---")
n_multi_human = sum(1 for r in ds if len(r['human_answers']) > 1)
n_multi_gpt   = sum(1 for r in ds if len(r['chatgpt_answers']) > 1)
print(f"  Registros com >1 human_answer   : {n_multi_human}")
print(f"  Registros com >1 chatgpt_answer : {n_multi_gpt}")

# comprimentos
human_lens = [len(r['human_answers'][0]) for r in ds if r['human_answers']]
gpt_lens   = [len(r['chatgpt_answers'][0]) for r in ds if r['chatgpt_answers']]
print(f"\n  human_answers[0] len — min:{min(human_lens)} / média:{sum(human_lens)//len(human_lens)} / max:{max(human_lens)}")
print(f"  chatgpt_answers[0] len — min:{min(gpt_lens)} / média:{sum(gpt_lens)//len(gpt_lens)} / max:{max(gpt_lens)}")

print("\n--- Campos disponíveis para unificação ---")
print("  texto humano  : human_answers[0]")
print("  texto gpt     : chatgpt_answers[0]")
print("  label humano  : 0")
print("  label gpt     : 1")
print("  source/dataset: 'hc3'")
print("  domínio       : source (reddit_eli5, stackoverflow, ...)")

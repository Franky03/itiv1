"""
Análise PPM no dataset HC3 (Human ChatGPT Comparison Corpus).

A análise de NCD requer pares (humano + GPT para a mesma pergunta),
disponíveis apenas no HC3.  O script pode carregar os dados de:
  1. data/unified_dataset.parquet  — subconjunto HC3 do dataset local
  2. Hello-SimpleAI/HC3            — fallback via HuggingFace

Uso:
    python analysis_hc3.py              # dataset local (500 pares)
    python analysis_hc3.py --hc3        # força HC3 via HuggingFace
    python analysis_hc3.py --full       # sem limite de amostras
"""
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm
from datasets import load_dataset

# Garante que o pacote local ppmc é encontrado
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ppmc.compressor import compress_text, train, compress_with_model

LOCAL_PARQUET = pathlib.Path('data/unified_dataset.parquet')

# ── Configurações ─────────────────────────────────────────────────────────────
MIN_CHARS   = 50       # textos abaixo deste limite são ignorados
MAX_ORDER   = 5        # ordem máxima do PPM
BACKEND     = 'hash'   # backend do modelo PPM

RESULTS_DIR = pathlib.Path('results')


# ── Utilitários ───────────────────────────────────────────────────────────────

def _get_answer(sample, key: str) -> str:
    """Extrai o primeiro elemento de uma lista de respostas; retorna '' se vazia."""
    answers = sample.get(key) or []
    return answers[0] if answers else ''


def _is_valid(human: str, gpt: str) -> bool:
    return len(human) >= MIN_CHARS and len(gpt) >= MIN_CHARS


# ── Pipeline principal ────────────────────────────────────────────────────────

def _load_pairs_from_local(n_samples: int | None) -> list[dict]:
    """
    Reconstrói pares (humano, gpt) a partir do dataset local unificado,
    usando apenas os registros source_dataset='hc3'.
    Os pares são reconstruídos agrupando por posição original (índice par=humano,
    índice ímpar=gpt, já que build_dataset.py os intercala assim).
    """
    df = pd.read_parquet(LOCAL_PARQUET)
    df = df[df['source_dataset'] == 'hc3'].reset_index(drop=True)
    # Pares intercalados: [h0, g0, h1, g1, ...]
    pairs = []
    for i in range(0, len(df) - 1, 2):
        h_row = df.iloc[i]
        g_row = df.iloc[i + 1]
        if h_row['label'] == 0 and g_row['label'] == 1:
            pairs.append({
                'human':  h_row['text'],
                'gpt':    g_row['text'],
                'source': h_row.get('domain', 'unknown'),
            })
    if n_samples:
        pairs = pairs[:n_samples]
    return pairs


def _load_pairs_from_hf(n_samples: int | None) -> list[dict]:
    print("Carregando HC3 via HuggingFace...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
        split="train",
    )
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))
    pairs = []
    for sample in ds:
        h = _get_answer(sample, 'human_answers')
        g = _get_answer(sample, 'chatgpt_answers')
        if _is_valid(h, g):
            pairs.append({'human': h, 'gpt': g, 'source': sample.get('source', 'unknown')})
    return pairs


def run_analysis(n_samples: int | None = 500, use_hc3: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Carrega dataset ───────────────────────────────────────────────────────
    if not use_hc3 and LOCAL_PARQUET.exists():
        print(f"Carregando pares HC3 do dataset local ({LOCAL_PARQUET})...")
        valid_samples = _load_pairs_from_local(n_samples)
    else:
        if not use_hc3:
            print(f"AVISO: {LOCAL_PARQUET} não encontrado — usando HuggingFace.")
        valid_samples = _load_pairs_from_hf(n_samples)

    print(f"Pares válidos: {len(valid_samples)}")

    if not valid_samples:
        print("Nenhuma amostra válida encontrada. Encerrando.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISE 1 — Compressão individual (bpc por tipo)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ANÁLISE 1] Compressão individual...")
    records_ind = []

    for i, s in enumerate(tqdm(valid_samples, desc="compress_text")):
        res_h = compress_text(s['human'], max_order=MAX_ORDER, backend=BACKEND)
        res_g = compress_text(s['gpt'],   max_order=MAX_ORDER, backend=BACKEND)

        records_ind.append({
            'index':     i,
            'source':    s['source'],
            'bpc_human': res_h['bpc'],
            'bpc_gpt':   res_g['bpc'],
            'len_human': len(s['human']),
            'len_gpt':   len(s['gpt']),
        })

    df_ind = pd.DataFrame(records_ind)
    df_ind.to_csv(RESULTS_DIR / 'compression_individual.csv', index=False)
    print(f"  Salvo: {RESULTS_DIR}/compression_individual.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISE 2 — Compressão cruzada (NCD)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ANÁLISE 2] Compressão cruzada com modelos treinados em corpus...")

    corpus_human = " ".join(s['human'] for s in valid_samples)
    corpus_gpt   = " ".join(s['gpt']   for s in valid_samples)

    print(f"  Corpus humano: {len(corpus_human):,} chars")
    print(f"  Corpus GPT:    {len(corpus_gpt):,} chars")

    print("  Treinando model_H (corpus humano)...")
    model_H = train(corpus_human, max_order=MAX_ORDER, backend=BACKEND)

    print("  Treinando model_G (corpus GPT)...")
    model_G = train(corpus_gpt, max_order=MAX_ORDER, backend=BACKEND)

    records_cross = []

    for i, s in enumerate(tqdm(valid_samples, desc="cross compress")):
        bpc_HH = compress_with_model(model_H, s['human'])['bpc']
        bpc_GG = compress_with_model(model_G, s['gpt'])['bpc']
        bpc_HG = compress_with_model(model_H, s['gpt'])['bpc']
        bpc_GH = compress_with_model(model_G, s['human'])['bpc']

        denom = max(bpc_HG, bpc_GH)
        ncd   = (bpc_HG + bpc_GH - bpc_HH - bpc_GG) / denom if denom > 0 else 0.0
        asym  = bpc_HG - bpc_GH

        records_cross.append({
            'index':      i,
            'source':     s['source'],
            'bpc_HH':     bpc_HH,
            'bpc_GG':     bpc_GG,
            'bpc_HG':     bpc_HG,
            'bpc_GH':     bpc_GH,
            'NCD':        ncd,
            'asymmetry':  asym,
        })

    df_cross = pd.DataFrame(records_cross)
    df_cross.to_csv(RESULTS_DIR / 'compression_crossmodel.csv', index=False)
    print(f"  Salvo: {RESULTS_DIR}/compression_crossmodel.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISE 3 — Agrupamento por source
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[ANÁLISE 3] Estatísticas por source...")

    df_merged = df_ind.merge(
        df_cross[['index', 'NCD', 'asymmetry']],
        on='index',
        how='inner',
    )

    stats_by_source = (
        df_merged.groupby('source')
        .agg(
            bpc_human_mean =('bpc_human',  'mean'),
            bpc_human_std  =('bpc_human',  'std'),
            bpc_gpt_mean   =('bpc_gpt',    'mean'),
            bpc_gpt_std    =('bpc_gpt',    'std'),
            NCD_mean       =('NCD',        'mean'),
            NCD_std        =('NCD',        'std'),
            asymmetry_mean =('asymmetry',  'mean'),
            asymmetry_std  =('asymmetry',  'std'),
            count          =('index',      'count'),
        )
        .reset_index()
    )

    stats_by_source.to_csv(RESULTS_DIR / 'stats_by_source.csv', index=False)
    print(f"  Salvo: {RESULTS_DIR}/stats_by_source.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # IMPRESSÃO DE RESULTADOS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    mean_h = df_ind['bpc_human'].mean()
    mean_g = df_ind['bpc_gpt'].mean()
    t_stat, p_val = scipy_stats.ttest_ind(df_ind['bpc_human'], df_ind['bpc_gpt'])
    print(f"\nbpc médio — humano : {mean_h:.4f}")
    print(f"bpc médio — GPT    : {mean_g:.4f}")
    print(f"t-test (bpc_h vs bpc_g): t={t_stat:.3f}, p={p_val:.4e}")

    ncd_mean = df_cross['NCD'].mean()
    print(f"\nNCD médio geral: {ncd_mean:.4f}")

    ncd_by_source = (
        df_cross.groupby('source')['NCD']
        .mean()
        .sort_values(ascending=False)
    )
    print("\nNCD por source (decrescente):")
    for src, val in ncd_by_source.items():
        print(f"  {src:30s}: {val:.4f}")

    print(f"\nMaior NCD: {ncd_by_source.index[0]}  ({ncd_by_source.iloc[0]:.4f})")
    print(f"Menor NCD: {ncd_by_source.index[-1]}  ({ncd_by_source.iloc[-1]:.4f})")

    asym_mean = df_cross['asymmetry'].mean()
    print(f"\nAssimetria média (bpc_HG - bpc_GH): {asym_mean:.4f}")
    if asym_mean > 0:
        print("Direção: GPT surpreende mais o modelo humano (bpc_HG > bpc_GH)")
    else:
        print("Direção: Humano surpreende mais o modelo GPT (bpc_GH > bpc_HG)")

    print("\nArquivos gerados em results/")
    print("Execute `python plot_results.py` para gerar os gráficos.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise PPM no HC3")
    parser.add_argument('--hc3', action='store_true',
                        help="Força HC3 via HuggingFace (ignora dataset local)")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument('--full', action='store_true',
                            help="Usa o dataset completo sem limite")
    size_group.add_argument('-n', '--n-samples', type=int, default=500,
                            help="Número de pares (padrão: 500)")
    args = parser.parse_args()

    run_analysis(
        n_samples=None if args.full else args.n_samples,
        use_hc3=args.hc3,
    )

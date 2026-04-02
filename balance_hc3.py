"""
Análise de balanceamento do dataset HC3 (Human ChatGPT Comparison Corpus).

Verifica:
  - Distribuição por source (domínio)
  - Razão humano/GPT por par
  - Distribuição de comprimento de texto
  - Amostras filtradas (< MIN_CHARS)

Uso:
    python balance_hc3.py              # primeiras 500 amostras
    python balance_hc3.py --full       # dataset completo
"""
import sys
import pathlib
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

sys.path.insert(0, str(pathlib.Path(__file__).parent))

MIN_CHARS   = 50
RESULTS_DIR = pathlib.Path('results')


def _first(answers) -> str:
    answers = answers or []
    return answers[0] if answers else ''


def run_balance(n_samples: int | None = 500) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Carregando dataset HC3...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
        split="train",
    )

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    total = len(ds)
    print(f"Total de pares carregados: {total}\n")

    # ── Coleta dados brutos ────────────────────────────────────────────────────
    rows = []
    for s in ds:
        h = _first(s.get('human_answers'))
        g = _first(s.get('chatgpt_answers'))
        rows.append({
            'source':      s.get('source', 'unknown'),
            'len_human':   len(h),
            'len_gpt':     len(g),
            'n_human_ans': len(s.get('human_answers') or []),
            'n_gpt_ans':   len(s.get('chatgpt_answers') or []),
            'valid':       len(h) >= MIN_CHARS and len(g) >= MIN_CHARS,
        })

    df = pd.DataFrame(rows)

    # ── 1. Distribuição por source ─────────────────────────────────────────────
    print("=" * 60)
    print("1. DISTRIBUIÇÃO POR SOURCE")
    print("=" * 60)
    src_counts = df['source'].value_counts().sort_values(ascending=False)
    src_valid  = df[df['valid']].groupby('source').size().reindex(src_counts.index, fill_value=0)

    print(f"{'Source':<30} {'Total':>7} {'%Total':>7} {'Válidos':>8} {'%Válidos':>9}")
    print("-" * 65)
    for src in src_counts.index:
        tot   = src_counts[src]
        pct   = tot / total * 100
        val   = src_valid[src]
        vpct  = val / tot * 100
        print(f"  {src:<28} {tot:>7}  {pct:>6.1f}%  {val:>8}  {vpct:>8.1f}%")

    print(f"\n  {'TOTAL':<28} {total:>7}  100.0%  {df['valid'].sum():>8}  {df['valid'].mean()*100:>8.1f}%")

    # ── 2. Razão respostas humanas vs GPT por par ──────────────────────────────
    print("\n" + "=" * 60)
    print("2. RESPOSTAS POR PAR (humano pode ter múltiplas)")
    print("=" * 60)
    total_h = df['n_human_ans'].sum()
    total_g = df['n_gpt_ans'].sum()
    multi   = (df['n_human_ans'] > 1).sum()
    print(f"  Total respostas humanas (incl. múltiplas): {total_h}")
    print(f"  Total respostas GPT:                       {total_g}")
    print(f"  Razão humano/GPT:                          {total_h/total_g:.2f}")
    print(f"  Pares com >1 resposta humana:              {multi}  ({multi/total*100:.1f}%)")

    nhuman_dist = df['n_human_ans'].value_counts().sort_index()
    print(f"\n  Distribuição n_respostas_humanas por par:")
    for k, v in nhuman_dist.items():
        print(f"    {k} resposta(s): {v} pares  ({v/total*100:.1f}%)")

    # ── 3. Comprimento das respostas ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. COMPRIMENTO DAS RESPOSTAS (chars)")
    print("=" * 60)
    for label, col in [('Humano', 'len_human'), ('GPT', 'len_gpt')]:
        s = df[col]
        print(f"  {label}:")
        print(f"    Média:    {s.mean():.0f}    Mediana: {s.median():.0f}")
        print(f"    Std:      {s.std():.0f}    Min: {s.min()}    Max: {s.max()}")
        print(f"    p25: {s.quantile(.25):.0f}   p75: {s.quantile(.75):.0f}   p95: {s.quantile(.95):.0f}")

    # ── 4. Amostras filtradas (< MIN_CHARS) ────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"4. FILTRAGEM (mínimo {MIN_CHARS} chars em ambas as respostas)")
    print("=" * 60)
    invalid    = df[~df['valid']]
    short_h    = (df['len_human'] < MIN_CHARS).sum()
    short_g    = (df['len_gpt']   < MIN_CHARS).sum()
    empty_h    = (df['len_human'] == 0).sum()
    empty_g    = (df['len_gpt']   == 0).sum()
    print(f"  Válidos:               {df['valid'].sum():>6}  ({df['valid'].mean()*100:.1f}%)")
    print(f"  Inválidos total:       {len(invalid):>6}  ({len(invalid)/total*100:.1f}%)")
    print(f"    Humana < {MIN_CHARS} chars:  {short_h:>6}")
    print(f"    GPT   < {MIN_CHARS} chars:  {short_g:>6}")
    print(f"    Humana vazia:        {empty_h:>6}")
    print(f"    GPT   vazia:         {empty_g:>6}")

    # ── 5. Índice de Gini por source ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. BALANCEAMENTO ENTRE SOURCES (índice de Gini)")
    print("=" * 60)
    counts = np.array(src_counts.values, dtype=float)
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)
    cumsum = np.cumsum(counts_sorted)
    gini = (2 * np.sum((np.arange(1, n+1)) * counts_sorted) / (n * cumsum[-1])) - (n + 1) / n
    print(f"  Gini (sources): {gini:.4f}  (0=perfeito, 1=máximo desequilíbrio)")
    if gini < 0.2:
        verdict = "Bem balanceado"
    elif gini < 0.4:
        verdict = "Moderadamente balanceado"
    else:
        verdict = "Desbalanceado"
    print(f"  Avaliação: {verdict}")

    source_ratio = src_counts.max() / src_counts.min()
    print(f"  Razão max/min source: {source_ratio:.1f}x")

    # ── Salva CSV resumo ───────────────────────────────────────────────────────
    summary = pd.DataFrame({
        'source':        src_counts.index,
        'total_pares':   src_counts.values,
        'pct_total':     (src_counts.values / total * 100).round(2),
        'pares_validos': src_valid.values,
        'pct_validos':   (src_valid.values / src_counts.values * 100).round(2),
        'len_human_mean': [df[df['source']==s]['len_human'].mean().round(1) for s in src_counts.index],
        'len_gpt_mean':   [df[df['source']==s]['len_gpt'].mean().round(1)   for s in src_counts.index],
    })
    out = RESULTS_DIR / 'balance_by_source.csv'
    summary.to_csv(out, index=False)
    print(f"\nCSV salvo em: {out}")

    # ── Gráficos ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HC3 — Análise de Balanceamento', fontsize=14)

    # (a) Barras por source
    ax = axes[0, 0]
    y_pos = range(len(src_counts))
    ax.barh(list(y_pos), src_counts.values, color='steelblue', label='Total')
    ax.barh(list(y_pos), src_valid.values, color='orange', alpha=0.8, label='Válidos')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(src_counts.index, fontsize=8)
    ax.set_xlabel('Número de pares')
    ax.set_title('Pares por source')
    ax.legend(fontsize=8)

    # (b) Pizza por source
    ax = axes[0, 1]
    ax.pie(src_counts.values, labels=src_counts.index, autopct='%1.1f%%', textprops={'fontsize': 7})
    ax.set_title('Proporção por source')

    # (c) Distribuição comprimento humano vs GPT
    ax = axes[1, 0]
    bins = np.histogram_bin_edges(
        list(df['len_human']) + list(df['len_gpt']), bins=40
    )
    ax.hist(df['len_human'], bins=bins, alpha=0.6, label='Humano', color='steelblue')
    ax.hist(df['len_gpt'],   bins=bins, alpha=0.6, label='GPT',    color='orange')
    ax.set_xlabel('Comprimento (chars)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de comprimento das respostas')
    ax.legend()

    # (d) Box-plot comprimento por source
    ax = axes[1, 1]
    sources_ordered = src_counts.index.tolist()
    data_h = [df[df['source'] == s]['len_human'].values for s in sources_ordered]
    bp = ax.boxplot(data_h, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.set_xticks(range(1, len(sources_ordered)+1))
    ax.set_xticklabels(sources_ordered, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Comprimento resposta humana (chars)')
    ax.set_title('Comprimento humano por source')

    plt.tight_layout()
    plot_out = RESULTS_DIR / 'balance_hc3.png'
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {plot_out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanceamento do HC3")
    parser.add_argument('--full', action='store_true',
                        help="Usa o dataset completo (padrão: 500 amostras)")
    args = parser.parse_args()
    run_balance(n_samples=None if args.full else 500)

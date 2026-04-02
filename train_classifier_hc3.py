"""
Treina modelos de compressão PPM para textos humanos e GPT,
classifica o conjunto de teste e salva os modelos treinados.

Fontes de dados (em ordem de preferência):
  1. data/unified_dataset.parquet  — dataset local unificado (build_dataset.py)
  2. Hello-SimpleAI/HC3            — fallback via HuggingFace

Uso:
    python train_classifier_hc3.py                  # dataset local (padrão) ou HC3
    python train_classifier_hc3.py --hc3            # força HC3 via HuggingFace
    python train_classifier_hc3.py --full           # sem limite de amostras
    python train_classifier_hc3.py -n 5000          # N amostras do dataset
    python train_classifier_hc3.py --source hc3     # filtra por fonte (local apenas)
"""
import sys
import pathlib
import argparse
import pickle
import json
import multiprocessing as mp
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc,
)
from datasets import load_dataset

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ppmc.compressor import train, compress_with_model, bpc_frozen

# ── Worker para classificação paralela ───────────────────────────────────────
# Globais herdados via fork — sem pickle, sem IPC dos modelos.
# Populados em run() antes de criar o Pool.

_worker_model_H = None
_worker_model_G = None

def _classify_one(args):
    i, text, label, source = args
    bpc_h = bpc_frozen(_worker_model_H, text)
    bpc_g = bpc_frozen(_worker_model_G, text)
    pred  = 0 if bpc_h <= bpc_g else 1
    return {
        'index':          i,
        'source_dataset': source,
        'true_label':     'human' if label == 0 else 'gpt',
        'pred_label':     'human' if pred  == 0 else 'gpt',
        'correct':        label == pred,
        'bpc_H':          bpc_h,
        'bpc_G':          bpc_g,
        'bpc_diff':       bpc_h - bpc_g,
    }

# ── Configurações ─────────────────────────────────────────────────────────────
MIN_CHARS  = 50
MAX_ORDER  = 5
BACKEND    = 'hash'
TEST_SIZE  = 0.2
RANDOM_SEED = 42

RESULTS_DIR  = pathlib.Path('results')
MODELS_DIR   = pathlib.Path('models')
LOCAL_PARQUET = pathlib.Path('data/unified_dataset.parquet')


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_local(n: int | None, source_filter: str | None) -> pd.DataFrame:
    """Carrega data/unified_dataset.parquet (produzido por build_dataset.py)."""
    print(f"Carregando dataset local: {LOCAL_PARQUET}")
    df = pd.read_parquet(LOCAL_PARQUET)
    if source_filter:
        df = df[df['source_dataset'] == source_filter].reset_index(drop=True)
        print(f"  Filtro source_dataset='{source_filter}': {len(df):,} amostras")
    df = df[df['text'].str.len() >= MIN_CHARS].reset_index(drop=True)
    if n:
        df = (df.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(n // 2, len(x)), random_state=RANDOM_SEED))
                .reset_index(drop=True))
    print(f"  Total: {len(df):,}  (humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    return df


def _load_hc3(n: int | None) -> pd.DataFrame:
    """Carrega HC3 diretamente do HuggingFace e converte para schema flat."""
    print("Carregando HC3 via HuggingFace...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
        split="train",
    )
    if n:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for s in ds:
        h = (s.get('human_answers') or [''])[0]
        g = (s.get('chatgpt_answers') or [''])[0]
        dom = s.get('source', 'unknown')
        if len(h) >= MIN_CHARS:
            rows.append({'text': h, 'label': 0, 'source_dataset': 'hc3', 'domain': dom})
        if len(g) >= MIN_CHARS:
            rows.append({'text': g, 'label': 1, 'source_dataset': 'hc3', 'domain': dom})
    df = pd.DataFrame(rows)
    print(f"  HC3: {len(df):,}  (humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    return df


def load_data(use_hc3: bool, n: int | None, source_filter: str | None) -> pd.DataFrame:
    if not use_hc3 and LOCAL_PARQUET.exists():
        return _load_local(n, source_filter)
    if not use_hc3:
        print(f"AVISO: {LOCAL_PARQUET} não encontrado — usando HC3 via HuggingFace.")
    return _load_hc3(n)


def split_flat(df: pd.DataFrame, test_size: float, seed: int):
    """Split estratificado por (label × source_dataset)."""
    strat = df['label'].astype(str) + '_' + df['source_dataset']
    # Grupos com < 2 amostras não podem ser estratificados
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 2, other='__other__')
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)



# ── Pipeline principal ────────────────────────────────────────────────────────

def run(n_samples: int | None, use_hc3: bool, source_filter: str | None) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # ── Carrega dataset ───────────────────────────────────────────────────────
    df = load_data(use_hc3, n_samples, source_filter)
    if df.empty:
        print("Nenhuma amostra válida. Encerrando.")
        return

    # ── Split treino / teste ──────────────────────────────────────────────────
    print(f"\nDividindo dataset (treino={1-TEST_SIZE:.0%} / teste={TEST_SIZE:.0%})...")
    df_train, df_test = split_flat(df, TEST_SIZE, RANDOM_SEED)
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print(f"  Treino: {len(df_train):,}  (humano={(df_train.label==0).sum():,} | IA={(df_train.label==1).sum():,})")
    print(f"  Teste : {len(df_test):,}   (humano={(df_test.label==0).sum():,}  | IA={(df_test.label==1).sum():,})")

    src_train = Counter(df_train['source_dataset'])
    src_test  = Counter(df_test['source_dataset'])
    print("\n  Distribuição por fonte (treino / teste):")
    for src in sorted(set(src_train) | set(src_test)):
        print(f"    {src:25s}: {src_train.get(src,0):6,} treino | {src_test.get(src,0):6,} teste")

    # ── Corpus de treino ──────────────────────────────────────────────────────
    corpus_human_train = " ".join(df_train[df_train.label == 0]['text'])
    corpus_gpt_train   = " ".join(df_train[df_train.label == 1]['text'])

    # ── Treina ou carrega modelos ─────────────────────────────────────────────
    model_h_path = MODELS_DIR / 'model_H.pkl'
    model_g_path = MODELS_DIR / 'model_G.pkl'

    if model_h_path.exists() and model_g_path.exists():
        print("\nModelos já existem — carregando sem treinar novamente.")
        with open(model_h_path, 'rb') as f:
            model_H = pickle.load(f)
        with open(model_g_path, 'rb') as f:
            model_G = pickle.load(f)
        print(f"  {model_h_path}  ({model_h_path.stat().st_size / 1024:.1f} KB)")
        print(f"  {model_g_path}  ({model_g_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"\n[TREINO] Corpus humano : {len(corpus_human_train):,} chars")
        print(f"[TREINO] Corpus GPT    : {len(corpus_gpt_train):,} chars")

        print("\nTreinando model_H (corpus humano)...")
        model_H = train(corpus_human_train, max_order=MAX_ORDER, backend=BACKEND)

        print("Treinando model_G (corpus GPT)...")
        model_G = train(corpus_gpt_train, max_order=MAX_ORDER, backend=BACKEND)

        with open(model_h_path, 'wb') as f:
            pickle.dump(model_H, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(model_g_path, 'wb') as f:
            pickle.dump(model_G, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nModelos salvos em:")
        print(f"  {model_h_path}  ({model_h_path.stat().st_size / 1024:.1f} KB)")
        print(f"  {model_g_path}  ({model_g_path.stat().st_size / 1024:.1f} KB)")

    # Salva metadados dos modelos
    meta = {
        'max_order':          MAX_ORDER,
        'backend':            BACKEND,
        'min_chars':          MIN_CHARS,
        'test_size':          TEST_SIZE,
        'random_seed':        RANDOM_SEED,
        'n_train':            len(df_train),
        'n_test':             len(df_test),
        'corpus_human_chars': len(corpus_human_train),
        'corpus_gpt_chars':   len(corpus_gpt_train),
        'sources':            df['source_dataset'].unique().tolist(),
    }
    with open(MODELS_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Classificação no conjunto de teste (paralela) ─────────────────────────
    print("\n[CLASSIFICAÇÃO] Avaliando no conjunto de teste...")

    test_texts   = df_test['text'].tolist()
    test_labels  = df_test['label'].tolist()
    flat_sources = df_test['source_dataset'].tolist()

    tasks = [
        (i, text, label, source)
        for i, (text, label, source) in enumerate(zip(test_texts, test_labels, flat_sources))
    ]

    # Seta globais antes do fork — workers herdam sem pickling
    global _worker_model_H, _worker_model_G
    _worker_model_H = model_H
    _worker_model_G = model_G

    n_workers = max(1, os.cpu_count() - 1)
    print(f"  Usando {n_workers} workers (cpu_count={os.cpu_count()})")

    import time
    records    = [None] * len(tasks)
    completed  = 0
    total      = len(tasks)
    start_time = time.time()

    def _on_done(result):
        nonlocal completed
        records[result['index']] = result
        completed += 1
        elapsed = time.time() - start_time
        rate    = completed / elapsed if elapsed > 0 else 0
        eta     = (total - completed) / rate if rate > 0 else float('inf')
        pct     = completed / total * 100
        print(
            f"\r  [{completed:>{len(str(total))}}/{total}] "
            f"{pct:5.1f}%  |  {rate:.2f} it/s  |  ETA {eta:6.0f}s",
            end='', flush=True,
        )

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        async_results = [
            pool.apply_async(_classify_one, (task,), callback=_on_done)
            for task in tasks
        ]
        pool.close()
        pool.join()

    print()  # nova linha após o progresso inline
    preds = [0 if r['pred_label'] == 'human' else 1 for r in records]

    # ── Métricas ──────────────────────────────────────────────────────────────
    acc = accuracy_score(test_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average='binary', pos_label=1
    )
    cm = confusion_matrix(test_labels, preds)

    print("\n" + "=" * 60)
    print("RESULTADOS DE CLASSIFICAÇÃO")
    print("=" * 60)
    print(f"\nAcurácia  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precisão  : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"\nMatriz de confusão (linhas=real, colunas=predito):")
    print(f"  Labels: 0=humano, 1=GPT")
    print(f"  {cm}")
    print(f"\nRelatório completo:")
    print(classification_report(test_labels, preds, target_names=['humano', 'gpt']))

    df_results = pd.DataFrame(records)
    out_csv = RESULTS_DIR / 'classification_results.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"\nResultados por amostra salvos em: {out_csv}")

    # Acurácia por source
    print("\nAcurácia por fonte:")
    acc_src = (
        df_results.groupby('source_dataset')['correct']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'accuracy', 'count': 'n'})
        .sort_values('accuracy', ascending=False)
    )
    for src, row in acc_src.iterrows():
        print(f"  {src:30s}: {row['accuracy']*100:.1f}%  (n={int(row['n'])})")

    # Acurácia separada por classe real
    df_human = df_results[df_results['true_label'] == 'human']
    df_gpt   = df_results[df_results['true_label'] == 'gpt']
    print(f"\nAcurácia em textos humanos : {df_human['correct'].mean()*100:.1f}%")
    print(f"Acurácia em textos GPT    : {df_gpt['correct'].mean()*100:.1f}%")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    # Score contínuo: bpc_diff = bpc_H - bpc_G
    #   positivo → model_G comprime melhor → mais parecido com IA
    #   negativo → model_H comprime melhor → mais parecido com humano
    scores = df_results['bpc_diff'].values   # positivo = voto IA

    fpr, tpr, thresholds = roc_curve(test_labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC-ROC : {roc_auc:.4f}")

    # Melhor threshold pelo índice de Youden (maximiza tpr - fpr)
    youden_idx  = (tpr - fpr).argmax()
    best_thresh = thresholds[youden_idx]
    print(f"Melhor threshold (Youden): {best_thresh:.4f}  "
          f"(TPR={tpr[youden_idx]:.3f}, FPR={fpr[youden_idx]:.3f})")

    # Salva dados da ROC
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
    roc_csv = RESULTS_DIR / 'roc_curve.csv'
    df_roc.to_csv(roc_csv, index=False)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f'PPM (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Aleatório (AUC = 0.500)')
    ax.scatter(fpr[youden_idx], tpr[youden_idx], s=80, zorder=5,
               label=f'Melhor threshold ({best_thresh:.3f})')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('ROC Curve — Classificador PPM (humano vs IA)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_png = RESULTS_DIR / 'roc_curve.png'
    fig.savefig(roc_png, dpi=150)
    plt.close(fig)

    print("\nArquivos gerados:")
    print(f"  {model_h_path}")
    print(f"  {model_g_path}")
    print(f"  {MODELS_DIR}/meta.json")
    print(f"  {out_csv}")
    print(f"  {roc_csv}")
    print(f"  {roc_png}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina classificador PPM")
    parser.add_argument('--hc3', action='store_true',
                        help="Força uso do HC3 via HuggingFace (ignora dataset local)")
    parser.add_argument('--source', type=str, default=None,
                        help="Filtra por fonte no dataset local (hc3 | ai_human_hf | ai_human_csv)")
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument('--full', action='store_true',
                            help="Usa o dataset completo sem limite")
    size_group.add_argument('-n', '--n-samples', type=int, default=None,
                            help="Limita a N amostras totais (balanceado por label)")
    args = parser.parse_args()

    run(
        n_samples=None if args.full else args.n_samples,
        use_hc3=args.hc3,
        source_filter=args.source,
    )

"""
Ablation study: treina 7 pares de modelos PPM com diferentes combinações
de fontes e compara as ROC curves no mesmo conjunto de teste.

Combinações:
  A  — todas as fontes (hc3 + ai_human_hf + ai_human_csv)
  B  — só hc3
  C  — só ai_human_hf
  D  — só ai_human_csv
  BC — hc3 + ai_human_hf
  BD — hc3 + ai_human_csv
  CD — ai_human_hf + ai_human_csv

O split treino/teste é feito UMA VEZ sobre o dataset completo e reutilizado
por todos os modelos — comparação justa no mesmo conjunto de teste.

Uso:
    python ablation_study.py              # dataset local completo
    python ablation_study.py -n 10000     # limita a N amostras totais
    python ablation_study.py --retrain    # força re-treino mesmo se pkl existir
"""
import argparse
import os
import pathlib
import pickle
import sys
import time
import multiprocessing as mp
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ppmc.compressor import train, bpc_frozen

# ── Configurações ─────────────────────────────────────────────────────────────
MIN_CHARS   = 50
MAX_ORDER   = 5
BACKEND     = 'hash'
TEST_SIZE   = 0.2
RANDOM_SEED = 42

LOCAL_PARQUET = pathlib.Path('data/unified_dataset.parquet')
MODELS_DIR    = pathlib.Path('models/ablation')
RESULTS_DIR   = pathlib.Path('results')

COMBINATIONS = {
    'A_all':     ['hc3', 'ai_human_hf', 'ai_human_csv'],
    'B_hc3':     ['hc3'],
    'C_hf':      ['ai_human_hf'],
    'D_csv':     ['ai_human_csv'],
    'BC_hc3+hf': ['hc3', 'ai_human_hf'],
    'BD_hc3+csv':['hc3', 'ai_human_csv'],
    'CD_hf+csv': ['ai_human_hf', 'ai_human_csv'],
}

# ── Workers paralelos ─────────────────────────────────────────────────────────
_worker_model_H = None
_worker_model_G = None

def _classify_one(args):
    seq, text, label = args
    bpc_h = bpc_frozen(_worker_model_H, text)
    bpc_g = bpc_frozen(_worker_model_G, text)
    return seq, label, bpc_h - bpc_g   # score positivo = voto IA


def _run_classification(model_H, model_G, df_test: pd.DataFrame):
    global _worker_model_H, _worker_model_G
    _worker_model_H = model_H
    _worker_model_G = model_G

    tasks   = [(i, row.text, row.label)
               for i, row in enumerate(df_test.itertuples(index=False))]
    total   = len(tasks)
    results = [None] * total
    done    = 0
    t0      = time.time()

    def _on_done(res):
        nonlocal done
        seq, label, score = res
        results[seq] = (label, score)
        done += 1
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / rate if rate > 0 else float('inf')
        print(f"\r    [{done:>{len(str(total))}}/{total}] "
              f"{done/total*100:5.1f}%  {rate:.1f} it/s  ETA {eta:.0f}s",
              end='', flush=True)

    n_workers = max(1, os.cpu_count() - 1)
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for task in tasks:
            pool.apply_async(_classify_one, (task,), callback=_on_done)
        pool.close()
        pool.join()
    print()

    labels = [r[0] for r in results]
    scores = [r[1] for r in results]
    return labels, scores


# ── Treino de um par de modelos ───────────────────────────────────────────────

def _train_pair(name: str, sources: list[str], df_train: pd.DataFrame,
                retrain: bool) -> tuple:
    h_path = MODELS_DIR / f'{name}_H.pkl'
    g_path = MODELS_DIR / f'{name}_G.pkl'

    if not retrain and h_path.exists() and g_path.exists():
        print(f"  [{name}] Carregando modelos salvos...")
        with open(h_path, 'rb') as f: model_H = pickle.load(f)
        with open(g_path, 'rb') as f: model_G = pickle.load(f)
        return model_H, model_G

    df_sub = df_train[df_train['source_dataset'].isin(sources)]
    corpus_H = " ".join(df_sub[df_sub['label'] == 0]['text'])
    corpus_G = " ".join(df_sub[df_sub['label'] == 1]['text'])

    n_h = (df_sub['label'] == 0).sum()
    n_g = (df_sub['label'] == 1).sum()
    print(f"  [{name}] Treino: {n_h:,} humanos | {n_g:,} IA  "
          f"({len(corpus_H):,} / {len(corpus_G):,} chars)")

    print(f"  [{name}] Treinando model_H...")
    model_H = train(corpus_H, max_order=MAX_ORDER, backend=BACKEND)
    print(f"  [{name}] Treinando model_G...")
    model_G = train(corpus_G, max_order=MAX_ORDER, backend=BACKEND)

    with open(h_path, 'wb') as f: pickle.dump(model_H, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(g_path, 'wb') as f: pickle.dump(model_G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [{name}] Modelos salvos.")
    return model_H, model_G


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(n_samples: int | None, retrain: bool) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Carrega dataset ───────────────────────────────────────────────────────
    print(f"Carregando {LOCAL_PARQUET}...")
    df = pd.read_parquet(LOCAL_PARQUET)
    df = df[df['text'].str.len() >= MIN_CHARS].reset_index(drop=True)

    if n_samples:
        df = (df.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(n_samples // 2, len(x)),
                                           random_state=RANDOM_SEED))
                .reset_index(drop=True))

    print(f"  Total: {len(df):,}  "
          f"(humano={(df.label==0).sum():,} | IA={(df.label==1).sum():,})")
    src_counts = df['source_dataset'].value_counts()
    for src, cnt in src_counts.items():
        print(f"    {src:25s}: {cnt:,}")

    # ── Split único — mesmo teste para todos os modelos ───────────────────────
    print(f"\nSplit treino/teste ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
    strat = (df['label'].astype(str) + '_' + df['source_dataset'])
    counts = strat.value_counts()
    strat  = strat.where(strat.map(counts) >= 2, other='__other__')

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strat
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"  Treino: {len(df_train):,} | Teste: {len(df_test):,}")

    # ── Treina os 7 pares ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TREINAMENTO DOS MODELOS")
    print("=" * 60)

    trained = {}
    for name, sources in COMBINATIONS.items():
        trained[name] = _train_pair(name, sources, df_train, retrain)

    # ── Classifica com cada par e coleta scores ───────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICAÇÃO")
    print("=" * 60)

    all_results = {}
    for name, (model_H, model_G) in trained.items():
        print(f"\n[{name}] Classificando {len(df_test):,} textos...")
        labels, scores = _run_classification(model_H, model_G, df_test)
        preds = [1 if s > 0 else 0 for s in scores]
        acc   = accuracy_score(labels, preds)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print(f"  Acurácia: {acc*100:.2f}%  |  AUC: {roc_auc:.4f}")
        all_results[name] = {
            'labels':  labels,
            'scores':  scores,
            'preds':   preds,
            'acc':     acc,
            'fpr':     fpr,
            'tpr':     tpr,
            'auc':     roc_auc,
        }

    # ── Resumo terminal ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"\n{'Modelo':<15} {'AUC':>6}  {'Acurácia':>9}")
    print("-" * 35)
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]['auc']):
        print(f"{name:<15} {res['auc']:>6.4f}  {res['acc']*100:>8.2f}%")

    # ── Salva scores por modelo ───────────────────────────────────────────────
    records = {'true_label': all_results[next(iter(all_results))]['labels']}
    for name, res in all_results.items():
        records[f'score_{name}'] = res['scores']
        records[f'pred_{name}']  = res['preds']
    pd.DataFrame(records).to_csv(RESULTS_DIR / 'ablation_scores.csv', index=False)

    # Salva ROC data
    for name, res in all_results.items():
        pd.DataFrame({
            'fpr': res['fpr'], 'tpr': res['tpr'],
        }).to_csv(RESULTS_DIR / f'roc_{name}.csv', index=False)

    # ── Plot: 7 ROC curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    # Ordena por AUC decrescente para legenda mais legível
    sorted_names = sorted(all_results, key=lambda n: -all_results[n]['auc'])
    colors = plt.cm.tab10.colors

    for i, name in enumerate(sorted_names):
        res = all_results[name]
        sources = COMBINATIONS[name]
        label_str = (f"{name}  (AUC={res['auc']:.3f}, "
                     f"acc={res['acc']*100:.1f}%)")
        ax.plot(res['fpr'], res['tpr'], lw=2, color=colors[i], label=label_str)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Aleatório (AUC=0.500)')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    ax.set_title('ROC Curves — Ablation Study PPM\n(humano vs IA)', fontsize=13)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = RESULTS_DIR / 'ablation_roc_curves.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nGráfico salvo: {out_png}")
    print(f"Scores salvos: {RESULTS_DIR}/ablation_scores.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study — 7 combinações de fontes')
    parser.add_argument('-n', '--n-samples', type=int, default=None,
                        help='Limita a N amostras totais (balanceado por label)')
    parser.add_argument('--retrain', action='store_true',
                        help='Força re-treino mesmo se modelos já existirem')
    args = parser.parse_args()
    run(n_samples=args.n_samples, retrain=args.retrain)

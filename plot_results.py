"""
Gera gráficos de análise HC3 a partir dos CSVs em results/.

Uso:
    python plot_results.py

Saída:
    plots/bpc_distribution.png
    plots/ncd_by_source.png
    plots/cross_compression_scatter.png
    plots/asymmetry_distribution.png
    plots/bpc_vs_length.png
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    _LOWESS = True
except ImportError:
    _LOWESS = False

RESULTS_DIR = pathlib.Path('results')
PLOTS_DIR   = pathlib.Path('plots')


def _require_csv(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        print(f"AVISO: {path} não encontrado. Execute analysis_hc3.py primeiro.")
        sys.exit(1)
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Salvo: {out}")


# ── Gráfico 1: Distribuição de bpc ───────────────────────────────────────────

def plot_bpc_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    bpc_h = df['bpc_human'].dropna()
    bpc_g = df['bpc_gpt'].dropna()

    bins = np.linspace(
        min(bpc_h.min(), bpc_g.min()),
        max(bpc_h.max(), bpc_g.max()),
        50,
    )

    ax.hist(bpc_h, bins=bins, alpha=0.6, color='steelblue',  label='Humano',  density=True)
    ax.hist(bpc_g, bins=bins, alpha=0.6, color='darkorange', label='ChatGPT', density=True)

    ax.axvline(bpc_h.mean(), color='steelblue',  linestyle='--', linewidth=1.5,
               label=f'Média humano = {bpc_h.mean():.3f}')
    ax.axvline(bpc_g.mean(), color='darkorange', linestyle='--', linewidth=1.5,
               label=f'Média GPT = {bpc_g.mean():.3f}')

    ax.set_xlabel('Bits por caractere (bpc)', fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.set_title('Distribuição de bits por caractere: humano vs ChatGPT', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, 'bpc_distribution.png')


# ── Gráfico 2: NCD por source ─────────────────────────────────────────────────

def plot_ncd_by_source(df_cross: pd.DataFrame) -> None:
    ncd_agg = (
        df_cross.groupby('source')['NCD']
        .agg(['mean', 'std'])
        .rename(columns={'mean': 'NCD_mean', 'std': 'NCD_std'})
        .sort_values('NCD_mean', ascending=True)
        .reset_index()
    )
    ncd_agg['NCD_std'] = ncd_agg['NCD_std'].fillna(0)

    n = len(ncd_agg)
    norm  = mcolors.Normalize(vmin=ncd_agg['NCD_mean'].min(), vmax=ncd_agg['NCD_mean'].max())
    cmap  = matplotlib.colormaps['Blues']
    colors = [cmap(norm(v)) for v in ncd_agg['NCD_mean']]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.5 + 1)))

    bars = ax.barh(
        ncd_agg['source'],
        ncd_agg['NCD_mean'],
        xerr=ncd_agg['NCD_std'],
        color=colors,
        edgecolor='gray',
        linewidth=0.5,
        error_kw={'elinewidth': 1.2, 'capsize': 4},
    )

    for bar, val in zip(bars, ncd_agg['NCD_mean']):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=9,
        )

    ax.set_xlabel('NCD médio', fontsize=12)
    ax.set_title('Distância de distribuição (NCD) por domínio', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(right=ncd_agg['NCD_mean'].max() + ncd_agg['NCD_std'].max() + 0.06)

    _save(fig, 'ncd_by_source.png')


# ── Gráfico 3: Scatter compressão cruzada ────────────────────────────────────

def plot_cross_compression_scatter(df_cross: pd.DataFrame, df_ind: pd.DataFrame) -> None:
    df = df_cross.merge(df_ind[['index', 'len_human']], on='index', how='left')

    ncd_vals   = df['NCD'].values
    len_vals   = df['len_human'].fillna(df['len_human'].median()).values

    size_norm  = (len_vals - len_vals.min()) / (np.ptp(len_vals) + 1e-9)
    point_size = 10 + size_norm * 80

    norm = mcolors.Normalize(vmin=np.percentile(ncd_vals, 5),
                             vmax=np.percentile(ncd_vals, 95))
    cmap = matplotlib.colormaps['RdYlBu_r']

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        df['bpc_HH'], df['bpc_GG'],
        c=ncd_vals, cmap=cmap, norm=norm,
        s=point_size, alpha=0.6, linewidths=0.3, edgecolors='k',
    )

    # linha diagonal y = x
    lim_min = min(df['bpc_HH'].min(), df['bpc_GG'].min()) * 0.97
    lim_max = max(df['bpc_HH'].max(), df['bpc_GG'].max()) * 1.03
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1.0, alpha=0.5,
            label='y = x')

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('NCD', fontsize=11)

    ax.set_xlabel('bpc_HH  (modelo humano → texto humano)', fontsize=11)
    ax.set_ylabel('bpc_GG  (modelo GPT → texto GPT)', fontsize=11)
    ax.set_title('Espaço de compressão cruzada por amostra', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, 'cross_compression_scatter.png')


# ── Gráfico 4: Distribuição de assimetria ────────────────────────────────────

def plot_asymmetry_distribution(df_cross: pd.DataFrame) -> None:
    asym = df_cross['asymmetry'].dropna()
    mean_asym = asym.mean()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(asym, bins=40, color='mediumpurple', alpha=0.75, edgecolor='white',
            linewidth=0.4, density=True)
    ax.axvline(0,           color='crimson',   linewidth=1.8, label='x = 0')
    ax.axvline(mean_asym,   color='goldenrod', linewidth=1.5, linestyle='--',
               label=f'Média = {mean_asym:.3f}')

    if mean_asym > 0:
        note = "GPT surpreende mais o modelo humano\n(bpc_HG > bpc_GH)"
    else:
        note = "Humano surpreende mais o modelo GPT\n(bpc_GH > bpc_HG)"

    ax.text(
        0.97, 0.95, note,
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8),
    )

    ax.set_xlabel('Assimetria = bpc_HG − bpc_GH', fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.set_title('Assimetria da compressão cruzada', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, 'asymmetry_distribution.png')


# ── Gráfico 5: bpc vs comprimento ────────────────────────────────────────────

def _lowess_or_poly(x, y, color, label, ax):
    order = np.argsort(x)
    xs, ys = x[order], y[order]

    if _LOWESS and len(xs) >= 10:
        smoothed = sm_lowess(ys, xs, frac=0.4, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2.0, label=label)
    else:
        # fallback: polynomial fit
        coeffs = np.polyfit(xs, ys, deg=2)
        x_line = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(x_line, np.polyval(coeffs, x_line), color=color, linewidth=2.0, label=label)


def plot_bpc_vs_length(df_ind: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df_ind['len_human'], df_ind['bpc_human'],
               color='steelblue', alpha=0.35, s=12, label='_nolegend_')
    ax.scatter(df_ind['len_gpt'], df_ind['bpc_gpt'],
               color='darkorange', alpha=0.35, s=12, label='_nolegend_')

    _lowess_or_poly(
        df_ind['len_human'].values,
        df_ind['bpc_human'].values,
        color='steelblue',
        label='Humano (tendência)',
        ax=ax,
    )
    _lowess_or_poly(
        df_ind['len_gpt'].values,
        df_ind['bpc_gpt'].values,
        color='darkorange',
        label='ChatGPT (tendência)',
        ax=ax,
    )

    # pontos fantasma para legenda de scatter
    ax.scatter([], [], color='steelblue',  alpha=0.6, s=20, label='Humano')
    ax.scatter([], [], color='darkorange', alpha=0.6, s=20, label='ChatGPT')

    ax.set_xlabel('Comprimento do texto (chars)', fontsize=12)
    ax.set_ylabel('Bits por caractere (bpc)', fontsize=12)
    ax.set_title('Estabilidade do bpc por comprimento do texto', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, 'bpc_vs_length.png')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Lendo CSVs de resultados...")
    df_ind   = _require_csv(RESULTS_DIR / 'compression_individual.csv')
    df_cross = _require_csv(RESULTS_DIR / 'compression_crossmodel.csv')

    print("\nGerando gráficos...")

    plot_bpc_distribution(df_ind)
    plot_ncd_by_source(df_cross)
    plot_cross_compression_scatter(df_cross, df_ind)
    plot_asymmetry_distribution(df_cross)
    plot_bpc_vs_length(df_ind)

    print(f"\nTodos os gráficos salvos em {PLOTS_DIR}/")


if __name__ == "__main__":
    main()

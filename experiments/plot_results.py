import csv
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experiments.analysis import load_samples, find_stabilization_point, find_transition_points

RESULTS_DIR = pathlib.Path("results/hash")


# ── Gráfico 1: Comprimento Médio Progressivo ─────────────────────────────────

def plot_progressive_mean(
    csv_path: str | pathlib.Path,
    title: str,
    output_path: str | pathlib.Path,
    show_stabilization: bool = True,
    show_transitions: bool = False,
    resets_csv: str | pathlib.Path | None = None
):
    samples = load_samples(csv_path)
    ns = [s[0] for s in samples]
    ls = [s[1] for s in samples]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ns, ls, linewidth=0.9, color='steelblue', label='L(n)')

    if show_stabilization:
        stab = find_stabilization_point(samples)
        if stab:
            ax.axhline(y=stab[1], color='crimson', linestyle='--', linewidth=1.2, alpha=0.8,
                       label=f'Estabilização: L={stab[1]:.3f} bps em n={stab[0]:,}')
            ax.axvline(x=stab[0], color='crimson', linestyle=':', linewidth=1.0, alpha=0.5)

    if show_transitions:
        transitions = find_transition_points(samples, jump_threshold=0.3)
        for i, t in enumerate(transitions):
            ax.axvline(x=t[0], color='darkorange', linestyle=':', linewidth=0.8, alpha=0.7,
                       label='Transição' if i == 0 else None)

    # Linhas vermelhas nos pontos de reset do monitor
    if resets_csv and pathlib.Path(resets_csv).exists():
        reset_positions = []
        with open(resets_csv) as f:
            reader = csv.reader(f)
            next(reader)  # pula header
            for row in reader:
                reset_positions.append(int(row[0]))
        for i, pos in enumerate(reset_positions):
            ax.axvline(x=pos, color='red', linestyle='-', linewidth=0.6, alpha=0.5,
                       label='Reset' if i == 0 else None)

    ax.set_xlabel('Posição n (bytes)', fontsize=12)
    ax.set_ylabel('L(n) = bits totais / n  (bits/símbolo)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo: {output_path}")


# ── Gráfico 2: Bits/símbolo vs Kmax (para um arquivo) ────────────────────────

def plot_bps_vs_kmax(
    benchmark_csv: str | pathlib.Path,
    filename: str,
    output_path: str | pathlib.Path
):
    rows = [r for r in csv.DictReader(open(benchmark_csv)) if r['file'] == filename]
    rows.sort(key=lambda r: int(r['kmax']))

    kmaxs = [int(r['kmax']) for r in rows]
    bpss  = [float(r['bits_per_symbol']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kmaxs, bpss, 'o-', color='steelblue', markersize=6)
    ax.set_xlabel('Kmax', fontsize=12)
    ax.set_ylabel('Bits por símbolo', fontsize=12)
    ax.set_title(f'Compressão PPM-C — {filename}', fontsize=13)
    ax.set_xticks(kmaxs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo: {output_path}")


# ── Gráfico 3: Comparação entre compressores ─────────────────────────────────

def plot_comparison(
    benchmark_csv: str | pathlib.Path,
    external_csv: str | pathlib.Path,
    kmax_to_compare: int = 5,
    output_path: str | pathlib.Path = "results/hash/comparison.png"
):
    # Carrega PPM-C
    ppmc_data = {
        r['file']: float(r['bits_per_symbol'])
        for r in csv.DictReader(open(benchmark_csv))
        if int(r['kmax']) == kmax_to_compare
    }
    # Carrega externo
    ext_data = {
        r['file']: (float(r['gzip9_bps']), float(r['7z_bps']))
        for r in csv.DictReader(open(external_csv))
    }

    files = sorted(set(ppmc_data) & set(ext_data))
    x     = range(len(files))

    ppmc_vals = [ppmc_data[f] for f in files]
    gzip_vals = [ext_data[f][0] for f in files]
    sz_vals   = [ext_data[f][1] for f in files]

    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width for i in x], ppmc_vals, width, label=f'PPM-C (K={kmax_to_compare})', color='steelblue')
    ax.bar([i         for i in x], gzip_vals, width, label='gzip -9',                       color='seagreen')
    ax.bar([i + width for i in x], sz_vals,   width, label='7-Zip -mx=9',                   color='darkorange')

    ax.set_xticks(list(x))
    ax.set_xticklabels(files, rotation=30, ha='right')
    ax.set_ylabel('Bits por símbolo')
    ax.set_title('Comparação de Compressores — Corpus Silesia')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo: {output_path}")


# ── Execução principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Gráfico do Dickens
    dickens_csv = RESULTS_DIR / "progressive_dickens_K5.csv"
    if dickens_csv.exists():
        plot_progressive_mean(
            dickens_csv,
            title="Comprimento Médio Progressivo — Dickens (PPM-C, Kmax=5)",
            output_path=RESULTS_DIR / "progressive_dickens.png",
            show_stabilization=True
        )

    # Gráfico do Silesia concatenado (se existir)
    silesia_csv = RESULTS_DIR / "progressive_silesia_all_K5.csv"
    if silesia_csv.exists():
        plot_progressive_mean(
            silesia_csv,
            title="Comprimento Médio Progressivo — Silesia Completo (PPM-C, Kmax=5)",
            output_path=RESULTS_DIR / "progressive_silesia.png",
            show_stabilization=False,
            show_transitions=True,
            resets_csv=RESULTS_DIR / "resets_silesia_K5.csv"
        )

    # Gráfico bps vs Kmax
    bench_csv = RESULTS_DIR / "benchmark.csv"
    if bench_csv.exists():
        plot_bps_vs_kmax(bench_csv, "dickens", RESULTS_DIR / "bps_vs_kmax_dickens.png")

    # Gráfico de comparação
    ext_csv = RESULTS_DIR / "external_comparison.csv"
    if bench_csv.exists() and ext_csv.exists():
        plot_comparison(bench_csv, ext_csv, kmax_to_compare=5)
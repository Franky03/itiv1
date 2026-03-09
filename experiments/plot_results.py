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


# ── Gráfico 4: Velocidade de aprendizado e saturação de Kmax (dickens) ───────

def plot_learning_and_saturation(
    progressive_csv: str | pathlib.Path,
    benchmark_csv: str | pathlib.Path,
    output_path: str | pathlib.Path = "results/hash/dickens_learning_saturation.png",
):
    samples = load_samples(progressive_csv)
    ns = [s[0] for s in samples]
    ls = [s[1] for s in samples]

    # Valor assintótico = L(n) final
    l_inf = ls[-1]
    threshold_5pct = l_inf * 1.05

    # Ponto de estabilização: primeiro n onde L(n) <= threshold_5pct (e permanece)
    stab_n, stab_l = None, None
    for n, l in zip(ns, ls):
        if l <= threshold_5pct:
            stab_n, stab_l = n, l
            break

    # Dados bps vs kmax
    rows = sorted(
        [r for r in csv.DictReader(open(benchmark_csv)) if r['file'] == 'dickens'],
        key=lambda r: int(r['kmax'])
    )
    kmaxs   = [int(r['kmax'])              for r in rows]
    bpss    = [float(r['bits_per_symbol']) for r in rows]
    times_s = [float(r['compress_time_s']) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Painel esquerdo: curva L(n) ──
    ax1.plot(ns, ls, linewidth=0.8, color='steelblue', label='L(n)')
    ax1.axhline(y=l_inf, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                label=f'Assintótico L∞ = {l_inf:.3f} bps')
    ax1.axhline(y=threshold_5pct, color='seagreen', linestyle=':', linewidth=1.0,
                label=f'L∞ × 1.05 = {threshold_5pct:.3f} bps')
    if stab_n is not None:
        total_bytes = ns[-1]
        pct = stab_n / total_bytes * 100
        ax1.axvline(x=stab_n, color='crimson', linestyle='--', linewidth=1.2,
                    label=f'Estab. em n={stab_n:,} ({pct:.1f}% do arquivo)')
        ax1.scatter([stab_n], [stab_l], color='crimson', zorder=5)
    ax1.set_xlabel('Posição n (bytes)', fontsize=11)
    ax1.set_ylabel('L(n) = bits totais / n  (bits/símbolo)', fontsize=11)
    ax1.set_title('Velocidade de aprendizado — Dickens, $K_{max}=5$', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # ── Painel direito: bps e tempo vs Kmax ──
    color1, color2 = 'steelblue', 'darkorange'
    ax2.plot(kmaxs, bpss, 'o-', color=color1, markersize=6, label='Bits/símbolo')
    for i in range(1, len(kmaxs)):
        delta = bpss[i] - bpss[i - 1]
        ax2.annotate(f'{delta:+.3f}', xy=(kmaxs[i], bpss[i]),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=8, color=color1)

    ax2b = ax2.twinx()
    ax2b.bar(kmaxs, times_s, alpha=0.25, color=color2, label='Tempo (s)')
    ax2b.set_ylabel('Tempo de compressão (s)', fontsize=11, color=color2)
    ax2b.tick_params(axis='y', labelcolor=color2)

    # Destaque no salto K=4->5
    if 4 in kmaxs and 5 in kmaxs:
        i4, i5 = kmaxs.index(4), kmaxs.index(5)
        dt_pct = (times_s[i5] - times_s[i4]) / times_s[i4] * 100
        db     = bpss[i5] - bpss[i4]
        ax2.annotate(
            f'K=4→5: {db:+.3f} bps\n+{dt_pct:.0f}% tempo',
            xy=(5, bpss[i5]), xytext=(4.2, bpss[i5] + 0.15),
            arrowprops=dict(arrowstyle='->', color='crimson'),
            fontsize=8.5, color='crimson',
        )

    ax2.set_xlabel('$K_{max}$', fontsize=11)
    ax2.set_ylabel('Bits por símbolo', fontsize=11, color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.set_title('Saturação de $K_{max}$ — Dickens', fontsize=12)
    ax2.set_xticks(kmaxs)
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo: {output_path}")


# ── Gráfico 5: Comparação com/sem reset adaptativo ───────────────────────────

def plot_reset_comparison(
    with_reset_csv: str | pathlib.Path,
    no_reset_csv: str | pathlib.Path,
    resets_csv: str | pathlib.Path,
    output_path: str | pathlib.Path = "results/hash/silesia_reset_comparison.png",
):
    def read_csv(path):
        samples = load_samples(path)
        return [s[0] for s in samples], [s[1] for s in samples]

    ns_r, ls_r = read_csv(with_reset_csv)
    ns_n, ls_n = read_csv(no_reset_csv)

    reset_positions = []
    if pathlib.Path(resets_csv).exists():
        with open(resets_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                reset_positions.append(int(row[0]))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ns_n, ls_n, linewidth=0.9, color='steelblue', label='Sem reset')
    ax.plot(ns_r, ls_r, linewidth=0.9, color='darkorange', label='Com reset adaptativo')

    for i, pos in enumerate(reset_positions):
        ax.axvline(x=pos, color='red', linestyle='-', linewidth=0.5, alpha=0.35,
                   label='Reset' if i == 0 else None)

    ax.set_xlabel('Posição n (bytes)', fontsize=12)
    ax.set_ylabel('L(n) = bits totais / n  (bits/símbolo)', fontsize=12)
    ax.set_title('Comparação com/sem Reset Adaptativo — Silesia Corpus, $K_{max}=5$', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

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
        )

    # Gráfico bps vs Kmax
    bench_csv = RESULTS_DIR / "benchmark.csv"
    if bench_csv.exists():
        plot_bps_vs_kmax(bench_csv, "dickens", RESULTS_DIR / "bps_vs_kmax_dickens.png")

    # Gráfico velocidade de aprendizado + saturação Kmax
    if dickens_csv.exists() and bench_csv.exists():
        plot_learning_and_saturation(
            progressive_csv=dickens_csv,
            benchmark_csv=bench_csv,
            output_path=RESULTS_DIR / "dickens_learning_saturation.png",
        )

    # Gráfico de comparação
    ext_csv = RESULTS_DIR / "external_comparison.csv"
    if bench_csv.exists() and ext_csv.exists():
        plot_comparison(bench_csv, ext_csv, kmax_to_compare=5)

    # Gráfico comparação com/sem reset — Silesia
    no_reset_csv = RESULTS_DIR / "progressive_silesia_no_reset_K5.csv"
    if silesia_csv.exists() and no_reset_csv.exists():
        plot_reset_comparison(
            with_reset_csv=silesia_csv,
            no_reset_csv=no_reset_csv,
            resets_csv=RESULTS_DIR / "resets_silesia_K5.csv",
            output_path=RESULTS_DIR / "silesia_reset_comparison.png",
        )
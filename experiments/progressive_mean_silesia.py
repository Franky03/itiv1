"""
Comprimento Médio Progressivo sobre o Silesia completo (arquivos concatenados).
Gera progressive_silesia_all_K5.csv para análise de transições entre arquivos.

Uso:
  python -m experiments.progressive_mean_silesia [kmax] [--no-reset]

  --no-reset  Roda sem o mecanismo de reset adaptativo (para comparação).
              Salva em progressive_silesia_no_reset_K{kmax}.csv.
"""

import csv
import pathlib
import time

from ppmc.utils      import BitWriter
from ppmc.arithmetic import ArithmeticEncoder
from ppmc.model      import HashPPMModel
from ppmc.encoder    import encode_symbol, encode_reset_token
from ppmc.monitor    import ResetMonitor

CORPUS_DIR  = pathlib.Path("corpus/silesia")
RESULTS_DIR = pathlib.Path("results/hash")


def load_silesia() -> bytearray:
    files = sorted(f for f in CORPUS_DIR.iterdir() if f.is_file())
    print("Concatenando arquivos do Silesia...")
    all_data = bytearray()
    for f in files:
        chunk = f.read_bytes()
        print(f"  {f.name:12s}  {len(chunk):>10,} bytes  (offset {len(all_data):,})")
        all_data.extend(chunk)
    print(f"  Total: {len(all_data):,} bytes\n")
    return all_data


def run_silesia_progressive(max_order: int = 5, sample_every: int = 100, use_reset: bool = True):
    """Concatena todos os arquivos do Silesia e rastreia L(n) continuamente."""
    RESULTS_DIR.mkdir(exist_ok=True)

    all_data = load_silesia()
    total_bytes = len(all_data)

    mode_label = "com reset" if use_reset else "sem reset"
    print(f"Comprimindo com Kmax={max_order} ({mode_label})...")
    t0 = time.perf_counter()

    writer  = BitWriter()
    arith   = ArithmeticEncoder(writer)
    model   = HashPPMModel(max_order)
    monitor = ResetMonitor(1000, 50.0) if use_reset else None

    samples: list[tuple[int, float]] = []
    reset_positions: list[int] = []

    for n, byte in enumerate(all_data, start=1):
        encode_symbol(arith, model, byte)
        model.update(byte)

        if monitor is not None:
            monitor.record(writer.bits_written)

        if n % sample_every == 0:
            l_n = writer.bits_written / n
            samples.append((n, round(l_n, 6)))

        if monitor is not None and monitor.should_reset():
            encode_reset_token(arith, model)
            model   = HashPPMModel(max_order)
            monitor.clear(writer.bits_written)
            reset_positions.append(n)

        if n % 1_000_000 == 0:
            print(f"  {n/1e6:.0f}M / {total_bytes/1e6:.0f}M  L(n)={writer.bits_written/n:.4f} bps")

    arith.finish()
    elapsed = time.perf_counter() - t0

    # Salvar CSV de amostras
    suffix = "all" if use_reset else "no_reset"
    out_csv = RESULTS_DIR / f"progressive_silesia_{suffix}_K{max_order}.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'L_n'])
        w.writerows(samples)

    if use_reset:
        # Salvar CSV de posições de reset
        resets_csv = RESULTS_DIR / f"resets_silesia_K{max_order}.csv"
        with open(resets_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['n'])
            for pos in reset_positions:
                w.writerow([pos])
        print(f"  Resets em: {resets_csv}")

    bps_final = writer.bits_written / total_bytes
    print(f"\n  L(n) final: {bps_final:.4f} bps  |  Amostras: {len(samples)}  |  Resets: {len(reset_positions)}  |  Tempo: {elapsed:.1f}s")
    print(f"  Salvo em: {out_csv}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    use_reset = "--no-reset" not in args
    args = [a for a in args if a != "--no-reset"]
    kmax = int(args[0]) if args else 5
    run_silesia_progressive(max_order=kmax, use_reset=use_reset)

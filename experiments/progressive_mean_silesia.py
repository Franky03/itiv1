"""
Comprimento Médio Progressivo sobre o Silesia completo (arquivos concatenados).
Gera progressive_silesia_all_K5.csv para análise de transições entre arquivos.
"""

import csv
import pathlib
import time

from ppmc.utils      import BitWriter
from ppmc.arithmetic import ArithmeticEncoder
from ppmc.model      import PPMModel
from ppmc.encoder    import encode_symbol, encode_reset_token
from ppmc.monitor    import ResetMonitor
from ppmc.compressor import HEADER_SIZE

CORPUS_DIR  = pathlib.Path("corpus/silesia")
RESULTS_DIR = pathlib.Path("results")


def run_silesia_progressive(max_order: int = 5, sample_every: int = 100):
    """Concatena todos os arquivos do Silesia e rastreia L(n) continuamente."""
    RESULTS_DIR.mkdir(exist_ok=True)

    files = sorted(f for f in CORPUS_DIR.iterdir() if f.is_file())

    # Concatenar todos os arquivos
    print("Concatenando arquivos do Silesia...")
    all_data = bytearray()
    for f in files:
        chunk = f.read_bytes()
        print(f"  {f.name:12s}  {len(chunk):>10,} bytes  (offset {len(all_data):,})")
        all_data.extend(chunk)

    total_bytes = len(all_data)
    print(f"  Total: {total_bytes:,} bytes\n")

    # Comprimir com rastreamento
    print(f"Comprimindo com Kmax={max_order}...")
    t0 = time.perf_counter()

    writer  = BitWriter()
    arith   = ArithmeticEncoder(writer)
    model   = PPMModel(max_order)
    monitor = ResetMonitor(1000, 10.0)

    samples: list[tuple[int, float]] = []

    for n, byte in enumerate(all_data, start=1):
        encode_symbol(arith, model, byte)
        model.update(byte)
        monitor.record(writer.bits_written)

        if n % sample_every == 0:
            l_n = writer.bits_written / n
            samples.append((n, round(l_n, 6)))

        if monitor.should_reset():
            encode_reset_token(arith, model)
            model   = PPMModel(max_order)
            monitor.clear()

        if n % 1_000_000 == 0:
            print(f"  {n/1e6:.0f}M / {total_bytes/1e6:.0f}M  L(n)={writer.bits_written/n:.4f} bps")

    arith.finish()
    elapsed = time.perf_counter() - t0

    # Salvar CSV
    out_csv = RESULTS_DIR / f"progressive_silesia_all_K{max_order}.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'L_n'])
        w.writerows(samples)

    bps_final = writer.bits_written / total_bytes
    print(f"\n  L(n) final: {bps_final:.4f} bps  |  Amostras: {len(samples)}  |  Tempo: {elapsed:.1f}s")
    print(f"  Salvo em: {out_csv}")


if __name__ == "__main__":
    import sys
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_silesia_progressive(max_order=kmax)

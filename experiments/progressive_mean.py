import csv
import pathlib
import time

from ppmc.utils        import BitWriter
from ppmc.arithmetic   import ArithmeticEncoder
from ppmc.model        import HashPPMModel
from ppmc.encoder      import encode_symbol, encode_reset_token
from ppmc.monitor      import ResetMonitor
from ppmc.compressor   import HEADER_SIZE

RESULTS_DIR = pathlib.Path("results/hash")


def compress_with_tracking(
    input_data: bytes,
    max_order: int = 5,
    window_size: int = 1000,
    reset_threshold_pct: float = 10.0,
    sample_every: int = 100
) -> tuple[bytes, list[tuple[int, float]]]:
    """
    Versão instrumentada do compressor.
    Retorna (dados_comprimidos, amostras)
    onde amostras = lista de (posição_n, L(n)).
    """
    import struct
    from ppmc.compressor import MAGIC, HEADER_FORMAT, BACKEND_HASH

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC, max_order, len(input_data), window_size, int(reset_threshold_pct),
        BACKEND_HASH
    )

    writer  = BitWriter()
    arith   = ArithmeticEncoder(writer)
    model   = HashPPMModel(max_order)
    monitor = ResetMonitor(window_size, reset_threshold_pct)

    samples: list[tuple[int, float]] = []

    for n, byte in enumerate(input_data, start=1):
        encode_symbol(arith, model, byte)
        model.update(byte)
        monitor.record(writer.bits_written)

        # Amostra o L(n) a cada sample_every bytes
        if n % sample_every == 0:
            bits_so_far = writer.bits_written
            l_n = bits_so_far / n
            samples.append((n, round(l_n, 6)))

        if monitor.should_reset():
            encode_reset_token(arith, model)
            model   = HashPPMModel(max_order)
            monitor.clear()

    arith.finish()
    bitstream = writer.flush()
    return header + bitstream, samples


def run_and_save(filepath: pathlib.Path, max_order: int = 5, sample_every: int = 10):
    """Comprime o arquivo e salva as amostras de L(n) em CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)

    data = filepath.read_bytes()
    print(f"Processando {filepath.name} ({len(data):,} bytes) com Kmax={max_order}...")

    t0 = time.perf_counter()
    compressed, samples = compress_with_tracking(
        data,
        max_order=max_order,
        sample_every=sample_every
    )
    elapsed = time.perf_counter() - t0

    out_csv = RESULTS_DIR / f"progressive_{filepath.name}_K{max_order}.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'L_n'])
        w.writerows(samples)

    bps_final = (len(compressed) - HEADER_SIZE) * 8 / len(data)
    print(f"  L(n) final: {bps_final:.4f} bps  |  Amostras: {len(samples)}  |  Tempo: {elapsed:.1f}s")
    print(f"  Salvo em: {out_csv}")
    return out_csv


if __name__ == "__main__":
    import sys
    corpus_dir = pathlib.Path("corpus/silesia")

    if len(sys.argv) > 1:
        filepath = corpus_dir / sys.argv[1]
        kmax     = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    else:
        filepath = corpus_dir / "dickens"
        kmax     = 5

    run_and_save(filepath, max_order=kmax)
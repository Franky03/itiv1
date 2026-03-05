import csv
import pathlib
import time

from ppmc.compressor   import compress, HEADER_SIZE
from ppmc.decompressor import decompress
from ppmc.parallel     import compress_parallel, decompress_parallel

CORPUS_DIR  = pathlib.Path("corpus/silesia")
RESULTS_DIR = pathlib.Path("results/hash")
OUTPUT_CSV  = RESULTS_DIR / "benchmark.csv"

FIELDNAMES = [
    'file', 'kmax', 'mode', 'chunk_size_kb', 'num_workers',
    'original_bytes', 'compressed_bytes',
    'bits_per_symbol', 'compress_time_s', 'decompress_time_s',
]


def benchmark_one(filepath: pathlib.Path, kmax: int,
                  parallel: bool = False, chunk_size: int = 1024 * 1024,
                  num_workers: int | None = None) -> dict:
    data = filepath.read_bytes()
    n    = len(data)

    if parallel:
        # Compressão paralela por blocos
        t0         = time.perf_counter()
        compressed = compress_parallel(data, max_order=kmax, window_size=1000,
                                       chunk_size=chunk_size, num_workers=num_workers)
        t_compress = time.perf_counter() - t0

        t0           = time.perf_counter()
        recovered    = decompress_parallel(compressed, num_workers=num_workers)
        t_decompress = time.perf_counter() - t0

        # Tamanho comprimido = total - header paralelo (8 bytes + 4*num_chunks)
        num_chunks = len(range(0, n, chunk_size))
        overhead   = 8 + 4 * num_chunks
        m          = len(compressed) - overhead
        mode       = 'parallel'
    else:
        # Compressão sequencial original
        t0         = time.perf_counter()
        compressed = compress(data, max_order=kmax, window_size=1000)
        t_compress = time.perf_counter() - t0

        t0           = time.perf_counter()
        recovered    = decompress(compressed)
        t_decompress = time.perf_counter() - t0

        m    = len(compressed) - HEADER_SIZE
        mode = 'sequential'

    # Integridade
    assert recovered == data, f"FALHOU round-trip: {filepath.name} Kmax={kmax} mode={mode}"

    bps = (m * 8) / n

    return {
        'file':              filepath.name,
        'kmax':              kmax,
        'mode':              mode,
        'chunk_size_kb':     chunk_size // 1024 if parallel else 0,
        'num_workers':       num_workers or 0,
        'original_bytes':    n,
        'compressed_bytes':  m,
        'bits_per_symbol':   round(bps, 4),
        'compress_time_s':   round(t_compress, 3),
        'decompress_time_s': round(t_decompress, 3),
    }


def run_benchmark(files=None, kmax_range=range(11),
                  parallel=False, chunk_size=1024*1024, num_workers=None):
    """
    files: lista de nomes de arquivo (None = todos no corpus)
    kmax_range: intervalo de Kmax a testar
    parallel: usar compressão paralela por blocos
    chunk_size: tamanho de cada bloco (padrão 1 MB)
    num_workers: número de processos paralelos (None = cpu_count)
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    all_files = sorted(CORPUS_DIR.iterdir()) if files is None else [
        CORPUS_DIR / f for f in files
    ]

    mode_str = "PARALELO" if parallel else "SEQUENCIAL"
    print(f"\n{'='*60}")
    print(f"  Benchmark PPMC — modo {mode_str}")
    if parallel:
        from multiprocessing import cpu_count as _cc
        w = num_workers or _cc()
        print(f"  Chunk: {chunk_size // 1024} KB | Workers: {w}")
    print(f"{'='*60}\n")

    results = []

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for filepath in all_files:
            if not filepath.is_file():
                continue
            for kmax in kmax_range:
                print(f"  {filepath.name:12s} Kmax={kmax:2d}...", end=" ", flush=True)
                try:
                    r = benchmark_one(filepath, kmax, parallel=parallel,
                                      chunk_size=chunk_size, num_workers=num_workers)
                    writer.writerow(r)
                    csvfile.flush()
                    results.append(r)
                    print(f"{r['bits_per_symbol']:.4f} bps  "
                          f"({r['compress_time_s']:.1f}s / {r['decompress_time_s']:.1f}s)")
                except Exception as e:
                    print(f"ERRO: {e}")

    print(f"\nResultados salvos em: {OUTPUT_CSV}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PPMC")
    parser.add_argument('files', nargs='*', default=None,
                        help="Arquivos do corpus para testar (todos se omitido)")
    parser.add_argument('--parallel', '-p', action='store_true',
                        help="Usar compressão paralela por blocos")
    parser.add_argument('--chunk-size', '-c', type=int, default=1024,
                        help="Tamanho do bloco em KB (padrão: 1024 = 1 MB)")
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help="Número de processos paralelos (padrão: cpu_count)")
    parser.add_argument('--kmax', '-k', type=int, nargs=2, default=[0, 11],
                        metavar=('MIN', 'MAX'),
                        help="Intervalo de Kmax (padrão: 0 10)")

    args = parser.parse_args()

    run_benchmark(
        files=args.files or None,
        kmax_range=range(args.kmax[0], args.kmax[1]),
        parallel=args.parallel,
        chunk_size=args.chunk_size * 1024,
        num_workers=args.workers,
    )

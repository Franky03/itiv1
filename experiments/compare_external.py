import csv
import pathlib
import subprocess
import tempfile
import time

CORPUS_DIR  = pathlib.Path("corpus/silesia")
RESULTS_DIR = pathlib.Path("results/hash")
OUTPUT_CSV  = RESULTS_DIR / "external_comparison.csv"


def compress_gzip(data: bytes, level: int = 9) -> tuple[int, float]:
    """Retorna (bytes_comprimidos, tempo_em_segundos)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(data)
        fname = f.name

    t0  = time.perf_counter()
    out = subprocess.run(
        ['gzip', f'-{level}', '-c', fname],
        capture_output=True
    )
    elapsed = time.perf_counter() - t0

    pathlib.Path(fname).unlink(missing_ok=True)
    return len(out.stdout), elapsed


def compress_7zip(data: bytes) -> tuple[int, float]:
    """Retorna (bytes_comprimidos, tempo_em_segundos)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = pathlib.Path(tmpdir) / "input.bin"
        out_path = pathlib.Path(tmpdir) / "output.7z"
        in_path.write_bytes(data)

        t0 = time.perf_counter()
        subprocess.run(
            ['7z', 'a', '-mx=9', '-mmt=1', str(out_path), str(in_path)],
            capture_output=True
        )
        elapsed = time.perf_counter() - t0

        size = out_path.stat().st_size if out_path.exists() else -1
        return size, elapsed


def run_comparison():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['file', 'original_bytes', 'gzip9_bytes', 'gzip9_bps',
                      'gzip9_time_s', '7z_bytes', '7z_bps', '7z_time_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in sorted(CORPUS_DIR.iterdir()):
            if not filepath.is_file():
                continue
            data = filepath.read_bytes()
            n    = len(data)

            gz_size, gz_time = compress_gzip(data)
            sz_size, sz_time = compress_7zip(data)

            row = {
                'file':            filepath.name,
                'original_bytes':  n,
                'gzip9_bytes':     gz_size,
                'gzip9_bps':       round((gz_size * 8) / n, 4),
                'gzip9_time_s':    round(gz_time, 3),
                '7z_bytes':        sz_size,
                '7z_bps':          round((sz_size * 8) / n, 4),
                '7z_time_s':       round(sz_time, 3),
            }
            writer.writerow(row)
            csvfile.flush()
            print(f"  {filepath.name:12s}  gzip: {row['gzip9_bps']:.3f} bps  "
                  f"7z: {row['7z_bps']:.3f} bps")

    print(f"\nResultados salvos em: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_comparison()
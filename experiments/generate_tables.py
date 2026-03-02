import csv
import pathlib

RESULTS_DIR = pathlib.Path("results")


def table_bps_by_kmax(filename: str):
    """Tabela: Kmax × {bps, t_compress, t_decompress} para um arquivo."""
    rows = [
        r for r in csv.DictReader(open(RESULTS_DIR / "benchmark.csv"))
        if r['file'] == filename
    ]
    rows.sort(key=lambda r: int(r['kmax']))

    print(f"\n### {filename} — PPM-C por Kmax\n")
    print("| Kmax | bits/símbolo | Compressão (s) | Descompressão (s) |")
    print("|------|-------------|----------------|-------------------|")
    for r in rows:
        print(f"| {int(r['kmax']):>4} | "
              f"{float(r['bits_per_symbol']):>11.4f} | "
              f"{float(r['compress_time_s']):>14.3f} | "
              f"{float(r['decompress_time_s']):>17.3f} |")


def table_comparison(kmax: int = 5):
    """Tabela: todos os arquivos × {PPM-C, gzip, 7z}."""
    ppmc = {r['file']: r for r in csv.DictReader(open(RESULTS_DIR / "benchmark.csv"))
            if int(r['kmax']) == kmax}
    ext  = {r['file']: r for r in csv.DictReader(open(RESULTS_DIR / "external_comparison.csv"))}

    print(f"\n### Comparação de Compressores (PPM-C Kmax={kmax}, gzip -9, 7z -mx=9)\n")
    print("| Arquivo    | Original (MB) | PPM-C bps | gzip bps | 7z bps |")
    print("|------------|---------------|-----------|----------|--------|")
    for fname in sorted(set(ppmc) & set(ext)):
        orig_mb = int(ppmc[fname]['original_bytes']) / 1e6
        print(f"| {fname:10s} | {orig_mb:>13.1f} | "
              f"{float(ppmc[fname]['bits_per_symbol']):>9.4f} | "
              f"{float(ext[fname]['gzip9_bps']):>8.4f} | "
              f"{float(ext[fname]['7z_bps']):>6.4f} |")


if __name__ == "__main__":
    table_bps_by_kmax("dickens")
    table_comparison(kmax=5)
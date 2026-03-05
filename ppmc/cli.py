import argparse
import pathlib
import sys
import time

from ppmc.compressor   import compress, get_compression_stats
from ppmc.decompressor import decompress


def cmd_compress(args):
    input_path  = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Erro: arquivo '{args.input}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    data = input_path.read_bytes()

    t0 = time.perf_counter()
    compressed = compress(
        data,
        max_order         = args.order,
        window_size       = args.window,
        reset_threshold_pct = args.threshold,
        backend           = args.backend,
    )
    elapsed = time.perf_counter() - t0

    output_path.write_bytes(compressed)
    stats = get_compression_stats(data, compressed)

    print(f"Comprimido: {input_path.name}")
    print(f"  Original:    {stats['original_bytes']:>10} bytes")
    print(f"  Comprimido:  {stats['compressed_bytes']:>10} bytes")
    print(f"  Razão:       {stats['ratio']:.4f}  ({stats['bits_per_symbol']:.4f} bits/símbolo)")
    print(f"  Tempo:       {elapsed:.3f}s")
    print(f"  Saída:       {output_path}")


def cmd_decompress(args):
    input_path  = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Erro: arquivo '{args.input}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    compressed = input_path.read_bytes()

    t0 = time.perf_counter()
    data = decompress(compressed)
    elapsed = time.perf_counter() - t0

    output_path.write_bytes(data)
    print(f"Descomprimido: {input_path.name}")
    print(f"  Recuperado: {len(data)} bytes em {elapsed:.3f}s")
    print(f"  Saída:      {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog='ppmc',
        description='Compressor PPM-C com mecanismo de reset adaptativo'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── Subcomando: compress ─────────────────────────────────────────────────
    p_enc = sub.add_parser('compress', help='Comprimir um arquivo')
    p_enc.add_argument('input',  help='Arquivo de entrada')
    p_enc.add_argument('output', help='Arquivo de saída (.ppmc)')
    p_enc.add_argument('--backend', choices=['trie', 'hash'], default='hash', help='Estrutura de dados do modelo (padrão: hash)')
    p_enc.add_argument('--order',     type=int,   default=5,    help='Kmax (padrão: 5)')
    p_enc.add_argument('--window',    type=int,   default=1000, help='Tamanho da janela j (padrão: 1000)')
    p_enc.add_argument('--threshold', type=float, default=10.0, help='Limiar de reset em %% (padrão: 10.0)')
    p_enc.set_defaults(func=cmd_compress)

    # ── Subcomando: decompress ───────────────────────────────────────────────
    p_dec = sub.add_parser('decompress', help='Descomprimir um arquivo')
    p_dec.add_argument('input',  help='Arquivo .ppmc de entrada')
    p_dec.add_argument('output', help='Arquivo de saída recuperado')
    p_dec.set_defaults(func=cmd_decompress)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
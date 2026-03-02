"""
Compressão PPMC paralela por blocos (chunking).

Divide os dados em blocos de tamanho fixo e comprime cada bloco
de forma independente em processos separados. Na descompressão,
cada bloco é descomprimido independentemente e os resultados
são concatenados.

Formato do arquivo:
    MAGIC_PARALLEL (4 bytes)  = b'PPKC'
    num_chunks     (4 bytes)  = uint32 big-endian
    chunk_sizes    (num_chunks * 4 bytes) = tamanho comprimido de cada bloco
    chunk_data     (variável) = blocos comprimidos concatenados
"""

import struct
from multiprocessing import Pool, cpu_count
from ppmc.compressor import compress
from ppmc.decompressor import decompress

MAGIC_PARALLEL = b'PPKC'
DEFAULT_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB


def _compress_chunk(args):
    """Worker: comprime um bloco. Recebe (chunk_bytes, max_order, window_size, reset_threshold_pct)."""
    chunk, max_order, window_size, reset_threshold_pct = args
    return compress(chunk, max_order=max_order, window_size=window_size,
                    reset_threshold_pct=reset_threshold_pct)


def _decompress_chunk(compressed_chunk):
    """Worker: descomprime um bloco."""
    return decompress(compressed_chunk)


def compress_parallel(
    input_data: bytes,
    max_order: int = 5,
    window_size: int = 1000,
    reset_threshold_pct: float = 10.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_workers: int | None = None,
) -> bytes:
    """
    Comprime input_data dividindo em blocos e comprimindo em paralelo.

    Args:
        input_data: dados a comprimir
        max_order: ordem máxima do modelo PPM (Kmax)
        window_size: tamanho da janela do monitor de reset
        reset_threshold_pct: limiar de degradação para reset
        chunk_size: tamanho de cada bloco em bytes (padrão 1 MB)
        num_workers: número de processos (None = cpu_count())

    Returns:
        bytes comprimidos no formato paralelo
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Dividir em blocos
    chunks = [input_data[i:i + chunk_size]
              for i in range(0, len(input_data), chunk_size)]
    num_chunks = len(chunks)

    # Preparar argumentos para cada worker
    args = [(chunk, max_order, window_size, reset_threshold_pct)
            for chunk in chunks]

    # Comprimir em paralelo
    with Pool(processes=num_workers) as pool:
        compressed_chunks = pool.map(_compress_chunk, args)

    # Montar saída: header + tamanhos + dados
    header = MAGIC_PARALLEL + struct.pack('>I', num_chunks)
    sizes = b''.join(struct.pack('>I', len(c)) for c in compressed_chunks)
    data = b''.join(compressed_chunks)

    return header + sizes + data


def decompress_parallel(
    compressed_data: bytes,
    num_workers: int | None = None,
) -> bytes:
    """
    Descomprime dados comprimidos com compress_parallel.

    Args:
        compressed_data: dados no formato paralelo
        num_workers: número de processos (None = cpu_count())

    Returns:
        dados originais reconstruídos
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Ler header
    if len(compressed_data) < 8:
        raise ValueError("Dados comprimidos muito curtos.")

    magic = compressed_data[:4]
    if magic != MAGIC_PARALLEL:
        raise ValueError(f"Magic inválido: {magic!r} (esperado {MAGIC_PARALLEL!r})")

    num_chunks = struct.unpack('>I', compressed_data[4:8])[0]

    # Ler tamanhos dos blocos
    sizes_start = 8
    sizes_end = sizes_start + num_chunks * 4
    if len(compressed_data) < sizes_end:
        raise ValueError("Dados comprimidos truncados (tabela de tamanhos).")

    chunk_sizes = [
        struct.unpack('>I', compressed_data[sizes_start + i * 4: sizes_start + (i + 1) * 4])[0]
        for i in range(num_chunks)
    ]

    # Extrair cada bloco comprimido
    offset = sizes_end
    compressed_chunks = []
    for size in chunk_sizes:
        compressed_chunks.append(compressed_data[offset:offset + size])
        offset += size

    # Descomprimir em paralelo
    with Pool(processes=num_workers) as pool:
        decompressed_chunks = pool.map(_decompress_chunk, compressed_chunks)

    return b''.join(decompressed_chunks)

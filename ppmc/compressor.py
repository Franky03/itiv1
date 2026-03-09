import struct
from ppmc.utils import BitWriter
from ppmc.arithmetic import ArithmeticEncoder
from ppmc.model import PPMModel, HashPPMModel
from ppmc.encoder import encode_symbol, encode_reset_token
from ppmc.monitor import ResetMonitor

MAGIC = b'PPMC'
HEADER_FORMAT = '>4sBIHBB'  # magic(4) + kmax(1) + size(4) + j(2) + thr(1) + backend(1)
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)   # = 13

BACKEND_TRIE = 0
BACKEND_HASH = 1

def _create_model(backend: str, max_order: int):
    if backend == 'hash':
        return HashPPMModel(max_order)
    return PPMModel(max_order)

def compress(
    input_data: bytes,
    max_order: int = 5,
    window_size: int = 1000,
    reset_threshold_pct: float = 200.0,
    backend: str = 'hash'
) -> bytes:
    """
    Comprime input_data usando PPM-C.
    Retorna os bytes comprimidos (header + bitstream).
    """
    backend_byte = BACKEND_HASH if backend == 'hash' else BACKEND_TRIE

    # ── Header ───────────────────────────────────────────────────────────────
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        max_order,
        len(input_data),
        window_size,
        int(reset_threshold_pct),
        backend_byte
    )

    # ── Codificação ───────────────────────────────────────────────────────────
    writer  = BitWriter()
    arith   = ArithmeticEncoder(writer)
    model   = _create_model(backend, max_order)
    monitor = ResetMonitor(window_size, reset_threshold_pct)

    for byte in input_data:
        encode_symbol(arith, model, byte)
        model.update(byte)
        monitor.record(writer.bits_written)

        if monitor.should_reset():
            # 1. Emite token RESET no bitstream
            encode_reset_token(arith, model)
            # 2. Reinicia o modelo (descarta todo o histórico)
            model   = _create_model(backend, max_order)
            # 3. Reinicia o monitor
            monitor.clear(writer.bits_written)

    arith.finish()
    bitstream = writer.flush()

    return header + bitstream

def get_compression_stats(input_data: bytes, compressed: bytes) -> dict:
    """Retorna dicionário com métricas de compressão."""
    n = len(input_data)
    m = len(compressed) - HEADER_SIZE
    return {
        'original_bytes':    n,
        'compressed_bytes':  m,
        'ratio':             m / n if n > 0 else 0.0,
        'bits_per_symbol':   (m * 8) / n if n > 0 else 0.0,
    }
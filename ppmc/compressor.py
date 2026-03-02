import struct
from ppmc.utils import BitWriter
from ppmc.arithmetic import ArithmeticEncoder
from ppmc.model import PPMModel
from ppmc.encoder import encode_symbol, encode_reset_token
from ppmc.monitor import ResetMonitor

MAGIC = b'PPMC'
HEADER_FORMAT = '>4sBIHB'   # magic(4) + kmax(1) + size(4) + j(2) + thr(1)
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)   # = 12

def compress(
    input_data: bytes,
    max_order: int = 5,
    window_size: int = 1000,
    reset_threshold_pct: float = 10.0
) -> bytes:
    """
    Comprime input_data usando PPM-C.
    Retorna os bytes comprimidos (header + bitstream).
    """
    # ── Header ───────────────────────────────────────────────────────────────
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        max_order,
        len(input_data),
        window_size,
        int(reset_threshold_pct)
    )

    # ── Codificação ───────────────────────────────────────────────────────────
    writer  = BitWriter()
    arith   = ArithmeticEncoder(writer)
    model   = PPMModel(max_order)
    monitor = ResetMonitor(window_size, reset_threshold_pct)

    for byte in input_data:
        bits_before = writer.bits_written
        encode_symbol(arith, model, byte)
        model.update(byte)
        monitor.record(writer.bits_written - bits_before)

        if monitor.should_reset():
            # 1. Emite token RESET no bitstream
            encode_reset_token(arith, model)
            # 2. Reinicia o modelo (descarta todo o histórico)
            model   = PPMModel(max_order)
            # 3. Reinicia o monitor
            monitor.clear()

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
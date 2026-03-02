import struct
from ppmc.utils import BitReader
from ppmc.arithmetic import ArithmeticDecoder
from ppmc.model import PPMModel
from ppmc.decoder import decode_symbol
from ppmc.compressor import MAGIC, HEADER_FORMAT, HEADER_SIZE

RESET_TOKEN = 256


def decompress(compressed_data: bytes) -> bytes:
    """
    Descomprime dados comprimidos pelo compressor PPM-C.
    Lança ValueError se o magic number não bater.
    """
    if len(compressed_data) < HEADER_SIZE:
        raise ValueError("Dados comprimidos muito curtos para conter header.")

    # ── Ler header ────────────────────────────────────────────────────────────
    magic, max_order, original_size, window_size, threshold = struct.unpack(
        HEADER_FORMAT, compressed_data[:HEADER_SIZE]
    )

    if magic != MAGIC:
        raise ValueError(f"Magic inválido: {magic!r} (esperado {MAGIC!r})")

    # ── Decodificação ─────────────────────────────────────────────────────────
    reader = BitReader(compressed_data[HEADER_SIZE:])
    arith  = ArithmeticDecoder(reader)
    model  = PPMModel(max_order)

    result = []

    while len(result) < original_size:
        sym = decode_symbol(arith, model)

        if sym == RESET_TOKEN:
            # Token de reset: reinicia o modelo e NÃO adiciona nada à saída
            model = PPMModel(max_order)
            continue

        result.append(sym)
        model.update(sym)

    return bytes(result)
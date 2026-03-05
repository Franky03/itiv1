import struct
from ppmc.utils import BitReader
from ppmc.arithmetic import ArithmeticDecoder
from ppmc.model import PPMModel, HashPPMModel
from ppmc.decoder import decode_symbol
from ppmc.compressor import MAGIC, HEADER_FORMAT, HEADER_SIZE, BACKEND_HASH

RESET_TOKEN = 256


def decompress(compressed_data: bytes) -> bytes:
    """
    Descomprime dados comprimidos pelo compressor PPM-C.
    Lança ValueError se o magic number não bater.
    """
    if len(compressed_data) < HEADER_SIZE:
        raise ValueError("Dados comprimidos muito curtos para conter header.")

    # ── Ler header ────────────────────────────────────────────────────────────
    magic, max_order, original_size, window_size, threshold, backend_byte = struct.unpack(
        HEADER_FORMAT, compressed_data[:HEADER_SIZE]
    )

    if magic != MAGIC:
        raise ValueError(f"Magic inválido: {magic!r} (esperado {MAGIC!r})")

    backend = 'hash' if backend_byte == BACKEND_HASH else 'trie'

    def make_model():
        if backend == 'hash':
            return HashPPMModel(max_order)
        return PPMModel(max_order)

    # ── Decodificação ─────────────────────────────────────────────────────────
    reader = BitReader(compressed_data[HEADER_SIZE:])
    arith  = ArithmeticDecoder(reader)
    model  = make_model()

    result = []

    while len(result) < original_size:
        sym = decode_symbol(arith, model)

        if sym == RESET_TOKEN:
            model = make_model()
            continue

        result.append(sym)
        model.update(sym)

    return bytes(result)
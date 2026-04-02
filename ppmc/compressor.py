import copy
import math
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

# ── Interface A: compressão simples de texto ──────────────────────────────────

def compress_text(
    text: str,
    max_order: int = 5,
    backend: str = 'hash',
) -> dict:
    """
    Comprime um texto (str) e retorna métricas.

    Retorna dict com:
        compressed_size : int   — bytes após compressão (sem header)
        original_size   : int   — bytes do texto em UTF-8
        ratio           : float — compressed_size / original_size
        bpc             : float — bits por caractere (compressed_size*8 / len(text))
    """
    if not text:
        return {'compressed_size': 0, 'original_size': 0, 'ratio': 0.0, 'bpc': 0.0}

    data = text.encode('utf-8')
    writer = BitWriter()
    arith  = ArithmeticEncoder(writer)
    model  = _create_model(backend, max_order)

    for byte in data:
        encode_symbol(arith, model, byte)
        model.update(byte)

    arith.finish()
    bitstream = writer.flush()

    compressed_size = len(bitstream)
    original_size   = len(data)
    n_chars         = len(text)

    return {
        'compressed_size': compressed_size,
        'original_size':   original_size,
        'ratio':           compressed_size / original_size if original_size > 0 else 0.0,
        'bpc':             (compressed_size * 8) / n_chars if n_chars > 0 else 0.0,
    }


# ── Interface B: treino em corpus + compressão cruzada ───────────────────────

def train(
    corpus: str,
    max_order: int = 5,
    backend: str = 'hash',
):
    """
    Treina o modelo PPM em um corpus e retorna o objeto de modelo.
    O modelo pode ser reutilizado em compress_with_model.

    O estado retornado é um snapshot "congelado" do modelo após ver todo
    o corpus; ele não é modificado por chamadas subsequentes a compress_with_model.
    """
    data  = corpus.encode('utf-8')
    model = _create_model(backend, max_order)
    for byte in data:
        model.update(byte)
    # Congela o contexto deslizante para que cada nova compressão comece do zero
    model.context = []
    return model


def compress_with_model(model, text: str) -> dict:
    """
    Comprime text usando um modelo pré-treinado (retornado por train).

    O modelo original NÃO é modificado — uma cópia profunda é usada
    internamente para cada chamada.

    Retorna o mesmo dict que compress_text.
    """
    if not text:
        return {'compressed_size': 0, 'original_size': 0, 'ratio': 0.0, 'bpc': 0.0}

    data       = text.encode('utf-8')
    model_copy = copy.deepcopy(model)
    model_copy.context = []   # contexto deslizante limpo para o novo texto

    writer = BitWriter()
    arith  = ArithmeticEncoder(writer)

    for byte in data:
        encode_symbol(arith, model_copy, byte)
        model_copy.update(byte)

    arith.finish()
    bitstream = writer.flush()

    compressed_size = len(bitstream)
    original_size   = len(data)
    n_chars         = len(text)

    return {
        'compressed_size': compressed_size,
        'original_size':   original_size,
        'ratio':           compressed_size / original_size if original_size > 0 else 0.0,
        'bpc':             (compressed_size * 8) / n_chars if n_chars > 0 else 0.0,
    }

# ── Interface C: scoring rápido com modelo congelado ─────────────────────────

def bpc_frozen(model, text: str) -> float:
    """
    Calcula bits-por-caractere usando um modelo PPM congelado.

    NÃO modifica o modelo e NÃO faz deepcopy — usa lookup direto nas
    tabelas de probabilidade com math.log2.  Ordens de magnitude mais
    rápido que compress_with_model para classificação.

    Mantém um contexto local separado do model.context, portanto é
    thread-safe e pode ser chamado em paralelo sem cópias.
    """
    if not text:
        return 0.0

    data      = text.encode('utf-8')
    is_hash   = isinstance(model, HashPPMModel)
    local_ctx: list[int] = []
    total_bits = 0.0

    def _get_entry(order: int):
        if is_hash:
            key = tuple(local_ctx[-order:]) if order > 0 else ()
            return model.contexts.get(key)
        else:  # PPMModel (trie)
            if order == 0:
                return model.root
            node = model.root
            for b in local_ctx[-order:]:
                node = node.children.get(b)
                if node is None:
                    return None
            return node

    for byte in data:
        exclusion_set: set[int] = set()
        found = False

        for order in range(model.max_order, -1, -1):
            entry = _get_entry(order)
            if entry is None or not entry.counts:
                continue

            symbols, cum_freqs, total, sym_to_idx = model.get_distribution(
                entry, exclusion_set
            )
            if total == 0:
                continue

            if byte in sym_to_idx:
                idx  = sym_to_idx[byte]
                prob = (cum_freqs[idx + 1] - cum_freqs[idx]) / total
                total_bits += -math.log2(prob) if prob > 0 else 32.0
                found = True
                break
            else:
                # escape: acumula custo e desce de ordem
                esc = entry.escape_count
                if esc > 0:
                    total_bits += -math.log2(esc / total)
                exclusion_set.update(entry.counts.keys())

        if not found:
            # fallback ordem -1: uniforme sobre bytes não excluídos
            available = max(1, 256 - len(exclusion_set))
            total_bits += math.log2(available)

        local_ctx.append(byte)
        if len(local_ctx) > model.max_order:
            local_ctx.pop(0)

    n_chars = len(text)
    return total_bits / n_chars if n_chars > 0 else 0.0

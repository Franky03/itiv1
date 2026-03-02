from ppmc.arithmetic import ArithmeticDecoder
from ppmc.model import PPMModel


def decode_symbol(arith: ArithmeticDecoder, model: PPMModel) -> int:
    """
    Decodifica um byte usando PPM-C com exclusão.
    Deve seguir exatamente o mesmo caminho de decisão que encode_symbol.
    """
    exclusion_set: set[int] = set()

    for order in range(model.max_order, -2, -1):

        # ── Ordem -1: equiprobabilidade ────────────────────────────────────
        if order == -1:
            available = [s for s in range(256) if s not in exclusion_set]
            available.append(256) # RESET sempre disponivel em -1
            n = len(available)
            cum_uniform = list(range(n + 1))   # [0, 1, 2, ..., n]
            idx = arith.decode_symbol(cum_uniform, n)
            return available[idx]

        # ── Ordens 0..Kmax ─────────────────────────────────────────────────
        node = model.get_context_node(order)

        if node is None or not node.counts:
            continue

        symbols, cum_freqs, total = model.get_distribution(node, exclusion_set)

        decoded_idx = arith.decode_symbol(cum_freqs, total)

        if decoded_idx < len(symbols):
            # Símbolo real decodificado
            return symbols[decoded_idx]
        else:
            # ESC decodificado: desce de ordem
            exclusion_set.update(node.counts.keys())
            # Continua para ordem inferior
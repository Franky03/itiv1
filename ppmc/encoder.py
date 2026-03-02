from ppmc.arithmetic import ArithmeticEncoder
from ppmc.model import PPMModel

def encode_symbol(arith: ArithmeticEncoder, model: PPMModel, symbol: int) -> None:
    """
    Codifica um byte usando PPM-C com exclusão.
    Percorre ordens de Kmax até -1, caindo para ordem inferior via ESC quando necessário.

    Analogia com o Huffman Contextual:
      - K=1 → K=0 → K=-1 (exclusão acumulada)
      - A diferença: em vez de build_huffman_tree + codes[symbol],
        chamamos arith.encode_symbol(cum_low, cum_high, total)
    """

    exclusion_set: set[int] = set()

    for order in range(model.max_order, -2, -1):

        # ── Ordem -1: equiprobabilidade sobre bytes não excluídos ──────────
        if order == -1:
            available = [s for s in range(256) if s not in exclusion_set]
            available.append(256)   # RESET sempre disponível
            n = len(available)
            idx = available.index(symbol)   # symbol pode ser 256 (RESET)
            arith.encode_symbol(idx, idx + 1, n)
            return
        
        # ── Ordens 0..Kmax ─────────────────────────────────────────────────
        node = model.get_context_node(order)

        # Nó inexistente ou vazio: pula para ordem inferior sem emitir bits
        if node is None or not node.counts:
            continue

        symbols, cum_freqs, total, sym_to_idx = model.get_distribution(node, exclusion_set)

        if symbol in sym_to_idx:
            # Símbolo encontrado: codifica diretamente
            idx = sym_to_idx[symbol]
            arith.encode_symbol(cum_freqs[idx], cum_freqs[idx + 1], total)
            return
        else:
            # Símbolo não encontrado: codifica ESC e desce de ordem
            esc_idx = len(symbols)   # ESC é o último elemento implícito
            arith.encode_symbol(cum_freqs[esc_idx], cum_freqs[esc_idx + 1], total)
            # Adiciona símbolos vistos neste nó ao conjunto de exclusão
            exclusion_set.update(node.counts.keys())
            # Continua para a próxima iteração (ordem inferior)
            
def encode_reset_token(arith: ArithmeticEncoder, model: PPMModel) -> None:
    """
    Emite o token RESET no bitstream.
    Usa o mesmo caminho PPM que encode_symbol para manter sincronia com o decoder.
    """
    encode_symbol(arith, model, 256)
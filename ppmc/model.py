class TrieNode:
    """
    Nó da trie de contexto PPM.
    Usa __slots__ para reduzir ~40% do uso de memória
    (em modelos de alta ordem, podem existir milhões de nós).
    """
    __slots__ = ('counts', 'children', 'escape_count', '_cache_key', '_cache_val')

    def __init__(self):
        self.counts: dict[int, int] = {}          # byte → frequência
        self.children: dict[int, 'TrieNode'] = {} # byte → nó filho
        self.escape_count: int = 0                # nº de símbolos distintos (Método C)
        self._cache_key = None   # (frozenset de exclusion_set, total_count)
        self._cache_val = None   # resultado cacheado

class ContextEntry:
    __slots__ = ('counts', 'escape_count', '_cache_key', '_cache_val')
    def __init__(self):
        self.counts: dict[int, int] = {}          # byte → frequência
        self.escape_count: int = 0                # nº de símbolos distintos (Método C)
        self._cache_key = None   # (frozenset de exclusion_set, total_count)
        self._cache_val = None   # resultado cacheado

class HashPPMModel:
    """
    Implementação alternativa de PPM usando dicionário hash para contextos.
    Pode ser mais eficiente em ordens muito altas, onde a trie se torna enorme.
    """
    def __init__(self, max_order: int):
        self.max_order = max_order
        # Hash: uma única tabela
        #   chave = tupla dos bytes do contexto, ex: (97, 98) para ordem 2
        #   valor = { "counts": {byte: freq}, "escape_count": int }
        self.contexts: dict[tuple[int, ...], ContextEntry] = {}
        self.context: list[int] = []    # janela deslizante dos últimos bytes vistos

    def _context_key(self, order: int) -> tuple[int, ...]:
        if order == 0:
            return ()
        return tuple(self.context[-order:])

    def get_context_node(self, order: int) -> ContextEntry | None:
        return self.contexts.get(self._context_key(order))

    def _get_or_create(self, order: int) -> ContextEntry:
        key = self._context_key(order)
        if key not in self.contexts:
            self.contexts[key] = ContextEntry()
        return self.contexts[key]

    def get_distribution(
        self,
        entry: ContextEntry,
        exclusion_set: set[int]
    ) -> tuple[list[int], list[int], int, dict[int, int]]:
        
        cache_key = (frozenset(exclusion_set), sum(entry.counts.values()))
        if entry._cache_key == cache_key:
            return entry._cache_val
        
        symbols = sorted(s for s in entry.counts if s not in exclusion_set)

        cum = [0]
        for s in symbols:
            cum.append(cum[-1] + entry.counts[s])
        
        esc_count = entry.escape_count
        total = cum[-1] + esc_count
        cum.append(total)

        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        result = (symbols, cum, total, sym_to_idx)
        
        entry._cache_key = cache_key
        entry._cache_val = result
        return result

    def update(self, symbol: int):
        max_usable = min(self.max_order, len(self.context))
        for order in range(max_usable + 1):
            entry = self._get_or_create(order)
            if symbol not in entry.counts:
                entry.escape_count += 1
            entry.counts[symbol] = entry.counts.get(symbol, 0) + 1

        self.context.append(symbol)
        if len(self.context) > self.max_order:
            self.context.pop(0)

class PPMModel:
    def __init__(self, max_order: int):
        self.max_order = max_order
        self.root = TrieNode()          # raiz da trie (representa ordem 0)
        self.context: list[int] = []    # janela deslizante dos últimos bytes vistos

    # ── Navegação na trie ──────────────────────────────────────────────────

    def get_context_node(self, order: int) -> TrieNode | None:
        """
        Retorna o nó da trie correspondente ao sufixo do contexto de comprimento 'order'.
        Retorna None se o caminho não existe na trie.

        Exemplo: se context = [97, 98, 99] e order=2,
                 navega root → 98 → 99 e retorna o nó final.
        """
        if order == 0:
            return self.root

        suffix = self.context[-order:]   # últimos 'order' bytes
        node = self.root
        for byte in suffix:
            if byte not in node.children:
                return None
            node = node.children[byte]
        return node
    
    # ── Distribuição de probabilidade ─────────────────────────────────────

    def get_distribution(
        self,
        node: TrieNode,
        exclusion_set: set[int]
    ) -> tuple[list[int], list[int], int, dict[int, int]]:
        """
        Calcula a distribuição de probabilidade para codificação aritmética,
        excluindo símbolos da exclusion_set.

        Retorna:
            symbols    : lista de bytes disponíveis (ordenada para determinismo)
            cum_freqs  : frequências cumulativas, comprimento = len(symbols) + 1
                         (última posição = total, inclui o ESC)
            total      : soma de todas as contagens (símbolos + ESC)
            sym_to_idx : dicionário símbolo → índice em symbols (O(1) lookup)

        O ESC não aparece em 'symbols' mas está implícito no último intervalo
        de cum_freqs (de cum_freqs[-2] até cum_freqs[-1]).
        """

        cache_key = (frozenset(exclusion_set), sum(node.counts.values()))
        if node._cache_key == cache_key:
            return node._cache_val
        
        # Símbolos disponíveis: vistos no nó e não excluídos
        symbols = sorted(s for s in node.counts if s not in exclusion_set)

        cum = [0]
        for s in symbols:
            cum.append(cum[-1] + node.counts[s])
        
        # ESC: contagem = nº de símbolos distintos no nó (Método C)
        # Mesmo que alguns estejam excluídos, o escape_count original é mantido
        # para preservar o sincronismo encoder/decoder
        esc_count = node.escape_count
        total = cum[-1] + esc_count
        cum.append(total)   # intervalo do ESC = [cum[-2], total)

        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        result = (symbols, cum, total, sym_to_idx)
        
        node._cache_key = cache_key
        node._cache_val = result
        return result
    
    # ── Atualização do modelo ─────────────────────────────────────────────

    def update(self, symbol: int):
        """
        Atualiza o modelo após codificar/decodificar 'symbol'.
        Incrementa as contagens em todos os contextos (ordem 0 até max_order).
        Usa o sufixo atual de self.context.
        """
        node = self.root

        # Caminho de nós a atualizar (da raiz até o contexto mais longo)
        path = [node]
        for byte in self.context[-self.max_order:]:
            if byte not in node.children:
                node.children[byte] = TrieNode()
            node = node.children[byte]
            path.append(node)

        # Atualizar cada nó no caminho
        for n in path:
            if symbol not in n.counts:
                n.escape_count += 1   # novo símbolo → incrementa ESC (Método C)
            n.counts[symbol] = n.counts.get(symbol, 0) + 1

        # Avança o contexto deslizante
        self.context.append(symbol)
        if len(self.context) > self.max_order:
            self.context.pop(0)
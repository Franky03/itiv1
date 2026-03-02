PRECISION = 32
FULL      = 1 << PRECISION   # 4_294_967_296
HALF      = FULL >> 1        # 2_147_483_648
QUARTER   = FULL >> 2        # 1_073_741_824

from ppmc.utils import BitWriter
from ppmc.utils import BitReader

class ArithmeticEncoder:
    def __init__(self, writer: BitWriter):
        self._writer = writer
        self._low     = 0
        self._high    = FULL
        self._pending = 0   # contador de bits E3 pendentes

    def encode_symbol(self, cum_low: int, cum_high: int, total: int):
        """
        Codifica um símbolo dado seu intervalo cumulativo [cum_low, cum_high) / total.

        Exemplo: para codificar 'b' numa distribuição {a:3, b:2, ESC:1}, total=6:
          - 'a' ocupa [0, 3)
          - 'b' ocupa [3, 5)
          - ESC ocupa [5, 6)
        Para codificar 'b': encode_symbol(3, 5, 6)
        """
        # 1. Estreitar o intervalo
        range_ = self._high - self._low
        self._high = self._low + (range_ * cum_high) // total
        self._low = self._low + (range_ * cum_low) // total

        # 2. Normalizar: emitir bits enquanto MSBs concordam ou E3
        self._normalize()

    def _emit_bit_and_pending(self, bit: int):
        """Emite 'bit' e em seguida 'self._pending' bits complementares."""
        self._writer.write_bit(bit)
        for _ in range(self._pending):
            self._writer.write_bit(1 - bit)
        self._pending = 0
    
    def _normalize(self):
        while True:
            if self._high <= HALF:
                # Ambos na metade inferior -> emite 0
                self._emit_bit_and_pending(0)
                self._low <<= 1
                self._high = (self._high << 1)
            
            elif self._low >= HALF:
                # Ambos na metade superior -> emite 1
                self._emit_bit_and_pending(1)
                self._low  = (self._low  - HALF) << 1
                self._high = (self._high - HALF) << 1

            elif self._low >= QUARTER and self._high <= 3 * QUARTER:
                # Condição E3: quase convergiu, mas em lados opostos
                self._pending += 1
                self._low  = (self._low  - QUARTER) << 1
                self._high = (self._high - QUARTER) << 1

            else:
                break   # intervalo estável

    def finish(self):
        """Finaliza: emite bits suficientes para identificar o intervalo."""
        self._pending += 1
        if self._low < QUARTER:
            self._emit_bit_and_pending(0)
        else:
            self._emit_bit_and_pending(1)


class ArithmeticDecoder:
    def __init__(self, reader: BitReader):
        self._reader  = reader
        self._low     = 0
        self._high    = FULL
        # Inicializa 'value' lendo os primeiros PRECISION bits
        self._value   = reader.read_bits(PRECISION)

    def decode_symbol(self, cum_freqs: list[int], total: int) -> int:
        """
        Determina qual símbolo foi codificado, dado o vetor de frequências cumulativas.

        cum_freqs tem comprimento (n_símbolos + 1):
          cum_freqs[0] = 0
          cum_freqs[i] = cum_freqs[i-1] + contagem do símbolo i-1
          cum_freqs[-1] = total

        Retorna o índice do símbolo (0-based).
        """

        range_  = self._high - self._low
        # Escalar value para o espaço de frequências
        scaled  = ((self._value - self._low + 1) * total - 1) // range_

        # Busca binária: encontra i tal que cum_freqs[i] <= scaled < cum_freqs[i+1]
        lo, hi = 0, len(cum_freqs) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cum_freqs[mid] <= scaled:
                lo = mid
            else:
                hi = mid - 1
        symbol_idx = lo

        # Atualizar intervalo (idêntico ao encoder)
        self._high = self._low + (range_ * cum_freqs[symbol_idx + 1]) // total
        self._low  = self._low + (range_ * cum_freqs[symbol_idx])     // total

        # Normalizar: ler bits do stream
        self._normalize()
        return symbol_idx

    def _normalize(self):
        while True:
            if self._high <= HALF:
                self._low   <<= 1
                self._high   = (self._high << 1)
                self._value  = (self._value << 1) | self._reader.read_bit()

            elif self._low >= HALF:
                self._low   = (self._low  - HALF) << 1
                self._high  = (self._high - HALF) << 1
                self._value = ((self._value - HALF) << 1) | self._reader.read_bit()

            elif self._low >= QUARTER and self._high <= 3 * QUARTER:
                self._low   = (self._low  - QUARTER) << 1
                self._high  = (self._high - QUARTER) << 1
                self._value = ((self._value - QUARTER) << 1) | self._reader.read_bit()

            else:
                break
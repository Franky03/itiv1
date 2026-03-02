# O codificador aritmético produz bits individuais. Eles precisam ser empacotados em bytes para escrever em arquivos e desempacotados na leitura. 

class BitWriter:
    def __init__(self):
        self._buffer = 0 # byte sendo montado
        self._bits_in_buf_ = 0 # número de bits atualmente no buffer
        self._output_ = bytearray() # bytes escritos
        self._total_bits_written_ = 0 # total de bits escritos

    def write_bit(self, bit: int):
        self._buffer = (self._buffer << 1) | (bit & 1) # adiciona o bit ao buffer 
        # & é AND bit a bit para garantir que apenas o bit menos significativo seja usado
        # << é shift left para mover os bits no buffer para a esquerda, fazendo espaço para o novo bit
        self._bits_in_buf_ += 1
        self._total_bits_written_ += 1
        if self._bits_in_buf_ == 8: # se o buffer estiver cheio (8 bits)
            self._output_.append(self._buffer) # adiciona o byte ao output
            self._buffer = 0 # reseta o buffer
            self._bits_in_buf_ = 0 # reseta o contador de bits no buffer

    def flush(self):
        if self._bits_in_buf_ > 0: # se houver bits restantes no buffer
            self._buffer <<= (8 - self._bits_in_buf_) # shift left para completar o byte
            self._output_.append(self._buffer) # adiciona o byte ao output
            self._buffer = 0 # reseta o buffer
            self._bits_in_buf_ = 0 # reseta o contador de bits no buffer
            
        # Adicione esta linha para retornar o array de bytes convertido
        return bytes(self._output_)

    @property
    def bits_written(self) -> int:
        return self._total_bits_written_ # retorna o total de bits escritos
    
class BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0   # 0 = MSB do byte atual

    def read_bit(self) -> int:
        """Retorna 0 ou 1. Levanta EOFError se acabar os dados."""
        if self._byte_pos >= len(self._data):
            return 0   # bits de padding após o fim do stream
        byte = self._data[self._byte_pos]
        bit = (byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        """Lê n bits e retorna como inteiro (MSB primeiro)."""
        result = 0
        for _ in range(n):
            result = (result << 1) | self.read_bit()
        return result
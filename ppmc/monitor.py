class ResetMonitor:
    """
    Monitora a taxa de compressão local usando janelas deslizantes de tamanho j.
    Dispara reset quando a janela atual piora mais que 'threshold_pct'% em relação
    à janela anterior.
    """

    def __init__(self, window_size: int, threshold_pct: float):
        self.j         = window_size
        self.threshold = threshold_pct / 100.0
        self._snapshots: list[int] = []   # bits_written após cada byte

    def record(self, bits_written: int):
        """Deve ser chamado após codificar cada byte."""
        self._snapshots.append(bits_written)

    def should_reset(self) -> bool:
        """
        Retorna True quando há pelo menos 2 janelas completas e a janela atual
        degradou mais que o limiar.

        Só avalia no exato ponto onde a segunda janela completa (múltiplos de j):
        isso evita disparar múltiplos resets no mesmo evento.
        """
        n = len(self._snapshots)

        # Avalia apenas quando completamos um múltiplo de j (exceto os primeiros j)
        if n < 2 * self.j or n % self.j != 0:
            return False
        
        # Janela anterior: posições [n-2j, n-j)
        bits_prev = self._snapshots[n - self.j - 1] - self._snapshots[n - 2*self.j - 1]
        # Janela atual:    posições [n-j,  n)
        bits_curr = self._snapshots[n - 1]          - self._snapshots[n - self.j - 1]

        avg_prev = bits_prev / self.j
        avg_curr = bits_curr / self.j

        return avg_curr > avg_prev * (1.0 + self.threshold)

    def clear(self):
        """Zera o histórico após um reset."""
        self._snapshots.clear()

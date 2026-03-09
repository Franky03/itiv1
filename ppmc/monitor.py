class ResetMonitor:
    """
    Monitora a taxa de compressão local usando janelas deslizantes de tamanho j.
    Dispara reset quando a janela atual piora mais que 'threshold_pct'% em relação
    à janela anterior.
    """

    WARMUP_WINDOWS = 3   # janelas ignoradas após cada reset (warmup do modelo)

    def __init__(self, window_size: int, threshold_pct: float):
        self.j         = window_size
        self.threshold = threshold_pct / 100.0
        self._snapshots: list[int] = []   # bits_written após cada byte
        self._base_offset: int = 0        # bits_written no momento do clear

    def record(self, bits_written: int):
        """Deve ser chamado após codificar cada byte."""
        self._snapshots.append(bits_written - self._base_offset)

    def should_reset(self) -> bool:
        """
        Retorna True quando há pelo menos 2 janelas completas e a janela atual
        degradou mais que o limiar.

        Só avalia no exato ponto onde a segunda janela completa (múltiplos de j):
        isso evita disparar múltiplos resets no mesmo evento.

        Após um reset, ignora as primeiras WARMUP_WINDOWS janelas para que o
        modelo tenha tempo de se adaptar antes de ser avaliado.
        """
        n = len(self._snapshots)

        # Avalia apenas quando completamos um múltiplo de j (exceto os primeiros j)
        if n < 2 * self.j or n % self.j != 0:
            return False

        # Período de warmup: não avalia nas primeiras janelas após reset
        if n < (self.WARMUP_WINDOWS + 1) * self.j:
            return False

        # Bits acumulados antes da janela anterior (0 se é a primeira janela)
        idx_before_prev = n - 2 * self.j - 1
        base = self._snapshots[idx_before_prev] if idx_before_prev >= 0 else 0

        # Janela anterior: posições [n-2j, n-j)
        bits_prev = self._snapshots[n - self.j - 1] - base
        # Janela atual:    posições [n-j,  n)
        bits_curr = self._snapshots[n - 1]          - self._snapshots[n - self.j - 1]

        avg_prev = bits_prev / self.j
        avg_curr = bits_curr / self.j

        return avg_curr > avg_prev * (1.0 + self.threshold)

    def clear(self, bits_written: int = 0):
        """Zera o histórico após um reset, preservando o offset do writer."""
        self._snapshots.clear()
        self._base_offset = bits_written

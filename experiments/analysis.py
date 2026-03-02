import csv
import pathlib
from typing import Optional


def load_samples(csv_path: str | pathlib.Path) -> list[tuple[int, float]]:
    """Carrega amostras de L(n) de um CSV gerado pelo progressive_mean.py."""
    rows = list(csv.reader(open(csv_path)))
    return [(int(r[0]), float(r[1])) for r in rows[1:]]


def find_stabilization_point(
    samples: list[tuple[int, float]],
    window: int = 50,
    epsilon: float = 0.01
) -> Optional[tuple[int, float]]:
    """
    Encontra o primeiro ponto onde L(n) se estabilizou.

    window : número de amostras consecutivas que devem permanecer estáveis
    epsilon: variação máxima permitida em bps dentro da janela

    Retorna (n, L(n)) do primeiro ponto estável, ou None se não encontrado.
    """
    if len(samples) < window:
        return None

    values = [s[1] for s in samples]

    for i in range(window, len(values)):
        window_vals = values[i - window : i]
        if max(window_vals) - min(window_vals) < epsilon:
            return samples[i]

    return None


def find_transition_points(
    samples: list[tuple[int, float]],
    jump_threshold: float = 0.3
) -> list[tuple[int, float]]:
    """
    Detecta saltos abruptos em L(n), indicando mudança de contexto
    (fronteira entre arquivos no Silesia).

    jump_threshold: salto mínimo em bps para considerar transição
    Retorna lista de amostras onde ocorreram os saltos.
    """
    transitions = []
    values = [s[1] for s in samples]

    for i in range(1, len(values)):
        delta = abs(values[i] - values[i - 1])
        if delta > jump_threshold:
            transitions.append(samples[i])

    return transitions


def print_summary(csv_path: str | pathlib.Path):
    """Imprime um resumo da análise de L(n)."""
    samples = load_samples(csv_path)
    if not samples:
        print("Nenhuma amostra encontrada.")
        return

    ns, ls = zip(*samples)
    print(f"\n=== Análise de L(n): {pathlib.Path(csv_path).name} ===")
    print(f"  Amostras:      {len(samples)}")
    print(f"  Posição final: n={ns[-1]:,}")
    print(f"  L(inicial):    {ls[0]:.4f} bps")
    print(f"  L(final):      {ls[-1]:.4f} bps")
    print(f"  Mínimo de L:   {min(ls):.4f} bps em n={ns[ls.index(min(ls))]:,}")

    stab = find_stabilization_point(samples)
    if stab:
        print(f"  Estabilização: n={stab[0]:,}  L={stab[1]:.4f} bps")
    else:
        print("  Estabilização: não detectada (arquivo muito curto ou epsilon pequeno)")

    transitions = find_transition_points(samples)
    if transitions:
        print(f"  Transições detectadas: {len(transitions)}")
        for t in transitions[:5]:   # mostra apenas as 5 primeiras
            print(f"    n={t[0]:,}  L={t[1]:.4f} bps")
    else:
        print("  Transições detectadas: nenhuma")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_summary(sys.argv[1])
    else:
        # Analisa todos os CSVs de progressivo encontrados em results/
        for f in sorted(pathlib.Path("results").glob("progressive_*.csv")):
            print_summary(f)
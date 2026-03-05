#!/bin/bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════╗
# ║  verify_all.sh — Verificação de integridade + Experimentos  ║
# ╚══════════════════════════════════════════════════════════════╝

CORPUS_DIR="corpus/silesia"
ERRORS=0

# ── 0. Verificação de integridade (round-trip) ────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Etapa 0: Verificação de integridade (compress/decompress)"
echo "═══════════════════════════════════════════════════════════"

for FILE in "$CORPUS_DIR"/*; do
    NAME=$(basename "$FILE")
    pypy3 -m ppmc.cli compress "$FILE" "/tmp/${NAME}.ppmc" --order 5 --window 1000
    pypy3 -m ppmc.cli decompress "/tmp/${NAME}.ppmc" "/tmp/${NAME}.recovered"

    if cmp -s "$FILE" "/tmp/${NAME}.recovered"; then
        echo "✓ $NAME"
    else
        echo "✗ $NAME — FALHOU!"
        ERRORS=$((ERRORS + 1))
    fi

    rm -f "/tmp/${NAME}.ppmc" "/tmp/${NAME}.recovered"
done

echo ""
if [ "$ERRORS" -ne 0 ]; then
    echo "$ERRORS arquivo(s) falharam na verificação!"
    exit 1
fi
echo "Todos os arquivos passaram na verificação."
echo ""

# ── 3.1. Análise de Ordem e Performance ───────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Etapa 3.1: Benchmark PPM-C (Kmax 0..10) + comparação"
echo "═══════════════════════════════════════════════════════════"

pypy3 -m experiments.benchmark --kmax 0 11
pypy3 -m experiments.compare_external
pypy3 -m experiments.generate_tables

echo ""

# ── 3.2. Análise de Aprendizado — Dickens ─────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Etapa 3.2: Comprimento Médio Progressivo — Dickens"
echo "═══════════════════════════════════════════════════════════"

pypy3 -m experiments.progressive_mean dickens 5

echo ""

# ── 3.3. Análise de Aprendizado — Silesia (concatenado) ──────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Etapa 3.3: Comprimento Médio Progressivo — Silesia"
echo "═══════════════════════════════════════════════════════════"

pypy3 -m experiments.progressive_mean_silesia

echo ""

# ── Gráficos e Análise ────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Gerando gráficos e análise"
echo "═══════════════════════════════════════════════════════════"

python3 -m experiments.plot_results
pypy3 -m experiments.analysis

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Tudo concluído! Resultados em results/hash/"
echo "═══════════════════════════════════════════════════════════"

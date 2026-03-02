#!/bin/bash
set -e

INPUT="$1"
COMPRESSED="${INPUT}.ppmc"
RECOVERED="${INPUT}.recovered"

echo "=== Teste de integridade: $INPUT ==="

echo "[1/3] Comprimindo..."
python -m ppmc.cli compress "$INPUT" "$COMPRESSED" --order 10

echo "[2/3] Descomprimindo..."
python -m ppmc.cli decompress "$COMPRESSED" "$RECOVERED"

echo "[3/3] Verificando integridade..."
if cmp -s "$INPUT" "$RECOVERED"; then
    echo "✓ OK: '$INPUT' e '$RECOVERED' são idênticos"
else
    echo "✗ FALHOU: arquivos diferem!"
    exit 1
fi

# Limpeza
rm "$COMPRESSED" "$RECOVERED"
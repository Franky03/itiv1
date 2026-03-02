#!/bin/bash

ERRORS=0
for FILE in corpus/silesia/*; do
    NAME=$(basename "$FILE")
    python -m ppmc.cli compress "$FILE" "/tmp/${NAME}.ppmc" --order 5 --window 1000
    python -m ppmc.cli decompress "/tmp/${NAME}.ppmc" "/tmp/${NAME}.recovered"

    if cmp -s "$FILE" "/tmp/${NAME}.recovered"; then
        echo "✓ $NAME"
    else
        echo "✗ $NAME — FALHOU!"
        ERRORS=$((ERRORS + 1))
    fi

    rm -f "/tmp/${NAME}.ppmc" "/tmp/${NAME}.recovered"
done

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "Todos os arquivos passaram na verificação."
else
    echo "$ERRORS arquivo(s) falharam!"
    exit 1
fi
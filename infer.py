"""
Inferência com os modelos treinados pelo ablation_study.py.

Classifica um texto (stdin, arquivo .txt ou string direta) como
AI ou Humano para todas as 7 combinações de fontes.

Uso:
    python infer.py texto.txt
    python infer.py -t "O texto que você quer classificar"
    cat texto.txt | python infer.py
"""
import argparse
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ppmc.compressor import bpc_frozen

MODELS_DIR = pathlib.Path('models/ablation')

COMBINATIONS = [
    'A_all',
    'B_hc3',
    'C_hf',
    'D_csv',
    'BC_hc3+hf',
    'BD_hc3+csv',
    'CD_hf+csv',
]


def load_pair(name: str):
    h_path = MODELS_DIR / f'{name}_H.pkl'
    g_path = MODELS_DIR / f'{name}_G.pkl'
    if not h_path.exists() or not g_path.exists():
        return None, None
    with open(h_path, 'rb') as f:
        model_H = pickle.load(f)
    with open(g_path, 'rb') as f:
        model_G = pickle.load(f)
    return model_H, model_G


def classify(text: str) -> None:
    print(f"\nTexto ({len(text)} chars): {text[:80].strip()!r}{'...' if len(text) > 80 else ''}\n")
    print(f"{'Modelo':<15} {'BPC_H':>8} {'BPC_G':>8} {'Score':>8}  {'Predição':>10}")
    print("-" * 58)

    votes = {'AI': 0, 'Humano': 0, 'N/A': 0}

    for name in COMBINATIONS:
        model_H, model_G = load_pair(name)
        if model_H is None:
            print(f"{name:<15} {'—':>8} {'—':>8} {'—':>8}  {'(sem modelo)':>10}")
            votes['N/A'] += 1
            continue

        bpc_h = bpc_frozen(model_H, text)
        bpc_g = bpc_frozen(model_G, text)
        score = bpc_h - bpc_g
        pred  = 'AI' if score > 0 else 'Humano'
        votes[pred] += 1
        marker = ' <--' if pred == 'AI' else ''
        print(f"{name:<15} {bpc_h:>8.4f} {bpc_g:>8.4f} {score:>+8.4f}  {pred:>10}{marker}")

    total_valid = votes['AI'] + votes['Humano']
    print("-" * 58)
    if total_valid > 0:
        print(f"\nVotos: AI={votes['AI']}  Humano={votes['Humano']}  "
              f"(de {total_valid} modelos disponíveis)")
        winner = 'AI' if votes['AI'] > votes['Humano'] else 'Humano'
        confidence = max(votes['AI'], votes['Humano']) / total_valid * 100
        print(f"Resultado final: {winner}  ({confidence:.0f}% dos votos)\n")


def main():
    parser = argparse.ArgumentParser(description='Inferência PPM — classifica texto como AI ou Humano')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('file', nargs='?', type=pathlib.Path,
                       help='Arquivo .txt a classificar')
    group.add_argument('-t', '--text', type=str,
                       help='Texto direto como argumento')
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = args.file.read_text(encoding='utf-8', errors='replace')
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    text = text.strip()
    if not text:
        print("Erro: texto vazio.", file=sys.stderr)
        sys.exit(1)

    classify(text)


if __name__ == '__main__':
    main()

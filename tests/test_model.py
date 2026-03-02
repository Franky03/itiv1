from ppmc.model import PPMModel

def test_update_and_lookup():
    print("\n" + "="*70)
    print("TESTE 1: Construção da Árvore Trie e Janela de Contexto")
    print("="*70)
    
    m = PPMModel(max_order=2)
    print("Modelo inicializado com Ordem Máxima = 2 (lembra até 2 letras para trás).")
    
    # Processando 'a'
    print(f"\n[ Passo 1 ] Lendo 'a'. Contexto atual: {m.context}")
    m.update(ord('a'))
    print(f" -> O nó Raiz (contexto vazio) aprendeu que 'a' apareceu.")
    
    # Processando 'b'
    print(f"\n[ Passo 2 ] Lendo 'b'. Contexto atual: {[chr(c) for c in m.context]}")
    m.update(ord('b'))
    print(f" -> O nó Raiz aprendeu que 'b' apareceu.")
    print(f" -> O nó do contexto ['a'] aprendeu que 'b' veio depois dele.")
    
    # Processando 'a'
    print(f"\n[ Passo 3 ] Lendo 'a'. Contexto atual: {[chr(c) for c in m.context]}")
    m.update(ord('a'))
    print(f" -> O nó Raiz aprendeu que 'a' apareceu (de novo).")
    print(f" -> O nó do contexto ['b'] (ordem 1) aprendeu que 'a' veio depois dele.")
    print(f" -> O nó do contexto ['a', 'b'] (ordem 2) aprendeu que 'a' veio depois dele.")

    print(f"\n--- Verificação ---")
    root = m.get_context_node(0)
    print(f"Contagens na Raiz (Ordem 0): {{'a': {root.counts[ord('a')]}, 'b': {root.counts[ord('b')]}}}")
    assert root.counts[ord('a')] == 2
    assert root.counts[ord('b')] == 1

    # O contexto final de m.context é ['b', 'a'] porque o max_order é 2 (o primeiro 'a' foi descartado)
    # Então a ordem 1 vai pegar o sufixo de tamanho 1, que é o ['a'] atual.
    node_a = m.get_context_node(1)
    print(f"O nó de contexto ordem 1 (última letra foi 'a') existe? {'Sim' if node_a else 'Não'}")
    assert node_a is not None

def test_escape_count_method_c():
    print("\n" + "="*70)
    print("TESTE 2: Contagem de Escape (Método C)")
    print("="*70)
    
    m = PPMModel(max_order=1)
    sequencia = ['a', 'b', 'a', 'c']
    
    print(f"Inserindo a sequência: {sequencia}")
    for letra in sequencia:
        m.update(ord(letra))
        
    root = m.root
    print("\nEstado final do nó Raiz:")
    counts_char = {chr(k): v for k, v in root.counts.items()}
    print(f"Contagens: {counts_char}")
    
    print(f"\nRegra do Método C: O valor do Escape deve ser igual ao número de símbolos ÚNICOS.")
    print(f"Símbolos únicos vistos: {len(counts_char)} ('a', 'b', 'c')")
    print(f"Valor do escape_count gerado pelo modelo: {root.escape_count}")
    
    assert root.escape_count == 3
    print("SUCESSO: O escape_count está perfeitamente sincronizado com os símbolos distintos!")

def test_get_distribution_with_exclusion():
    print("\n" + "="*70)
    print("TESTE 3: Distribuição de Probabilidade e Conjunto de Exclusão")
    print("="*70)
    
    m = PPMModel(max_order=0)
    for ch in ['a', 'a', 'b', 'c']:
        m.update(ord(ch))

    root = m.root
    counts_char = {chr(k): v for k, v in root.counts.items()}
    print(f"Histórico do nó: {counts_char} | Escape (Método C): {root.escape_count}")
    print("-" * 70)

    # 1. Sem exclusão
    symbols, cum, total, sym_to_idx = m.get_distribution(root, set())
    sym_char = [chr(s) for s in symbols]
    print(f"CENÁRIO 1: Sem exclusão (tentando adivinhar de primeira)")
    print(f"Símbolos disponíveis: {sym_char}")
    print(f"Frequências Cumulativas (cum_freqs): {cum}")
    print(f"Total (Soma das contagens + Escape): {total}")
    print(f"  -> Note que o ESC ocupa a última fatia do intervalo: [{cum[-2]}, {cum[-1]})")
    
    assert symbols == sorted([ord('a'), ord('b'), ord('c')])
    assert total == 2 + 1 + 1 + 3

    print("-" * 70)
    
    # 2. Com exclusão
    print(f"CENÁRIO 2: Com exclusão de 'a' (caímos de um contexto maior e sabemos que não é 'a')")
    symbols2, cum2, total2, sym_to_idx2 = m.get_distribution(root, {ord('a')})
    sym_char2 = [chr(s) for s in symbols2]
    
    print(f"Símbolos disponíveis agora: {sym_char2}  <-- O 'a' sumiu!")
    print(f"Novas Frequências Cumulativas: {cum2}")
    print(f"Novo Total: {total2}  <-- Diminuiu, o que aumenta a probabilidade matemática de 'b' e 'c'!")
    print(f"  -> O Escape continua valendo {root.escape_count}, preservando a regra C.")
    
    assert ord('a') not in symbols2
    assert total2 < total
    print("\nSUCESSO: A exclusão redistribuiu as probabilidades corretamente!")
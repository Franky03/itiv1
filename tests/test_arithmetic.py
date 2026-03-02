from ppmc.utils import BitWriter, BitReader
from ppmc.arithmetic import ArithmeticEncoder, ArithmeticDecoder

def test_single_symbol_certain():
    print("\n" + "="*60)
    print("TESTE 1: Símbolo com 100% de probabilidade")
    print("="*60)
    
    w = BitWriter()
    enc = ArithmeticEncoder(w)
    
    print("Codificando 1 símbolo com intervalo [0, 1) e total 1...")
    enc.encode_symbol(0, 1, 1)
    enc.finish()
    compressed = w.flush()
    
    print(f"Bits gastos (úteis + finalização): {w.bits_written}")
    print(f"Bytes gerados: {list(compressed)}")
    
    r = BitReader(compressed)
    dec = ArithmeticDecoder(r)
    idx = dec.decode_symbol([0, 1], 1)
    print(f"Símbolo decodificado com sucesso: índice {idx}")
    assert idx == 0

def test_two_symbols_equal():
    print("\n" + "="*60)
    print("TESTE 2: Distribuição 50/50 (Cara ou Coroa)")
    print("="*60)
    
    w = BitWriter()
    enc = ArithmeticEncoder(w)
    
    print("Codificando símbolo 0. Fatias [0, 1) de um total de 2...")
    enc.encode_symbol(0, 1, 2)
    enc.finish()
    compressed = w.flush()

    print(f"Bits gastos (úteis + finalização): {w.bits_written} bits")
    print(f"Bytes gerados: {list(compressed)}")

    r = BitReader(compressed)
    dec = ArithmeticDecoder(r)
    idx = dec.decode_symbol([0, 1, 2], 2)
    print(f"Símbolo decodificado com sucesso: índice {idx}")
    assert idx == 0

def test_roundtrip_sequence():
    print("\n" + "="*60)
    print("TESTE 3: Sequência Completa (Distribuição Fixa A=4, B=2, C=1)")
    print("="*60)
    
    # Mapeamento para visualização: 0='A', 1='B', 2='C'
    alfabeto = ['A', 'B', 'C']
    symbols  = [0, 0, 1, 2, 0, 1]   # Sequência de índices: A, A, B, C, A, B
    cum_full = [0, 4, 6, 7]         # Frequências cumulativas
    total    = 7
    
    seq_str = [alfabeto[s] for s in symbols]
    print(f"Mensagem original a ser codificada: {seq_str}")
    print("-" * 60)

    # --- CODIFICAÇÃO ---
    w = BitWriter()
    enc = ArithmeticEncoder(w)
    
    for s in symbols:
        letra = alfabeto[s]
        low, high = cum_full[s], cum_full[s+1]
        print(f"Codificando '{letra}' -> Ocupa a fatia [{low}, {high}) do total de {total}")
        enc.encode_symbol(low, high, total)
        
    enc.finish()
    compressed = w.flush()
    
    print("-" * 60)
    print(f"COMPRESSÃO FINALIZADA!")
    print(f"Tamanho total: {w.bits_written} bits empacotados em {len(compressed)} bytes.")
    print(f"Stream de Bytes: {list(compressed)}")
    print("-" * 60)

    # --- DECODIFICAÇÃO ---
    r = BitReader(compressed)
    dec = ArithmeticDecoder(r)
    recovered = []
    
    print("Iniciando decodificação...")
    for i in range(len(symbols)):
        idx = dec.decode_symbol(cum_full, total)
        letra_recuperada = alfabeto[idx]
        print(f"Passo {i+1}: Leu do stream, intervalo apontou para índice {idx} ('{letra_recuperada}')")
        recovered.append(idx)

    print("-" * 60)
    rec_str = [alfabeto[s] for s in recovered]
    print(f"Mensagem recuperada: {rec_str}")
    
    assert recovered == symbols
    print("SUCESSO: A mensagem decodificada é idêntica à original!")
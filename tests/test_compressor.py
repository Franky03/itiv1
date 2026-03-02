from ppmc.monitor import ResetMonitor
import os
from ppmc.compressor   import compress
from ppmc.decompressor import decompress

def test_monitor_no_reset_stable():
    print("\n" + "="*70)
    print("TESTE 1: Compressão Estável (Sem degradação)")
    print("="*70)
    
    m = ResetMonitor(window_size=10, threshold_pct=20.0)
    print("Monitor configurado: Janela=10 bytes | Limite de piora=20.0%")
    print("Simulando a leitura de 30 bytes (3 janelas), custando sempre 8 bits cada...")
    
    bits = 0
    for i in range(30):
        bits += 8
        m.record(bits)
        
    resultado = m.should_reset()
    print(f"O monitor disparou o reset? {'Sim' if resultado else 'Não'}")
    
    assert not resultado
    print("-> SUCESSO: Como a taxa ficou estável em 8 bits/byte, o reset NÃO foi acionado.")

def test_monitor_triggers_on_degradation():
    print("\n" + "="*70)
    print("TESTE 2: Degradação de Compressão (Gatilho de Reset)")
    print("="*70)
    
    m = ResetMonitor(window_size=10, threshold_pct=10.0)
    print("Monitor configurado: Janela=10 bytes | Limite de piora=10.0%")
    
    bits = 0
    print("\n[ Janela 1 ] Simulando texto fácil: 10 bytes custando 4 bits cada (Compressão boa)")
    for _ in range(10):
        bits += 4
        m.record(bits)
        
    print("[ Janela 2 ] Simulando mudança de arquivo: 10 bytes custando 8 bits cada (Compressão ruim)")
    for _ in range(10):
        bits += 8
        m.record(bits)
        
    print("\nAnalisando a mudança: A média saltou de 4 para 8 bits/byte (Piora de 100%).")
    print("Como 100% de piora é bem maior que o limite de 10%, o monitor deve disparar!")
    
    resultado = m.should_reset()
    print(f"O monitor disparou o reset? {'Sim' if resultado else 'Não'}")
    
    assert resultado
    print("-> SUCESSO: O monitor detectou a degradação e acionou o pedido de reset corretamente.")

def test_monitor_clear_resets_state():
    print("\n" + "="*70)
    print("TESTE 3: Limpeza de Estado do Monitor (Clear)")
    print("="*70)
    
    m = ResetMonitor(window_size=5, threshold_pct=10.0)
    print("Preenchendo o monitor com dados de 2 janelas completas (10 bytes)...")
    
    for i in range(10):
        m.record(i * 8)
        
    print("Acionando o comando m.clear() para zerar a memória do monitor...")
    m.clear()
    
    resultado = m.should_reset()
    print(f"Logo após o clear, o monitor pede reset? {'Sim' if resultado else 'Não'}")
    
    assert not resultado
    print("-> SUCESSO: O estado foi apagado com sucesso. O monitor voltou a ficar zerado e não exige reset.")


def test_roundtrip_text():
    data = b"the quick brown fox jumps over the lazy dog" * 100
    assert decompress(compress(data)) == data


def test_roundtrip_binary():
    data = os.urandom(5000)
    assert decompress(compress(data, max_order=3)) == data


def test_roundtrip_empty():
    data = b""
    assert decompress(compress(data)) == data


def test_roundtrip_single_byte():
    data = b"x"
    assert decompress(compress(data)) == data


def test_roundtrip_all_bytes():
    data = bytes(range(256)) * 10
    assert decompress(compress(data, max_order=2)) == data


# ── Testes de compressão ───────────────────────────────────────────────────────

def test_compresses_better_than_raw():
    """Texto repetitivo deve comprimir para menos de 8 bits/símbolo."""
    data = b"abcabcabc" * 1000
    compressed = compress(data, max_order=4)
    bps = (len(compressed) * 8) / len(data)
    assert bps < 8.0, f"Esperado < 8 bps, obtido {bps:.2f}"


# ── Testes do mecanismo de reset ──────────────────────────────────────────────

def test_reset_does_not_corrupt():
    """
    Dados com mudança abrupta: compressível seguido de aleatório.
    O reset pode disparar, mas o resultado deve ser idêntico ao original.
    """
    compressible = b"aaaa" * 2000
    random_part  = os.urandom(2000)
    data = compressible + random_part

    compressed = compress(data, max_order=5, window_size=500, reset_threshold_pct=5.0)
    recovered  = decompress(compressed)
    assert recovered == data


def test_different_orders_all_correct():
    """Todos os valores de Kmax 0..5 devem produzir round-trip correto."""
    data = b"lorem ipsum dolor sit amet" * 50
    for kmax in range(6):
        recovered = decompress(compress(data, max_order=kmax))
        assert recovered == data, f"Falhou com Kmax={kmax}"


# ── Testes de formato ─────────────────────────────────────────────────────────

def test_header_magic():
    """O arquivo comprimido deve começar com b'PPMC'."""
    compressed = compress(b"hello world")
    assert compressed[:4] == b'PPMC'


def test_invalid_magic_raises():
    """Dados inválidos devem levantar ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Magic"):
        decompress(b"INVALID_HEADER_DATA" + b"\x00" * 20)
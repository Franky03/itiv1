from ppmc.utils import BitWriter, BitReader
from ppmc.arithmetic import ArithmeticEncoder, ArithmeticDecoder
from ppmc.model import PPMModel
from ppmc.encoder import encode_symbol
from ppmc.decoder import decode_symbol
import os

def _roundtrip(data: bytes, max_order: int, test_name: str = "") -> bytes:
    """Codifica e decodifica 'data'. Retorna os bytes recuperados."""
    # ── Codificação ────────────────────────────────────────────────────────
    writer = BitWriter()
    arith_enc = ArithmeticEncoder(writer)
    model_enc = PPMModel(max_order=max_order)

    for byte in data:
        encode_symbol(arith_enc, model_enc, byte)
        model_enc.update(byte)
    arith_enc.finish()
    compressed = writer.flush()

    # Exibindo as estatísticas
    tamanho_original = len(data)
    tamanho_comprimido = len(compressed)
    razao = (tamanho_comprimido / tamanho_original) * 100 if tamanho_original > 0 else 0
    
    print(f"[{test_name}] Tamanho Original: {tamanho_original} bytes")
    print(f"[{test_name}] Tamanho Comprimido: {tamanho_comprimido} bytes ({razao:.1f}% do original)")

    # ── Decodificação ──────────────────────────────────────────────────────
    reader = BitReader(compressed)
    arith_dec = ArithmeticDecoder(reader)
    model_dec = PPMModel(max_order=max_order)

    result = []
    for _ in range(len(data)):
        sym = decode_symbol(arith_dec, model_dec)
        result.append(sym)
        model_dec.update(sym)

    return bytes(result)


def test_roundtrip_tiny():
    print("\n" + "="*70)
    print("TESTE 1: Compressão Básica ('abracadabra')")
    print("="*70)
    data = b"abracadabra"
    resultado = _roundtrip(data, max_order=3, test_name="Texto Curto")
    assert resultado == data
    print("-> Sucesso! Mensagem decodificada perfeitamente.")


def test_roundtrip_all_bytes():
    print("\n" + "="*70)
    print("TESTE 2: Todos os 256 bytes possíveis")
    print("="*70)
    data = bytes(range(256))
    resultado = _roundtrip(data, max_order=2, test_name="Alfabeto Inteiro")
    assert resultado == data
    print("-> Sucesso! Nenhum byte se perdeu.")


def test_roundtrip_repetitive():
    print("\n" + "="*70)
    print("TESTE 3: Texto Altamente Repetitivo (Onde o PPMC Brilha!)")
    print("="*70)
    data = b"aaaaaaaaaa" * 100  # 1000 'a's seguidos
    resultado = _roundtrip(data, max_order=4, test_name="1000 letras 'a'")
    assert resultado == data
    print("-> Sucesso! Note como a compressão foi esmagadora (menos de 1% do tamanho).")


def test_roundtrip_order_zero():
    print("\n" + "="*70)
    print("TESTE 4: Ordem 0 (Sem usar contextos anteriores)")
    print("="*70)
    data = b"the quick brown fox jumps over the lazy dog"
    resultado = _roundtrip(data, max_order=0, test_name="Pangrama Ordem 0")
    assert resultado == data
    print("-> Sucesso! Funcionou perfeitamente como um compressor de probabilidade simples.")


def test_roundtrip_binary():
    print("\n" + "="*70)
    print("TESTE 5: Dados Binários Aleatórios (Entropia Máxima)")
    print("="*70)
    data = os.urandom(500)
    resultado = _roundtrip(data, max_order=3, test_name="Random Bin")
    assert resultado == data
    print("-> Sucesso! O arquivo até 'inchou' um pouco, o que é esperado para dados 100% aleatórios (pois não há padrões para o modelo aprender).")


def test_compression_improves_with_learning():
    print("\n" + "="*70)
    print("TESTE 6: A Prova do Aprendizado Dinâmico")
    print("="*70)
    data = b"ab" * 500   # 1000 bytes no total, padrão 'ab'

    writer = BitWriter()
    arith_enc = ArithmeticEncoder(writer)
    model_enc = PPMModel(max_order=2)

    snapshots = []
    print("Lendo 1000 bytes com o padrão repititivo 'abababab...'")
    for byte in data:
        bits_before = writer.bits_written
        encode_symbol(arith_enc, model_enc, byte)
        model_enc.update(byte)
        bits_after = writer.bits_written
        snapshots.append(bits_after - bits_before)

    n = len(data)
    first_half_avg  = sum(snapshots[:n//2])  / (n//2)
    second_half_avg = sum(snapshots[n//2:])  / (n//2)

    print(f"\n[ Resultado do Aprendizado ]")
    print(f"-> Custo médio na PRIMEIRA metade: {first_half_avg:.3f} bits por caractere")
    print(f"-> Custo médio na SEGUNDA metade:  {second_half_avg:.3f} bits por caractere")
    
    # Depois de aprender o padrão, a segunda metade deve custar menos
    assert second_half_avg < first_half_avg, (
        f"Esperado melhora: {first_half_avg:.3f} → {second_half_avg:.3f}"
    )
    print("\nSUCESSO! O modelo percebeu o padrão e passou a gastar quase ZERO bits por letra!")
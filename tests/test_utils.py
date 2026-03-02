from ppmc.utils import BitWriter, BitReader

def test_roundtrip_bits():
    bits = [1,0,1,1,0,0,0,1,  0,1,1,0,0,1,0,1]
    w = BitWriter()
    for b in bits:
        w.write_bit(b)
    data = w.flush()
    assert data == bytes([0b10110001, 0b01100101])

    r = BitReader(data)
    recovered = [r.read_bit() for _ in range(16)]
    assert recovered == bits

def test_bits_written_counter():
    w = BitWriter()
    for _ in range(10):
        w.write_bit(1)
    assert w.bits_written == 10
    w.flush()
    assert w.bits_written == 10  # flush não altera o contador

def test_partial_byte_flush():
    w = BitWriter()
    w.write_bit(1)
    w.write_bit(0)
    w.write_bit(1)   # 3 bits → deve virar 0b10100000 = 160
    data = w.flush()
    assert data == bytes([0b10100000])
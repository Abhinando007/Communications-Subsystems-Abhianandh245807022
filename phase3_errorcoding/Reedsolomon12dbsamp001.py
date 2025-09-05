import numpy as np
import json
import galois
import sk_dsp_comm.fec_conv as fec

# ----------------------
# QPSK Hard + Soft Demod
# ----------------------
def qpsk_demodulate(symbols):
    """
    QPSK hard decision demodulation -> bits
    Gray mapping assumed.
    """
    bits = []
    for s in symbols:
        # Real part: 0 if >0 else 1
        bits.append(0 if s.real > 0 else 1)
        # Imag part: 0 if >0 else 1
        bits.append(0 if s.imag > 0 else 1)
    return np.array(bits, dtype=np.uint8)

def qpsk_soft_demod(symbols):
    """
    QPSK soft decision demodulation -> LLR-like values.
    Here just use the raw real/imag values as soft metrics.
    """
    llrs = []
    for s in symbols:
        llrs.append(s.real)  # soft for first bit
        llrs.append(s.imag)  # soft for second bit
    return np.array(llrs)

# ----------------------
# Metrics
# ----------------------
def calculate_ber_fer(decoded_bits, ref_bits, frame_size=None):
    """
    Compare decoded bits against reference bits and return BER and FER.
    """
    n = min(len(decoded_bits), len(ref_bits))
    decoded_bits = decoded_bits[:n]
    ref_bits = ref_bits[:n]
    # Bit errors
    bit_errors = np.sum(decoded_bits != ref_bits)
    ber = bit_errors / n if n > 0 else float("nan")
    # Frame errors
    if frame_size is None:
        frame_size = n  # treat as one big frame
    num_frames = n // frame_size
    fer_count = 0
    for i in range(num_frames):
        start, end = i * frame_size, (i + 1) * frame_size
        if np.any(decoded_bits[start:end] != ref_bits[start:end]):
            fer_count += 1
    fer = fer_count / num_frames if num_frames > 0 else float("nan")
    return ber, fer, n, bit_errors

# ----------------------
# Evaluation
# ----------------------
def evaluate_debug(rx_samples, meta_content):
    print(f"Loaded {len(rx_samples)} complex samples")
    sps = meta_content["sps"]
    ground_truth_bits = np.array(meta_content["clean_bits"], dtype=np.uint8)
    # --- 1. Symbol extraction ---
    symbols = rx_samples[::sps]
    print(f"Extracted {len(symbols)} symbols")
    # --- 2. Hard demod ---
    raw_bits = qpsk_demodulate(symbols)
    # --- 3. Soft demod + Viterbi ---
    soft_bits = qpsk_soft_demod(symbols)
    # Conv. code: rate 1/2, constraint length 7, polynomials [133, 171] (octal)
    # Converted to binary strings for sk_dsp_comm usage
    G1 = '1011011'  # octal 133
    G2 = '1111001'  # octal 171
    cc = fec.FECConv((G1, G2), Depth=35)
    viterbi_out = cc.viterbi_decoder(soft_bits, 'soft', 35)
    # --- Fix: Convert soft float output to bits for packing ---
    viterbi_trim = viterbi_out[: (len(viterbi_out) // 4) * 4]
    viterbi_trim_bits = (viterbi_trim > 0).astype(np.uint8)
    nibbles = np.packbits(viterbi_trim_bits, bitorder="big").view(np.uint8)
    nibbles = np.unpackbits(nibbles, bitorder="big").reshape(-1, 4)
    nibbles = np.array(
        [int("".join(map(str, row)), 2) for row in nibbles], dtype=int
    )
    # --- 4. RS(15,11) over GF(2^4) ---
    RS = galois.ReedSolomon(15, 11, field=galois.GF(2**4))
    num_codewords = len(nibbles) // 15
    rs_decoded_bits = []
    for i in range(num_codewords):
        cw = nibbles[i * 15:(i + 1) * 15]
        try:
            msg = RS.decode(cw)
        except galois.ReedSolomonError:
            msg = cw[:11]  # fallback
        # convert symbols back to bits
        for val in msg:
            bits = [(val >> (3 - b)) & 1 for b in range(4)]
            rs_decoded_bits.extend(bits)
    rs_decoded_bits = np.array(rs_decoded_bits, dtype=np.uint8)
    # ----------------------
    # Results
    # ----------------------
    print("\n--- RESULTS ---")
    ber, fer, n, bit_errors = calculate_ber_fer(
        raw_bits, ground_truth_bits, frame_size=512
    )
    print(
        f"Raw hard demod → Compared: {n}, Errors: {bit_errors}, "
        f"BER: {ber:.3e}, FER: {fer:.2f}"
    )
    ber, fer, n, bit_errors = calculate_ber_fer(
        viterbi_out, ground_truth_bits, frame_size=512
    )
    print(
        f"After Viterbi → Compared: {n}, Errors: {bit_errors}, "
        f"BER: {ber:.3e}, FER: {fer:.2f}"
    )
    ber, fer, n, bit_errors = calculate_ber_fer(
        rs_decoded_bits, ground_truth_bits, frame_size=352
    )
    print(
        f"After Viterbi + RS(15,11) → Compared: {n}, Errors: {bit_errors}, "
        f"BER: {ber:.3e}, FER: {fer:.2f}"
    )
    # Save decoded bits
    np.save("reesolomonsample001_12db.npy", rs_decoded_bits)
    print("\n✅ Decoded bits saved to 'reesolomonsample001_12db.npy'")
    return rs_decoded_bits

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    RX_PATH = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase3_coding/reed_solomon/snr_12db/sample_001/rx.npy"
    META_PATH = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase3_coding/reed_solomon/snr_12db/sample_001/meta.json"
    rx_samples = np.load(RX_PATH)
    with open(META_PATH, "r") as f:
        meta_content = json.load(f)
    evaluate_debug(rx_samples, meta_content)

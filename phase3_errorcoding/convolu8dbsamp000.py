import numpy as np
import json
import matplotlib.pyplot as plt

# ----------------------
# QPSK Hard + Soft Demod
# ----------------------
def qpsk_demodulate(symbols):
    """QPSK hard decision demodulation -> bits (Gray mapping)."""
    bits = []
    for s in symbols:
        bits.append(0 if s.real > 0 else 1)   # bit from real
        bits.append(0 if s.imag > 0 else 1)   # bit from imag
    return np.array(bits, dtype=np.uint8)


def estimate_noise_var(symbols):
    """Blind noise variance estimation using distance to nearest QPSK constellation point."""
    const_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    diffs = symbols.reshape(-1, 1) - const_points.reshape(1, -1)
    dmin = np.min(np.abs(diffs) ** 2, axis=1)
    sigma2 = np.mean(dmin) / 2.0   # per real dimension
    return sigma2


def qpsk_soft_demod(symbols):
    """QPSK soft decision demodulation -> scaled LLR values."""
    sigma2 = estimate_noise_var(symbols)
    scale = 2.0 / sigma2
    llrs = []
    for s in symbols:
        llrs.append(scale * s.real)   # LLR for first bit
        llrs.append(scale * s.imag)   # LLR for second bit
    return np.array(llrs)


# ----------------------
# Viterbi Decoder (No commpy)
# ----------------------
def viterbi_decode(llrs, constraint_len=7, g=[0o133, 0o171], tb_depth=50):
    """
    Viterbi decoder for rate-1/2 convolutional code.
    - llrs: soft input values (length must be even)
    - g: generator polynomials (octal list)
    """
    K = constraint_len
    num_states = 2 ** (K - 1)
    num_bits = len(llrs) // 2

    # Precompute outputs for each state/input
    next_state = np.zeros((num_states, 2), dtype=int)
    outputs = np.zeros((num_states, 2, 2), dtype=int)

    for state in range(num_states):
        for bit in [0, 1]:
            shift_reg = ((state << 1) | bit) & (num_states - 1)
            out_bits = []
            for g_poly in g:
                val = bin(shift_reg & g_poly).count("1") % 2
                out_bits.append(val)
            next_state[state, bit] = shift_reg
            outputs[state, bit] = out_bits

    # Initialize path metrics
    PM = np.full(num_states, np.inf)
    PM[0] = 0.0
    paths = -np.ones((num_bits + 1, num_states), dtype=int)

    # Viterbi algorithm
    for i in range(num_bits):
        llr_pair = llrs[2 * i: 2 * i + 2]
        new_PM = np.full(num_states, np.inf)
        new_paths = -np.ones(num_states, dtype=int)

        for state in range(num_states):
            if PM[state] < np.inf:
                for bit in [0, 1]:
                    ns = next_state[state, bit]
                    expected = outputs[state, bit]
                    # Branch metric = -LLR * bit (MAP approximation)
                    bm = -np.sum((1 - 2 * np.array(expected)) * llr_pair)
                    metric = PM[state] + bm
                    if metric < new_PM[ns]:
                        new_PM[ns] = metric
                        new_paths[ns] = state * 2 + bit

        PM = new_PM
        paths[i + 1] = new_paths

    # Traceback
    state = np.argmin(PM)
    decoded = []
    for i in range(num_bits, 0, -1):
        prev = paths[i, state]
        if prev == -1:
            decoded.append(0)
            continue
        bit = prev % 2
        decoded.append(bit)
        state = prev // 2

    return np.array(decoded[::-1], dtype=np.uint8)


# ----------------------
# Metrics
# ----------------------
def calculate_ber_fer(decoded_bits, ref_bits, frame_size=None):
    n = min(len(decoded_bits), len(ref_bits))
    decoded_bits = decoded_bits[:n]
    ref_bits = ref_bits[:n]

    # Bit errors
    bit_errors = np.sum(decoded_bits != ref_bits)
    ber = bit_errors / n if n > 0 else float("nan")

    # Frame errors
    if frame_size is None:
        frame_size = n
    num_frames = n // frame_size
    fer_count = 0
    for i in range(num_frames):
        start, end = i * frame_size, (i + 1) * frame_size
        if np.any(decoded_bits[start:end] != ref_bits[start:end]):
            fer_count += 1
    fer = fer_count / num_frames if num_frames > 0 else float("nan")
    return ber, fer, n, bit_errors


# ----------------------
# Plots
# ----------------------
def plot_constellation(symbols, num_points=2000):
    plt.figure(figsize=(5, 5))
    use = symbols[:num_points]
    plt.scatter(use.real, use.imag, s=10, alpha=0.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.7)
    plt.title(f"Constellation (first {len(use)} symbols)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_llr_histogram(llrs):
    plt.figure(figsize=(6, 4))
    plt.hist(llrs, bins=100, alpha=0.75, edgecolor="black")
    plt.title("LLR Distribution")
    plt.xlabel("LLR value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()


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

    plot_constellation(symbols)

    # --- 2. Hard demod ---
    raw_bits = qpsk_demodulate(symbols)

    # --- 3. Soft demod + Convolutional decoding ---
    soft_bits = qpsk_soft_demod(symbols)

    plot_llr_histogram(soft_bits)

    viterbi_out = viterbi_decode(soft_bits, constraint_len=7, g=[0o133, 0o171], tb_depth=50)

    # ----------------------
    # Results
    # ----------------------
    print("\n--- RESULTS ---")
    ber, fer, n, bit_errors = calculate_ber_fer(raw_bits, ground_truth_bits, frame_size=512)
    print(f"Raw hard demod → Compared: {n}, Errors: {bit_errors}, BER: {ber:.3e}, FER: {fer:.2f}")

    ber, fer, n, bit_errors = calculate_ber_fer(viterbi_out, ground_truth_bits, frame_size=512)
    print(f"After Convolutional decoding → Compared: {n}, Errors: {bit_errors}, BER: {ber:.3e}, FER: {fer:.2f}")

    np.save("convolutionalsample000_8db.npy", viterbi_out)
    print("\n✅ Decoded bits saved to 'convolutionalsample000_8db.npy'")

    return viterbi_out


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    RX_PATH = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase3_coding/convolutional/snr_8db/sample_000/rx.npy"
    META_PATH = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase3_coding/convolutional/snr_8db/sample_000/meta.json"

    rx_samples = np.load(RX_PATH)
    with open(META_PATH, "r") as f:
        meta_content = json.load(f)

    evaluate_debug(rx_samples, meta_content)

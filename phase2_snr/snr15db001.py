import numpy as np
import json
import requests
import os
from urllib.parse import urlparse
from scipy import signal
from scipy.special import erfc
import matplotlib.pyplot as plt

# =======================
# PASTE YOUR LINKS HERE (for single file processing)
# =======================
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ADC3OyNiWy0Ue2pg9O4SaOg/phase2_snr/snr_15db/sample_001/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AGq3FDhBu87Kh39hd0GMF1o/phase2_snr/snr_15db/sample_001/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
OUT_BITS = "rx_bits00115db.bin"  # output file


# ------------------------
# File loading functions
# ------------------------
def download_file(url, local_name=None):
    """Downloads a file, now with validation to prevent saving HTML error pages."""
    if urlparse(url).scheme not in ("http", "https"):
        return url
    if local_name is None:
        local_name = os.path.basename(urlparse(url).path) or "tmpfile"
    print(f"[INFO] Downloading {url}")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        content_type = r.headers.get('content-type', '')
        if 'text/html' in content_type:
            print(f"[ERROR] Download failed: The URL returned an HTML page, not a data file.")
            print(f"[ERROR] Please check if the link '{url}' is a valid direct download link.")
            return None

        with open(local_name, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not download file: {e}")
        return None
    return local_name


def load_rx(rx_path):
    """Loads the received signal samples from a numpy file."""
    rx = np.load(rx_path, allow_pickle=True)
    if np.isrealobj(rx):
        rx = rx.astype(np.complex128)
    return rx


def load_meta(meta_path):
    """Loads the metadata from a JSON file."""
    with open(meta_path, "r") as f:
        return json.load(f)


# ------------------------
# DSP Helper functions
# ------------------------
def signal_power(x):
    """Calculates the average power of a complex signal."""
    return np.mean(np.abs(x) ** 2)


def noise_var_from_snr_db(sig_power, snr_db):
    """Calculates the expected noise variance from signal power and SNR."""
    snr_lin = 10 ** (snr_db / 10.0)
    return sig_power / snr_lin


def coarse_freq_estimate(rx):
    """Estimates carrier frequency offset."""
    return np.mean(np.angle(rx[1:] * np.conj(rx[:-1])))


def remove_freq_offset(rx, freq_off):
    """Corrects carrier frequency offset."""
    n = np.arange(len(rx))
    correction = np.exp(-1j * freq_off * n)
    return rx * correction


def rrc_filter(upsamp, span=6, beta=0.35):
    """Generates a Root-Raised Cosine (RRC) filter."""
    N = span * upsamp * 2 + 1
    t = np.arange(-span * upsamp, span * upsamp + 1) / float(upsamp)
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4 * beta)) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            taps[i] = (np.sin(np.pi * ti * (1 - beta)) +
                       4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                      (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps


def symbol_sync_energy(rx, sps, search_range=10):
    """Finds the best sampling offset."""
    best_off, max_e = 0, -1
    for off in range(search_range):
        syms = rx[off::sps]
        e = np.mean(np.abs(syms) ** 2)
        if e > max_e:
            best_off, max_e = off, e
    return best_off


# ------------------------
# Transmitter & Channel (for simulation)
# ------------------------
def modulate(bits, modulation="qpsk"):
    """Modulates a bit array into complex symbols."""
    modulation = modulation.lower()
    if modulation == "qpsk":
        mapping = {(0, 0): 1 + 1j, (0, 1): -1 + 1j,
                   (1, 1): -1 - 1j, (1, 0): 1 - 1j}
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([mapping[tuple(p)] for p in bit_pairs])
        return symbols / np.sqrt(2)
    else:
        raise ValueError("Unsupported modulation for simulation: " + modulation)


def add_noise(signal, snr_db):
    """Adds complex Additive White Gaussian Noise (AWGN) to a signal."""
    sig_power = signal_power(signal)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) * \
            (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise


# ------------------------
# Demodulation & BER
# ------------------------
def demod_symbols_to_bits(symbols, modulation="qpsk"):
    """Demodulates complex symbols into bits."""
    modulation = modulation.lower()
    if modulation == "bpsk":
        return (np.real(symbols) < 0).astype(np.uint8).reshape(-1, 1)
    elif modulation in ("qpsk", "4psk"):
        bits = np.zeros((len(symbols), 2), dtype=np.uint8)
        bits[:, 0] = (np.real(symbols) < 0)
        bits[:, 1] = (np.imag(symbols) < 0)
        return bits
    elif modulation == "16qam":
        avg_mag = np.mean(np.abs(symbols))
        level_spacing = avg_mag / 2.0

        def level_bits(x):
            if x < -2 * level_spacing: return [0, 0]
            if x < 0: return [0, 1]
            if x < 2 * level_spacing: return [1, 1]
            return [1, 0]

        bits = []
        for s in symbols:
            bits.append(level_bits(np.real(s)) + level_bits(np.imag(s)))
        return np.array(bits, dtype=np.uint8)
    else:
        raise ValueError("Unsupported modulation: " + modulation)


def qpsk_ber_theoretical(snr_db):
    """Calculates the theoretical BER for QPSK in an AWGN channel."""
    eb_no_lin = 10 ** (snr_db / 10.0) / 2
    return 0.5 * erfc(np.sqrt(eb_no_lin))


# ------------------------
# Main Receiver Logic
# ------------------------
def process_signal(rx, sps, modulation):
    """Runs the core receiver chain and returns demodulated bits."""
    freq_off = coarse_freq_estimate(rx)
    rx = remove_freq_offset(rx, freq_off)

    taps = rrc_filter(sps, span=6, beta=0.35)
    rx_filtered = signal.lfilter(taps, 1.0, rx)

    offset = symbol_sync_energy(rx_filtered, sps, search_range=sps)
    symbols = rx_filtered[offset::sps]

    mean_phase = np.angle(np.mean(symbols ** 4)) / 4.0
    symbols *= np.exp(-1j * mean_phase)

    symbols_normalized = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    bits_matrix = demod_symbols_to_bits(symbols_normalized, modulation)
    return bits_matrix.flatten()


def run_receiver_from_file(rx_path, meta_path):
    """Processes a single signal file and plots its constellation."""
    rx = load_rx(rx_path)
    if rx is None:
        return
    meta = load_meta(meta_path)
    if meta is None:
        return

    sps = int(meta.get("sps", meta.get("samples_per_symbol", 8)))
    modulation = meta.get("modulation", "qpsk")
    snr_db = float(meta.get("snr_db", meta.get("SNR", 0)))

    print(f"[INFO] Samples/symbol = {sps}, Modulation = {modulation}, Meta SNR = {snr_db} dB")

    P_sig = signal_power(rx)
    P_noise_exp = noise_var_from_snr_db(P_sig, snr_db)
    meas_snr_db = 10 * np.log10(P_sig / P_noise_exp)

    print(f"[CAL] Measured signal power: {P_sig:.4f}")
    print(f"[CAL] Expected noise variance: {P_noise_exp:.4e}")
    print(f"[CAL] Effective measured SNR ≈ {meas_snr_db:.2f} dB")

    freq_off = coarse_freq_estimate(rx)
    print(f"[SYNC] Estimated frequency offset: {freq_off:.4f} rad/sample")
    rx_freq_corrected = remove_freq_offset(rx, freq_off)

    taps = rrc_filter(sps, span=6, beta=0.35)
    rx_filtered = signal.lfilter(taps, 1.0, rx_freq_corrected)

    offset = symbol_sync_energy(rx_filtered, sps, search_range=sps)
    print(f"[SYNC] Found best symbol offset: {offset}")
    symbols = rx_filtered[offset::sps]

    mean_phase_rad = np.angle(np.mean(symbols ** 4)) / 4.0
    mean_phase_deg = np.degrees(mean_phase_rad)
    print(f"[SYNC] Estimated phase offset: {mean_phase_deg:.2f} degrees")
    symbols_phase_corrected = symbols * np.exp(-1j * mean_phase_rad)

    symbols_raw = symbols_phase_corrected.copy()
    symbols_normalized = symbols_raw / np.sqrt(np.mean(np.abs(symbols_raw) ** 2))

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Constellation for {modulation.upper()} at {snr_db} dB SNR", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.scatter(np.real(symbols_raw), np.imag(symbols_raw), s=5, alpha=0.5)
    plt.title("Synced Constellation")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.scatter(np.real(symbols_normalized), np.imag(symbols_normalized),
                s=5, alpha=0.5, c="red")
    plt.title("Normalized Constellation (Es=1)")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    rx_bits = demod_symbols_to_bits(symbols_normalized, modulation).flatten()
    print(f"[INFO] Wrote {len(rx_bits)} bits to {OUT_BITS}")
    rx_bits.tofile(OUT_BITS)


# ------------------------
# BER Simulation
# ------------------------
def run_ber_simulation():
    """Runs a BER simulation for QPSK and plots the results."""
    sps = 8
    modulation = 'qpsk'
    num_symbols = 20000
    bits_per_symbol = 2
    num_bits = num_symbols * bits_per_symbol
    snr_range_db = np.arange(0, 11, 1)

    print(f"--- Starting BER Simulation for {modulation.upper()} ---")

    tx_bits = np.random.randint(0, 2, num_bits)
    tx_symbols = modulate(tx_bits, modulation)

    tx_symbols_upsampled = np.zeros(len(tx_symbols) * sps, dtype=np.complex128)
    tx_symbols_upsampled[::sps] = tx_symbols

    rrc_taps = rrc_filter(sps, span=6, beta=0.35)
    tx_signal = signal.lfilter(rrc_taps, 1.0, tx_symbols_upsampled)

    ber_simulated = []
    for snr_db in snr_range_db:
        rx_signal = add_noise(tx_signal, snr_db)
        rx_bits = process_signal(rx_signal, sps, modulation)

        rx_symbols_count = len(rx_bits) // bits_per_symbol
        tx_bits_aligned = tx_bits[:rx_symbols_count * bits_per_symbol]

        num_errors = np.sum(tx_bits_aligned != rx_bits)
        ber = num_errors / len(tx_bits_aligned)
        ber_simulated.append(ber)
        print(f"SNR: {snr_db:2d} dB -> BER: {ber:.1e} ({num_errors} errors)")

    ber_theoretical = qpsk_ber_theoretical(snr_range_db)

    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_range_db, ber_simulated, 'bo-', label='Simulated BER')
    plt.semilogy(snr_range_db, ber_theoretical, 'r--', label='Theoretical QPSK BER')
    plt.grid(True, which='both')
    plt.xlabel('SNR (Es/N0) [dB]')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('QPSK Bit Error Rate Performance')
    plt.legend()
    plt.ylim([1e-5, 1e0])
    plt.show()


# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    # Run BER Simulation
    run_ber_simulation()

    # Run the original single-file receiver
    print("\n--- Running Single File Receiver ---")
    rx_path = download_file(RX_URL, "rx.npy")
    meta_path = download_file(META_URL, "meta.json")
    if rx_path and meta_path:
        run_receiver_from_file(rx_path, meta_path)

import numpy as np
import requests
import io

# URLs (must be ?dl=1 direct download)
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AOkJlQr_yHLfeAe0iqOZYGM/phase1_timing/snr_10db/sample_002/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ADF7oyGMTh3J0b1VMUr-oXI/phase1_timing/snr_10db/sample_002/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"

# --- Load RX file ---
try:
    r_waveform = requests.get(RX_URL)
    r_waveform.raise_for_status()
    rx = np.load(io.BytesIO(r_waveform.content), allow_pickle=True)
    print(f"Waveform loaded: shape={rx.shape}, dtype={rx.dtype}")
except Exception as e:
    print(f"Error loading waveform: {e}")
    exit()

# --- Load metadata ---
try:
    r_meta = requests.get(META_URL)
    r_meta.raise_for_status()
    meta = r_meta.json()
    print("Metadata loaded.")
except Exception as e:
    print(f"Error loading or parsing metadata: {e}")
    exit()

# --- Parameters ---
sps = meta["sps"]
clean_bits = np.array(meta["clean_bits"])

# --- Step 1: Root Raised Cosine (RRC) Matched Filter ---
def rrc_filter(beta, span, sps):
    """Generate Root-Raised Cosine filter taps."""
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0.0:
            h[i] = 1.0 - beta + (4*beta/np.pi)
        elif abs(t[i]) == 1/(4*beta):
            h[i] = (beta/np.sqrt(2)) * (
                ((1+2/np.pi)*np.sin(np.pi/(4*beta))) +
                ((1-2/np.pi)*np.cos(np.pi/(4*beta)))
            )
        else:
            h[i] = (np.sin(np.pi*t[i]*(1-beta)) +
                    4*beta*t[i]*np.cos(np.pi*t[i]*(1+beta))) / \
                   (np.pi*t[i]*(1-(4*beta*t[i])**2))
    return h / np.sum(h)   # normalize taps to unit gain

rrc = rrc_filter(beta=0.35, span=12, sps=sps)
filtered = np.convolve(rx, rrc, mode="same")

# --- Step 2: Gardner Timing Recovery ---
def gardner_timing_recovery(signal, sps):
    """Simple Gardner timing recovery."""
    mu = 0.0  # fractional offset
    out = []
    i = 0
    while i < len(signal) - sps:
        idx = int(i + mu)
        frac = mu - int(mu)
        # linear interpolation
        sym = (1-frac)*signal[idx] + frac*signal[idx+1]
        out.append(sym)

        # mid-symbol sample
        mid = (1-frac)*signal[idx+sps//2] + frac*signal[idx+sps//2+1]

        # Gardner error
        err = np.real(np.conj(sym) * (mid - sym))

        # update mu
        mu += sps + 0.01 * err
        i += int(mu)
        mu -= int(mu)
    return np.array(out)

sampled = gardner_timing_recovery(filtered, sps)

# --- Step 3: Carrier Phase Recovery (Costas Loop) ---
corrected = np.zeros_like(sampled, dtype=complex)
phase = 0.0
freq = 0.0
alpha, beta = 0.05, 0.001  # loop gains

for i, x in enumerate(sampled):
    rot = x * np.exp(-1j*phase)
    corrected[i] = rot
    # BPSK error detector
    err = np.sign(np.real(rot)) * np.imag(rot)
    # Loop update
    freq += beta * err
    phase += freq + alpha * err

# --- Step 4: Normalize amplitude ---
corrected /= np.abs(corrected).mean()

# --- Step 5: Hard decision ---
decoded_bits = (np.real(corrected) > 0).astype(int)

# --- Step 6: Align before BER ---
def align_and_ber(decoded, ref):
    corr = np.correlate(2*decoded-1, 2*ref-1, mode="full")
    shift = np.argmax(corr) - len(ref) + 1
    aligned = decoded[max(0,shift):shift+len(ref)]
    aligned = aligned[:len(ref)]
    bit_errors = np.sum(aligned != ref[:len(aligned)])
    return bit_errors / len(aligned), bit_errors, shift

ber, bit_errors, shift = align_and_ber(decoded_bits, clean_bits)
print(f"Improved BER: {ber:.6f} ({bit_errors} errors, shift={shift})")

# --- Step 7: Save ---
np.save("improveddecodedbitssamp002_10db.npy", decoded_bits)
print("Decoded bits saved as 'improveddecodedbitssamp002_10db.npy'")


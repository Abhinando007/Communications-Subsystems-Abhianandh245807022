import numpy as np
import requests
import io
import json
import os
from scipy.signal import upfirdn, fir_filter_design, lfilter

# URLs (must be ?dl=1 direct download)
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AEByBxKkHvCdLrfuHATp2PU/phase1_timing/snr_5db/sample_000/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/AFAWLloJZnf8kZtaQk_CsFk/phase1_timing/snr_5db/sample_000/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"

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
timing_offset = meta["timing_offset"]
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
    return h / np.sqrt(np.sum(h**2))

rrc = rrc_filter(beta=0.35, span=10, sps=sps)
filtered = np.convolve(rx, rrc, mode="same")

# --- Step 2: Symbol timing recovery ---
# Use metadata offset (coarse) + refine
sampled = filtered[timing_offset::sps]

# --- Step 3: Carrier Phase Recovery (Decision-Directed Loop) ---
corrected = np.zeros_like(sampled, dtype=complex)
phase = 0
alpha = 0.05  # loop gain
for i, sym in enumerate(sampled):
    # Apply phase rotation
    sym_rot = sym * np.exp(-1j * phase)
    # Hard decision for BPSK (+1 or -1)
    decision = 1 if np.real(sym_rot) > 0 else -1
    corrected[i] = sym_rot
    # Update phase estimate (error = decision * conjugate(symbol))
    error = np.angle(decision * np.conj(sym_rot))
    phase += alpha * error

# --- Step 4: Normalize amplitude ---
corrected /= np.abs(corrected).mean()

# --- Step 5: Hard decision ---
decoded_bits = (np.real(corrected) > 0).astype(int)
decoded_bits = decoded_bits[:len(clean_bits)]

# --- Step 6: BER ---
bit_errors = np.sum(decoded_bits != clean_bits)
ber = bit_errors / len(clean_bits)
print(f"Improved BER: {ber:.6f} ({bit_errors} errors out of {len(clean_bits)})")

# --- Step 7: Save ---
np.save("semiimproved_bitssamp0005db.npy", decoded_bits)
print("Decoded bits saved as 'semiimproved_bitssamp0005db.npy'")



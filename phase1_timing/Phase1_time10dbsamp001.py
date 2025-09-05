import numpy as np
import requests
import io
import json

# Corrected URLs with dl=1
RX_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ANiYPM0jSagz0_l41S-bGdc/phase1_timing/snr_10db/sample_001/rx.npy?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"
META_URL = "https://www.dropbox.com/scl/fo/0fgz5lo991qc2z82kuqdb/ADM8jpBslUN69uYCDWeCB3A/phase1_timing/snr_10db/sample_001/meta.json?rlkey=yoal3tzf0eyy5i7qtvutyr36q&dl=1"

# Load waveform data from URL
try:
    r_waveform = requests.get(RX_URL)
    r_waveform.raise_for_status()
    rx = np.load(io.BytesIO(r_waveform.content), allow_pickle=True)
    print(f"Waveform loaded: shape={rx.shape}, dtype={rx.dtype}")
except Exception as e:
    print(f"Error loading waveform: {e}")
    exit()

# Load metadata JSON from URL
try:
    r_meta = requests.get(META_URL)
    r_meta.raise_for_status()
    meta = r_meta.json()
    print("Metadata loaded.")
except Exception as e:
    print(f"Error loading or parsing metadata: {e}")
    print("Response content received:")
    print(r_meta.text)
    exit()

# --- The rest of your script remains the same ---

# Extract parameters from meta
sps = meta["sps"]
timing_offset = meta["timing_offset"]
clean_bits = np.array(meta["clean_bits"])

# Step 1: Matched filter
matched_filter = np.ones(sps)
filtered = np.convolve(rx, matched_filter, mode='same')

# Step 2: Symbol timing recovery
sampled = filtered[timing_offset::sps]

# Step 3: Hard decision BPSK demodulation
decoded_bits = (np.real(sampled) > 0).astype(int)

# Step 4: Trim bits
decoded_bits = decoded_bits[:len(clean_bits)]

# Step 5: Calculate BER
bit_errors = np.sum(decoded_bits != clean_bits)
ber = bit_errors / len(clean_bits)
print(f"BER: {ber:.6f} ({bit_errors} errors out of {len(clean_bits)})")

# Step 6: Save decoded bits
np.save('decoded_bitssample00110dbT.npy', decoded_bits)
print("Decoded bits saved as 'decoded_bitssample00110dbT.npy'.")

# Optional: Show first 20 decoded bits
print("First 20 decoded bits:", decoded_bits[:20])




# (Code from Step 1 and 2 is the same)
sps = meta["sps"]
timing_offset = meta["timing_offset"]
clean_bits = np.array(meta["clean_bits"])
matched_filter = np.ones(sps)
filtered = np.convolve(rx, matched_filter, mode='same')
sampled = filtered[timing_offset::sps]

# --- NEW STEP: Carrier Phase Correction ---
# A simple way to estimate the phase is to calculate the average angle of the symbols.
# We multiply by 4 to remove the BPSK modulation ambiguity, average, then divide by 4.
phase_offset_estimate = np.angle(np.mean(sampled**4)) / 4
# "De-rotate" the symbols by applying the opposite phase rotation
corrected_sampled = sampled * np.exp(-1j * phase_offset_estimate)
# --- END OF NEW STEP ---

# Step 3: Hard decision BPSK demodulation (ON THE CORRECTED SYMBOLS)
decoded_bits = (np.real(corrected_sampled) > 0).astype(int)

# (The rest of the code is the same)
decoded_bits = decoded_bits[:len(clean_bits)]
bit_errors = np.sum(decoded_bits != clean_bits)
ber = bit_errors / len(clean_bits)
print(f"BER: {ber:.6f} ({bit_errors} errors out of {len(clean_bits)})")



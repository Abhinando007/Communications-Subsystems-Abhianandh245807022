# Communications-Subsystems-Abhinandh245807022
communications subsystems under Cubesat recu.it



a)Phase1_Timing:

XXXX(THE CODE AND OUTPUT SCREENSHOTS ARE ATTACHED TO A SEPARATE FILE)XXXX

 This script first fetches the received signal (rx.npy) and its corresponding metadata (meta.json) from Dropbox links.
 rx.npy conatins the raw recieved waveform and meta.json provides the simulation parameters.A simple rectangular matched     filter  is applied using convolution.It improves the signal to noise ratio.Using the timing_offset from metadata, the     script samples the filtered waveform at the correct symbol instants.A rough phase offset of the received symbols is    estimated using a 4th-power method.The estimated phase is then corrected by rotating the samples.After this           demodulation is performed.The decoded bit sequence is trimmed to match the length of the reference.Bit Error Rate (BER) is computed as the ratio of incorrect bits to total bits.The BER is then printed along with the number of  errors.Decoded bits are saved to a .npy file for further use.

Resources used: NumPy (Used for numerical computations, array manipulation, convolution (matched filtering), and bit error calculations.),Requests (Used to fetch the .npy waveform file and .json metadata file directly from Dropbox cloud links.),
io (Used with BytesIO to handle in-memory binary streams when loading .npy files received over HTTP.),json (Used to parse metadata (meta.json) into Python dictionaries for extracting simulation parameters like sps, timing_offset, and clean_bits.)

FAILURES AND FIXES:
1) Error loading waveform: This file contains pickled (object) data. 

it comes from NumPy when you try to load a .npy file that contains pickled Python objects instead of plain arrays.
If you trust the source of the .npy file, you can load it like this:
import numpy as np

data = np.load("your_file.npy", allow_pickle=True)
print(data)

it is because our .npy file on Dropbox is stored with pickled objects, not a plain NumPy array.

2)Failed to interpret file <_io.BytesIO object ...> as a pickle

This usually happens when the downloaded file from Dropbox is not actually the .npy file content but an HTML page.

Make sure we’re downloading the raw binary file from Dropbox.
In Dropbox URLs:

?dl=0 → shows the web preview page (HTML) ❌

?dl=1 → forces a direct download (raw file) ✅

3) The BER value was not below 0.1

The pipeline was very basic (rectangular matched filter + fixed timing offset + hard-decision demod). That’s why we were stuck at BER ≈ 0.245.
To push BER down to ≤ 0.01, we need to strengthen the receiver with a few standard comms techniques.
fixes I added:
3.1) used a proper root-raised cosine (RRC) matched filter instead of np.ones(sps).
3.2) Refined symbol timing with a timing recovery algorithm (instead of trusting timing_offset blindly).
3.3) Better carrier phase estimation (BPSK: angle-based, use decision-directed loop instead of a one-shot estimate).
3.4) Normalize amplitude before hard decisions.

RRC filter removes out-of-band noise → reduces ISI.

Decision-directed carrier recovery keeps phase lock across the whole stream → avoids drift.

Amplitude normalization ensures threshold at 0 is correct.

4) after timing recovery I got 1023 decoded bits, but the ground-truth clean_bits has 1024 bits.
   Quick fix: truncate both arrays to the same minimum length before comparing.

b)PHASE2_SNR:

Data Handling (Input Signals and Metadata):

The script begins by downloading the received waveform (rx.npy) and metadata (meta.json) from Dropbox links.
The metadata provides key parameters like samples per symbol (sps), modulation type, and SNR values.
The raw signal is loaded into NumPy arrays for further processing.

Receiver Front-End (Synchronization & Filtering):

Frequency Offset Estimation & Correction
A coarse frequency offset is estimated by examining phase differences between consecutive samples.
The estimated offset is removed by rotating the received signal in the complex plane.

Matched Filtering with Root-Raised Cosine (RRC) Filter:

An RRC filter is designed and applied to maximize SNR and reduce inter-symbol interference (ISI).

Symbol Timing Recovery:

The best sampling offset is found by maximizing average symbol energy.
Correctly aligned symbols are extracted at one sample per symbol.

Carrier Phase Synchronization:

The script estimates residual carrier phase offset using a 4th-power method (effective for QPSK).
Corrective rotation is applied to align constellation points properly.

Normalization:

The corrected symbols are normalized to unit average energy.
This ensures consistency before demodulation.

Demodulation:

Symbols are mapped back into bits depending on the modulation type (BPSK, QPSK, or 16-QAM).

For QPSK:

Sign of real part → first bit

Sign of imaginary part → second bit.

Visualization:

Two main plots are generated:

Constellation diagrams (synced vs normalized) for real signals.
BER vs SNR curve comparing simulated and theoretical results.

Output:

The receiver saves the decoded bitstream into a binary file (rx_bits0000db.bin).
Console logs provide detailed info: power, estimated offsets, BER, and symbol counts.

RESOURECES USED:

Python Libraries

1) NumPy (import numpy as np)
Used for handling complex arrays, numerical computations, modulation/demodulation mapping, and BER calculations.

2) Requests (import requests)
Used to download the received signal (rx.npy) and metadata (meta.json) from Dropbox links.

3) os / urllib.parse
Used to manage file saving and validate URLs for downloads.

4) json (import json)
Used to parse metadata files (meta.json) containing system parameters (samples per symbol, modulation type, SNR).

5) SciPy (signal, special)

scipy.signal: Used for applying digital filters (Root-Raised Cosine filtering, convolution).
scipy.special.erfc: Used to calculate theoretical BER for QPSK.

6) Matplotlib (import matplotlib.pyplot as plt)
Used for plotting constellation diagrams and BER vs. SNR curves.

FAILURES AND FIXES:

1) Error loading waveform: This file contains pickled (object) data. 

it comes from NumPy when you try to load a .npy file that contains pickled Python objects instead of plain arrays.
If you trust the source of the .npy file, you can load it like this:
import numpy as np

data = np.load("your_file.npy", allow_pickle=True)
print(data)

it is because our .npy file on Dropbox is stored with pickled objects, not a plain NumPy array.

2)Failed to interpret file <_io.BytesIO object ...> as a pickle

This usually happens when the downloaded file from Dropbox is not actually the .npy file content but an HTML page.

Make sure we’re downloading the raw binary file from Dropbox.
In Dropbox URLs:

?dl=0 → shows the web preview page (HTML) ❌

?dl=1 → forces a direct download (raw file) ✅

COMMANDS/STEPS TO REPRODUCE:

Install Python (if not already installed)
Make sure you have Python 3.8+ installed. Check with:
python --version

Set Up a Virtual Environment (recommended)
python -m venv comms_env
source comms_env/bin/activate   # On Linux / Mac
comms_env\Scripts\activate      # On Windows

pip install numpy requests scipy matplotlib

save the script and run the script.

C) PHASE3_ERROR DECODING:

1. Input Data

Loads received complex baseband samples (rx.npy).
Loads metadata (meta.json) containing:
sps (samples per symbol),
clean_bits (ground-truth transmitted bits).

2. Symbol Extraction

Picks every sps-th sample from rx_samples → extracts QPSK symbols.
symbols = rx_samples[::sps]

3. QPSK Demodulation

Hard Demod (qpsk_demodulate): Maps each complex symbol → 2 bits using Gray mapping (real part for bit1, imag part for bit2).
Soft Demod (qpsk_soft_demod): Keeps raw real/imag values as soft log-likelihood ratios (LLRs).

4. Viterbi Decoding (Convolutional Code)

Uses a rate 1/2 convolutional code with constraint length 7 and generator polynomials [133, 171] (octal).
Runs Viterbi decoder in soft-decision mode (uses LLRs from step 3).
Produces a decoded bitstream.

5. Reed-Solomon (RS) Decoding

Interprets groups of 4 bits as GF(2⁴) symbols (nibbles).
Uses RS(15,11) code over GF(2⁴):
Each codeword has 15 symbols (can correct up to 2 symbol errors).
Decoder tries to correct errors; if decoding fails, it falls back to first 11 symbols.
Converts decoded RS symbols back into bits.

6. Error Metrics

Compares against clean_bits from metadata.

Computes:

BER (Bit Error Rate),
FER (Frame Error Rate),
At three stages:
Hard demod output,
After Viterbi,
After Viterbi + RS.

7. Save Output

Saves final RS-decoded bits to a .npy file (reesolomonsample000_8db.npy).

**convolution**

Input Data

Loads:

rx.npy → Received complex baseband samples,
meta.json → Metadata with:
sps (samples per symbol),
clean_bits (ground-truth transmitted bits).

2. Symbol Extraction

Downsamples the raw input by sps to get one QPSK symbol per symbol period:
symbols = rx_samples[::sps]

3. QPSK Demodulation

Hard Demod (qpsk_demodulate)

Maps each symbol → 2 bits (Gray coding).

Real > 0 → 0 else 1,
Imag > 0 → 0 else 1.

Soft Demod (qpsk_soft_demod)
First estimates noise variance (estimate_noise_var),
Uses it to scale symbol values into approximate Log-Likelihood Ratios (LLRs).
Produces soft metrics for more reliable decoding.

4. Convolutional Decoding (Custom Viterbi)

Implements its own Viterbi decoder (no external library).

Parameters:

Constraint length = 7,
Generator polynomials = [133, 171] (octal),
Rate = 1/2 code,
Traceback depth ≈ 50.
Works in soft-decision mode (uses LLRs from step 3).
Produces decoded bitstream.

5. Error Metrics

Compares decoded bits to clean_bits.

Computes:

BER (Bit Error Rate),
FER (Frame Error Rate, per frame of given size).
Evaluates at 2 stages:
After hard QPSK demod,
After convolutional decoding.

6. Visualization

Constellation plot of received symbols.
Histogram of LLRs from soft demod.
Helps debug modulation quality and noise.

7. Output

Prints BER/FER results to console.
Saves final Viterbi-decoded bitstream to .npy file:

convolutionalsample000_12db.npy

**FAILURES AND FIXES:**

 The initial code contained commpy module but I was'nt able to download the module due to the errors.So I replaced commpy with sk_dsp_comm.

I got errors regarding file path and it can be avoided by ensuring the file is saved to the correct location.

**commands/steps to reproduce**:

Install Dependencies

Make sure you have Python 3.8+ installed.
Install required packages:

pip install numpy galois sk-dsp-comm

3. Prepare Input Data

You need:
rx.npy → Received complex baseband samples.
meta.json → Metadata file containing:
sps (samples per symbol),
clean_bits (ground-truth bitstream).
Place them in the appropriate folder (update paths in the script if needed).

4. Run the Decoder

Run the script:

python e1e91f7e-3a7b-44ff-9779-0ba6290aa617.py

5. Outputs

Console will print BER/FER results at 3 stages:
Raw QPSK demodulation,
After Viterbi decoding,
After Reed-Solomon decoding.
Final RS-decoded bitstream is saved as:

reesolomonsample000_8db.npy

**convolution**

Install Dependencies

Make sure you have Python 3.8+ installed.
Install the required packages:

pip install numpy matplotlib

3. Prepare Input Data

You need:

rx.npy → Received complex baseband samples,
meta.json → Metadata containing:
sps (samples per symbol),
clean_bits (ground-truth bitstream).
Update the paths inside the script if your files are in a different location.

4. Run the Decoder

Run the script:
python 73996b52-64c9-4b07-b23d-1bef22ec0fa3.py

5. Outputs

Console prints BER/FER results at:
Hard QPSK demodulation,
After convolutional decoding.

Debug plots:

Constellation diagram of received symbols,
LLR histogram from soft demodulation.
Final decoded bitstream is saved as:

convolutionalsample000_12db.npy

D)**PHASE4_DOPPLER**

1. Input Data

Loads:

rx.npy → received complex baseband samples,
meta.json → metadata (sample_rate, sps, clean_bits, filter params).

2. Matched Filtering

Designs an RRC (Root Raised Cosine) filter (rrc_filter),
Applies it to the raw samples with convolution → improves SNR and removes out-of-band noise.

3. Symbol Synchronization

Coarse timing offset search: picks the best sampling offset by maximizing average symbol magnitude.
Gardner Timing Recovery: interpolates samples to correct symbol timing drift.

4. Doppler / CFO Estimation & Correction

Splits signal into blocks, estimates frequency offset in each block (estimate_cfo_per_block),
Interpolates correction and applies it (apply_cfo_correction),
Produces frequency-stabilized symbols.

5. Carrier Recovery

Uses a QPSK Costas Loop (costas_qpsk) to track residual carrier phase/frequency errors.

6. Demodulation

Soft Demod (LLRs): Computes scaled log-likelihood ratios (qpsk_llrs).
Hard Decision Bits: Directly thresholds real/imag components (qpsk_hard_bits).

7. BER Evaluation

If clean_bits available in metadata → compares recovered bits vs. ground truth → prints BER.

8. Outputs

Saves results into outputs/ folder:
Decoded bits: doppler2000Hz10db.npy (NumPy array),
Human-readable bits: doppler2000Hz10db.txt,
Soft metrics: recovered_llrs210.npy.

9. Visualization

If plot_doppler=True:
Plots estimated Doppler frequency before/after correction,
Shows constellations before vs. after correction.

# COMMANDS/STEPS TO REPRODUCE

# 2. Install dependencies
pip install numpy matplotlib scipy
bash
Copy code
# 3. Prepare input data
# Ensure you have:
#   rx.npy   -> received baseband samples
#   meta.json -> metadata with { "sps": .., "sample_rate": .., "clean_bits": .. }
bash
Copy code
# 4. Run the receiver
python 58689bcd-fbad-4772-9af1-afa7e824bc90.py --rx rx.npy --meta meta.json --out_dir outputs
bash
Copy code
# 5. Outputs
# Saved in outputs/ :
#   doppler2000Hz10db.npy   (decoded bits, numpy format)
#   doppler2000Hz10db.txt   (decoded bits, text format)
#   recovered_llrs210.npy   (soft LLR values)

**FAILURES AND FIXES**

Failure:
Hardcoded defaults:

RX_PATH_DEFAULT = "/Users/Dell/Desktop/icarsus/.../rx.npy"
META_PATH_DEFAULT = "/Users/Dell/Desktop/icarsus/.../meta.json"


These won’t exist on another machine.

Fix:

Use relative paths or command-line arguments (already supported with --rx and --meta).

Document expected folder structure in README.md.

2. Empty or Missing Metadata

Failure:
If meta.json doesn’t contain "sps", "sample_rate", "clean_bits", or filter parameters →
defaults may be wrong (sps default is 8, fs default is 32 kHz).
This causes wrong timing recovery or BER mismatch.

Fix:
Add robust fallbacks:

sps = int(meta.get("sps", 4))   # pick more realistic default
fs  = int(meta.get("sample_rate", 48000))  # assume audio-like


Or raise an error if critical metadata is missing.

3. Very Short Signals

Failure:
If the signal is very short (few symbols),
block_len = min(128, len(syms)) → may be < 2 → Doppler estimation skipped.
Then BER will be high because CFO not corrected.

Fix:
Guard against too-short input:

if len(syms) < 256:
    print("⚠️ Warning: Input too short, skipping Doppler correction")

4. Numerical Instability (LLRs)

Failure:
In qpsk_llrs, noise variance can get very small → division by near-zero → exploding LLRs.

sigma2 = max(sigma2, 1e-12)


mitigates it, but BER could still be wrong.

Fix:
Add clipping:

llr_real = np.clip(2*symbs.real/sigma2, -20, 20)
llr_imag = np.clip(2*symbs.imag/sigma2, -20, 20)

5. Gardner Timing Recovery Drift

Failure:
Wrong gain_mu or gain_omega → timing loop diverges → garbled constellation.

Fix:
Tune loop gains depending on sps. For example:

syms = gardner_timing_recovery(x_os, sps, gain_mu=0.005, gain_omega=0.0005)


Or expose gains as CLI arguments.

6. Costas Loop Lock Failure

Failure:
If SNR is low or Doppler too high, Costas loop may not lock, constellation will spin.

Fix:

Increase loop bandwidth:

syms_cd = costas_qpsk(syms_cd, loop_bandwidth=0.05)


Or add a coarse frequency search before Costas.

7. Memory / Performance

Failure:
For very long signals (millions of samples), fftconvolve may be slow.

Fix:
Switch to block-wise convolution or use lfilter for RRC filtering.

8. Output File Overwrites

Failure:
Script always writes to:

doppler2000Hz10db.npy
doppler2000Hz10db.txt
recovered_llrs210.npy


Running multiple experiments overwrites results.

Fix:
Auto-generate filenames with timestamp/SNR/Doppler tags:

tag = f"{int(fs/1000)}kHz_{sps}sps"
np.save(f"doppler_{tag}.npy", bits_hard)



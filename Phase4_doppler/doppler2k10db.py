

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# -----------------------------
# User paths (edit if needed)
# -----------------------------
RX_PATH_DEFAULT = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase4_doppler/doppler_2000Hz/snr_10db/rx.npy"
META_PATH_DEFAULT = "/Users/Dell/Desktop/icarsus/cubesat_dataset/phase4_doppler/doppler_2000Hz/snr_10db/meta.json"

# -----------------------------
# Output folder setup
# -----------------------------
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_BITS_NPY = os.path.join(OUTPUT_DIR, "doppler2000Hz10db.npy")
OUTPUT_BITS_TXT = os.path.join(OUTPUT_DIR, "doppler2000Hz10db.txt")
OUTPUT_LLRS     = os.path.join(OUTPUT_DIR, "recovered_llrs210.npy")

# -----------------------------
# Helper functions
# -----------------------------
def rrc_filter(beta, sps, span):
    N = span*sps
    t = np.arange(-N/2, N/2+1)/float(sps)
    h = np.zeros_like(t)
    pi = np.pi
    for i, ti in enumerate(t):
        if np.isclose(ti,0.0):
            h[i] = 1.0 - beta + 4*beta/pi
        elif beta != 0 and np.isclose(abs(ti), 1.0/(4*beta)):
            h[i] = (beta/np.sqrt(2))*((1+2/pi)*np.sin(pi/(4*beta)) + (1-2/pi)*np.cos(pi/(4*beta)))
        else:
            num = np.sin(pi*ti*(1-beta)) + 4*beta*ti*np.cos(pi*ti*(1+beta))
            den = pi*ti*(1-(4*beta*ti)**2)
            h[i] = num/den
    return h/np.sqrt(np.sum(h**2))

def lin_interp(samples, idx):
    if idx < 0:
        return samples[0]
    elif idx >= len(samples)-1:
        return samples[-1]
    i = int(np.floor(idx))
    frac = idx - i
    return (1-frac)*samples[i] + frac*samples[i+1]

def gardner_timing_recovery(x, sps, mu_init=0.0, n_symbols=None, gain_mu=0.01, gain_omega=0.001):
    N = len(x)
    omega = float(sps)
    mu = float(mu_init)
    t = 0.0
    out = []
    if n_symbols is None:
        n_symbols = int((N - 2*sps) // sps)
    for k in range(n_symbols):
        idx = t + mu
        s_k = lin_interp(x, idx)
        s_km1 = lin_interp(x, idx - omega)
        s_mid = lin_interp(x, idx + 0.5*omega)
        err = np.real(np.conj(s_mid)*(s_k - s_km1))
        omega += gain_omega * err
        mu += omega + gain_mu * err
        t_inc = int(np.floor(mu))
        mu -= t_inc
        t += t_inc
        if (idx + 1) >= (N-1):
            break
        out.append(s_k)
    return np.array(out, dtype=np.complex128)

def costas_qpsk(symbs, loop_bandwidth=0.01):
    N = len(symbs)
    out = np.zeros(N, dtype=np.complex128)
    phase = 0.0
    freq = 0.0
    alpha = 0.132 * loop_bandwidth
    beta = 0.00932 * loop_bandwidth**2
    for n in range(N):
        mixed = symbs[n] * np.exp(-1j*phase)
        dec_real = 1.0 if mixed.real >=0 else -1.0
        dec_imag = 1.0j if mixed.imag >=0 else -1.0j
        dec = (dec_real + dec_imag)/np.sqrt(2)
        err = np.angle(mixed * np.conj(dec))
        freq += beta*err
        phase += freq + alpha*err
        out[n] = mixed
    return out

def qpsk_llrs(symbs, noise_var=None):
    if noise_var is None:
        hard = np.sign(symbs.real) + 1j*np.sign(symbs.imag)
        hard = hard/np.sqrt(2)
        errs = symbs - hard
        sigma2 = np.mean(np.abs(errs)**2)/2.0
    else:
        sigma2 = noise_var
    sigma2 = max(sigma2, 1e-12)
    llr_real = 2*symbs.real / sigma2
    llr_imag = 2*symbs.imag / sigma2
    return np.vstack([llr_real, llr_imag]).T

def qpsk_hard_bits(symbs):
    bits = np.zeros(2*len(symbs), dtype=np.uint8)
    bits[0::2] = (symbs.real < 0).astype(np.uint8)
    bits[1::2] = (symbs.imag < 0).astype(np.uint8)
    return bits

# -----------------------------
# Doppler estimation / correction
# -----------------------------
def estimate_cfo_per_block(r, fs, block_len=1024, hop=None):
    if hop is None:
        hop = block_len // 2
    indices = []
    freqs = []
    n = len(r)
    for start in range(0, n - block_len + 1, hop):
        blk = r[start:start+block_len]
        prod = blk[1:] * np.conj(blk[:-1])
        ang = np.angle(np.mean(prod))
        f_est = ang / (2*np.pi) * fs
        center = start + block_len // 2
        indices.append(center)
        freqs.append(f_est)
    return np.array(indices), np.array(freqs)

def apply_cfo_correction(r, fs, indices, freqs):
    if len(indices) == 0 or len(freqs) == 0:
        return r.copy()
    n = len(r)
    if len(indices) == 1:
        f0 = freqs[0]
        return r * np.exp(-1j*2*np.pi*f0*np.arange(n)/fs)
    x = np.concatenate(([0], indices, [n-1]))
    y = np.concatenate(([freqs[0]], freqs, [freqs[-1]]))
    sample_idx = np.arange(n)
    f_interp = np.interp(sample_idx, x, y)
    phase = 2*np.pi*np.cumsum(f_interp)/fs
    return r * np.exp(-1j*phase)

# -----------------------------
# Main receiver
# -----------------------------
def recover_bits(rx_path, meta_path, verbose=True, plot_doppler=True):
    rx = np.load(rx_path).astype(np.complex128)
    with open(meta_path,'r') as f:
        meta = json.load(f)
    fs = meta.get("sample_rate") or meta.get("fs") or meta.get("samp_rate") or 32000
    sps = int(meta.get("sps",8))
    if verbose:
        print(f"fs={fs}, sps={sps}, rx len={len(rx)}")

    # Matched filter
    beta = float(meta.get("rrc_rolloff",0.35))
    span = int(meta.get("rrc_span",8))
    rrc = rrc_filter(beta,sps,span)
    mf_out = fftconvolve(rx, rrc, mode="same")

    # Coarse timing offset
    best_offset = 0
    best_metric = -1
    for offset in range(sps):
        metric = np.mean(np.abs(mf_out[offset::sps]))
        if metric > best_metric:
            best_metric = metric
            best_offset = offset
    x_os = mf_out[best_offset:]

    # Gardner timing recovery
    n_symbols = int((len(x_os) - 2*sps)//sps)
    syms = gardner_timing_recovery(x_os, sps, n_symbols=n_symbols)

    # -----------------------------
    # Doppler estimation & correction
    # -----------------------------
    block_len = min(128, len(syms))
    if block_len < 2:
        syms_cd = syms.copy()
        freqs_before = np.array([0])
        freqs_after = np.array([0])
    else:
        indices, freqs_before = estimate_cfo_per_block(syms, fs, block_len=block_len)
        syms_cd = apply_cfo_correction(syms, fs, indices, freqs_before)
        _, freqs_after = estimate_cfo_per_block(syms_cd, fs, block_len=block_len)

    # Costas loop
    syms_cd = costas_qpsk(syms_cd)

    # LLRs and hard bits
    llrs = qpsk_llrs(syms_cd)
    bits_hard = qpsk_hard_bits(syms_cd)

    # BER (if available)
    gt_bits = np.array(meta.get("clean_bits",[]),dtype=np.uint8)
    if len(gt_bits)>0:
        L = min(len(bits_hard), len(gt_bits))
        errors = np.sum(bits_hard[:L] != gt_bits[:L])
        ber = errors / L
        if verbose:
            print(f"BER = {ber:.6e} ({errors}/{L})")
    else:
        if verbose:
            print("No ground truth bits found.")

    # Save outputs in outputs/ folder
    np.save(OUTPUT_BITS_NPY, bits_hard)
    np.savetxt(OUTPUT_BITS_TXT, bits_hard, fmt="%d")
    np.save(OUTPUT_LLRS, llrs)
    if verbose:
        print(f"Saved bits -> {OUTPUT_BITS_NPY}")
        print(f"Saved bits (txt) -> {OUTPUT_BITS_TXT}")
        print(f"Saved LLRs -> {OUTPUT_LLRS}")

    # -----------------------------
    # Doppler & constellation plots
    # -----------------------------
    if plot_doppler:
        mf_before = fftconvolve(rx, rrc, mode='same')
        mf_after  = fftconvolve(apply_cfo_correction(rx, fs, indices if len(indices)>0 else [0], freqs_before if len(freqs_before)>0 else [0]), rrc, mode='same')
        syms_before = mf_before[best_offset::sps]
        syms_after  = mf_after[best_offset::sps]

        fig, axes = plt.subplots(2,2, figsize=(12,8))
        # Doppler
        axes[0,0].plot(indices if len(indices)>0 else [0], freqs_before, 'r-o', label='Before correction')
        axes[0,0].plot(indices if len(indices)>0 else [0], freqs_after, 'b-o', label='After correction')
        axes[0,0].set_xlabel("Sample index (block center)")
        axes[0,0].set_ylabel("Estimated Doppler frequency (Hz)")
        axes[0,0].set_title("Doppler Frequency Compensation")
        axes[0,0].legend()
        axes[0,0].grid(True)
        axes[0,1].axis('off')
        # Constellations
        axes[1,0].plot(syms_before.real, syms_before.imag, 'ro', markersize=4)
        axes[1,0].set_xlabel("I")
        axes[1,0].set_ylabel("Q")
        axes[1,0].set_title("Constellation BEFORE Doppler correction")
        axes[1,0].grid(True)
        axes[1,0].axis('equal')
        axes[1,1].plot(syms_after.real, syms_after.imag, 'bo', markersize=4)
        axes[1,1].set_xlabel("I")
        axes[1,1].set_ylabel("Q")
        axes[1,1].set_title("Constellation AFTER Doppler correction")
        axes[1,1].grid(True)
        axes[1,1].axis('equal')
        plt.tight_layout()
        plt.show()

    return bits_hard, llrs

# -----------------------------
# Detect notebook / terminal
# -----------------------------
def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False

if __name__ == "__main__":
    if in_notebook():
        print("Detected Jupyter notebook. Calling recover_bits directly.")
        recover_bits(RX_PATH_DEFAULT, META_PATH_DEFAULT)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--rx", default=RX_PATH_DEFAULT)
        parser.add_argument("--meta", default=META_PATH_DEFAULT)
        parser.add_argument("--out_dir", default=OUTPUT_DIR, help="Output folder")
        args = parser.parse_args()

        # Update output paths
        os.makedirs(args.out_dir, exist_ok=True)
        OUTPUT_BITS_NPY = os.path.join(args.out_dir, "doppler2000Hz10db.npy")
        OUTPUT_BITS_TXT = os.path.join(args.out_dir, "doppler2000Hz10db.txt")
        OUTPUT_LLRS     = os.path.join(args.out_dir, "recovered_llrs2k10.npy")

        recover_bits(args.rx, args.meta)


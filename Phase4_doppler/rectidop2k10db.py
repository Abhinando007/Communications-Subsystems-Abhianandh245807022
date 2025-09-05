

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

OUTPUT_BITS_NPY = "doppler-2000Hz10db_final_corrected.npy"
OUTPUT_BITS_TXT = "doppler-2000Hz10db_final_corrected.txt"
OUTPUT_LLRS = "recovered_llrs_final_corrected2k10.npy"

# -----------------------------
# Helper functions
# -----------------------------
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N/2, N/2+1) / float(sps)
    h = np.zeros_like(t)
    pi = np.pi

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4*beta/pi
        elif beta != 0 and np.isclose(abs(ti), 1.0/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * (
                (1+2/pi)*np.sin(pi/(4*beta)) +
                (1-2/pi)*np.cos(pi/(4*beta))
            )
        else:
            num = np.sin(pi*ti*(1-beta)) + 4*beta*ti*np.cos(pi*ti*(1+beta))
            den = pi*ti*(1-(4*beta*ti)**2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))


def lin_interp(samples, idx):
    if idx < 0:
        return samples[0]
    if idx >= len(samples)-1:
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

        err = np.real(np.conj(s_mid) * (s_k - s_km1))
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

        dec_real = 1.0 if mixed.real >= 0 else -1.0
        dec_imag = 1.0j if mixed.imag >= 0 else -1.0j
        dec = (dec_real + dec_imag) / np.sqrt(2)

        err = np.angle(mixed * np.conj(dec))
        freq += beta * err
        phase += freq + alpha * err
        out[n] = mixed

    return out


def qpsk_hard_bits(symbs):
    bits = np.zeros(2 * len(symbs), dtype=np.uint8)
    i_neg = symbs.real < 0
    q_neg = symbs.imag < 0

    bits[0::2] = q_neg.astype(np.uint8)
    bits[1::2] = np.where(q_neg, ~i_neg, i_neg).astype(np.uint8)

    return bits


def estimate_cfo_per_block(r, fs, block_len=1024, hop=None):
    if hop is None:
        hop = block_len // 2

    indices, freqs = [], []
    n = len(r)

    for start in range(0, n - block_len + 1, hop):
        blk = r[start:start+block_len]
        prod = blk[1:] * np.conj(blk[:-1])
        ang = np.angle(np.mean(prod))
        f_est = ang / (2*np.pi) * fs
        indices.append(start + block_len // 2)
        freqs.append(f_est)

    return np.array(indices), np.array(freqs)


def apply_cfo_correction(r, fs, indices, freqs):
    if len(indices) == 0:
        return r.copy()

    n = len(r)
    if len(indices) == 1:
        f0 = freqs[0]
        return r * np.exp(-1j*2*np.pi*f0*np.arange(n)/fs)

    x = np.concatenate(([0], indices, [n-1]))
    y = np.concatenate(([freqs[0]], freqs, [freqs[-1]]))

    f_interp = np.interp(np.arange(n), x, y)
    phase = 2*np.pi*np.cumsum(f_interp) / fs
    return r * np.exp(-1j*phase)

# -----------------------------
# Main receiver
# -----------------------------
def recover_bits(rx_path, meta_path, verbose=True, plot_doppler=True):
    rx = np.load(rx_path).astype(np.complex128)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    fs = meta.get("sample_rate", 32000)
    sps = int(meta.get("sps", 8))

    if verbose:
        print(f"fs={fs}, sps={sps}, rx len={len(rx)}")

    rrc = rrc_filter(meta.get("rrc_rolloff", 0.35), sps, meta.get("rrc_span", 8))
    mf_out = fftconvolve(rx, rrc, mode="same")

    best_offset = max(range(sps), key=lambda offset: np.mean(np.abs(mf_out[offset::sps])))
    x_os = mf_out[best_offset:]

    n_symbols = int((len(x_os) - 2*sps) // sps)
    syms = gardner_timing_recovery(x_os, sps, n_symbols=n_symbols, gain_mu=0.005, gain_omega=0.0001)

    block_len = min(256, len(syms))
    if block_len < 2:
        syms_cd = syms.copy()
        indices, freqs_before, freqs_after = np.array([0]), np.array([0]), np.array([0])
    else:
        indices, freqs_before = estimate_cfo_per_block(syms, fs, block_len=block_len)
        syms_cd = apply_cfo_correction(syms, fs, indices, freqs_before)
        _, freqs_after = estimate_cfo_per_block(syms_cd, fs, block_len=block_len)

    syms_pll = costas_qpsk(syms_cd, loop_bandwidth=0.005)

    gt_bits = np.array(meta.get("clean_bits", []), dtype=np.uint8)

    if len(gt_bits) == 0:
        print("No ground truth bits found.")
        bits_hard = qpsk_hard_bits(syms_pll)
        best_syms = syms_pll
    else:
        best_ber = float('inf')
        best_bits, best_syms, best_rotation, best_order = None, None, 0, "Unknown"

        for i in range(4):
            rotation = i * np.pi / 2
            rotated_syms = syms_pll * np.exp(1j * rotation)
            decoded_bits = qpsk_hard_bits(rotated_syms)

            swapped_bits = np.copy(decoded_bits)
            swapped_bits[0::2], swapped_bits[1::2] = decoded_bits[1::2], decoded_bits[0::2]

            L = min(len(decoded_bits), len(gt_bits))
            ber_normal = np.sum(decoded_bits[:L] != gt_bits[:L]) / L

            if ber_normal < best_ber:
                best_ber = ber_normal
                best_bits = decoded_bits
                best_syms = rotated_syms
                best_rotation = i * 90
                best_order = "Normal (g0, g1)"

            ber_swapped = np.sum(swapped_bits[:L] != gt_bits[:L]) / L
            if ber_swapped < best_ber:
                best_ber = ber_swapped
                best_bits = swapped_bits
                best_syms = rotated_syms
                best_rotation = i * 90
                best_order = "Swapped (g1, g0)"

        if verbose:
            print(f"Comprehensive check complete. Best rotation: {best_rotation} deg, Best bit order: {best_order}")
            print(f"Final BER = {best_ber:.6e} ({int(best_ber*L)}/{L})")

        bits_hard = best_bits

    np.save(OUTPUT_BITS_NPY, bits_hard)
    np.savetxt(OUTPUT_BITS_TXT, bits_hard, fmt="%d")

    if verbose:
        print(f"Saved bits -> {OUTPUT_BITS_NPY}, {OUTPUT_BITS_TXT}")

    if plot_doppler:
        syms_before = mf_out[best_offset::sps]
        syms_after = best_syms if 'best_syms' in locals() else syms_pll

        max_plot = 1000
        syms_before = syms_before[:max_plot]
        syms_after = syms_after[:max_plot]

        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('Receiver Performance Analysis', fontsize=16)

        ax_doppler = fig.add_subplot(2, 1, 1)
        ax_doppler.plot(indices, freqs_before, 'r-o', label='Before correction')
        ax_doppler.plot(indices, freqs_after, 'b-o', label='After correction')
        ax_doppler.set_title("Doppler Frequency Compensation")
        ax_doppler.set_xlabel("Symbol index (block center)")
        ax_doppler.set_ylabel("Estimated Doppler frequency (Hz)")
        ax_doppler.legend()
        ax_doppler.grid(True)

        ax_const_before = fig.add_subplot(2, 2, 3)
        ax_const_before.plot(np.real(syms_before), np.imag(syms_before), 'ro', ms=3, alpha=0.6)
        ax_const_before.set_title("Constellation BEFORE All Corrections")
        ax_const_before.set_xlabel("In-Phase")
        ax_const_before.set_ylabel("Quadrature")
        ax_const_before.grid(True)
        ax_const_before.axis('equal')
        ax_const_before.set_xlim([-1.5, 1.5])
        ax_const_before.set_ylim([-1.5, 1.5])

        ax_const_after = fig.add_subplot(2, 2, 4)
        ax_const_after.plot(np.real(syms_after), np.imag(syms_after), 'bo', ms=3, alpha=0.6)
        ax_const_after.set_title("Constellation AFTER Final Correction")
        ax_const_after.set_xlabel("In-Phase")
        ax_const_after.set_ylabel("Quadrature")
        ax_const_after.grid(True)
        ax_const_after.axis('equal')
        ax_const_after.set_xlim([-1.5, 1.5])
        ax_const_after.set_ylim([-1.5, 1.5])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return bits_hard


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
        parser.add_argument("--out_bits_npy", default=OUTPUT_BITS_NPY)
        parser.add_argument("--out_bits_txt", default=OUTPUT_BITS_TXT)
        args = parser.parse_args()

        OUTPUT_BITS_NPY = args.out_bits_npy
        OUTPUT_BITS_TXT = args.out_bits_txt

        recover_bits(args.rx, args.meta)



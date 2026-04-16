import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

PATH_DATA = "/Users/BSL/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/Spectrogram_WallPressure/debug/files"


# ── Set params ────────────────────────────
# e.g. if your CFD timestep is 0.001 s, set fs = 1000.0
fs = 10000/0.915   # sampling frequency

ROI1, ROI2 = "ROI900", "ROI1190"

# Pick two node indices (ind1 from ROI1 and ind2 from ROI2)
ind1, ind2 = 300, 500

# Restrict analysis to time window [t_start, t_end] seconds
t_start, t_end = 1.0, 5.0

# ------ Load Data -----------------------------------------------------------
DATA_ROI1 = f"raw_signal_{ROI1}.npz"
DATA_ROI2 = f"raw_signal_{ROI2}.npz"

data_ROI1 = np.load(f"{PATH_DATA}/{DATA_ROI1}")
data_ROI2 = np.load(f"{PATH_DATA}/{DATA_ROI2}")

pressure_ROI1 = data_ROI1["quantity_array"]   # shape: [n_nodes, n_timesteps] --> convert [Pa] to [mmHg]
indices_ROI1  = data_ROI1["point_indices"]          # original mesh node IDs
#print(np.shape(pressure_ROI1))

pressure_ROI2 = data_ROI2["quantity_array"]   # shape: [n_nodes, n_timesteps] --> convert [Pa] to [mmHg]
indices_ROI2  = data_ROI2["point_indices"]          # original mesh node IDs
#print(np.shape(pressure_ROI2))

i_start = int(t_start * fs)
i_end   = int(t_end   * fs)
sig_a = pressure_ROI1[ind1, i_start:i_end]
sig_b = pressure_ROI2[ind2, i_start:i_end]

label_a = f"{ROI1} node {indices_ROI1[ind1]}"
label_b = f"{ROI2} node {indices_ROI2[ind2]}"




# ── Spectrogram & spectral setup ─────────────────────────────────────────────
eps      = 1e-12
nperseg  = 200 #min(64, len(sig_a) // 4)
noverlap = nperseg // 2

print(f"window length = {nperseg} and overlap = {noverlap}")

f_a, t_a, Sxx_a = signal.spectrogram(sig_a, fs=fs, nperseg=nperseg, noverlap=noverlap)
f_b, t_b, Sxx_b = signal.spectrogram(sig_b, fs=fs, nperseg=nperseg, noverlap=noverlap)

Sxx_a_dB = 10 * np.log10(Sxx_a + eps)
Sxx_b_dB = 10 * np.log10(Sxx_b + eps)
diff_dB   = np.abs(Sxx_a_dB - Sxx_b_dB)


# Welch PSD
f_psd_a, psd_a = signal.welch(sig_a, fs=fs, nperseg=nperseg)
f_psd_b, psd_b = signal.welch(sig_b, fs=fs, nperseg=nperseg)

# ── Helper functions ──────────────────────────────────────────────────────────
def centroid_over_time(f, Sxx):
    return np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + eps)

def rolloff_over_time(f, Sxx, threshold=0.85):
    cumsum = np.cumsum(Sxx, axis=0)
    idx    = np.argmax(cumsum >= threshold * cumsum[-1, :], axis=0)
    return f[idx]

def windowed_rms(sig, nperseg, noverlap):
    step = nperseg - noverlap
    n_frames = (len(sig) - nperseg) // step + 1
    return np.array([np.sqrt(np.mean(sig[i*step : i*step+nperseg]**2))
                     for i in range(n_frames)])

def sc(f, psd):
    return np.sum(f * psd) / np.sum(psd)

def sbw(f, psd):
    c = sc(f, psd)
    return np.sqrt(np.sum(((f - c) ** 2) * psd) / np.sum(psd))

def rolloff_scalar(f, psd, threshold=0.85):
    cumsum = np.cumsum(psd)
    return f[np.argmax(cumsum >= threshold * cumsum[-1])]




# ── Console metrics ───────────────────────────────────────────────────────────
metrics = {
    "Total PSD Power":      [np.trapz(psd_a, f_psd_a),       np.trapz(psd_b, f_psd_b)],
    "Signal RMS (Pa)":      [np.sqrt(np.mean(sig_a**2)),     np.sqrt(np.mean(sig_b**2))],
    "Signal Std (Pa)":      [np.std(sig_a),                  np.std(sig_b)],
    "Peak Frequency (Hz)":  [f_psd_a[np.argmax(psd_a)],      f_psd_b[np.argmax(psd_b)]],
    "Spectral Rolloff 85%": [rolloff_scalar(f_psd_a, psd_a), rolloff_scalar(f_psd_b, psd_b)],
    "Spectral Centroid Hz": [sc(f_psd_a, psd_a),             sc(f_psd_b, psd_b)],
    "Spectral Bandwidth":   [sbw(f_psd_a, psd_a),            sbw(f_psd_b, psd_b)],
}


print("\n── Spectral Metrics ───────────────────────────────────────────────────────────────")
print(f"{'Metric':<25} {label_a:>22} {label_b:>22}  Difference")
print("-" * 85)
for metric, (val_a, val_b) in metrics.items():
    rel_diff = abs(val_a - val_b) / (abs(val_a) + eps)
    print(f"{metric:<25} {val_a:>22.4f} {val_b:>22.4f}  {rel_diff:.1%}")

# ════════════════════════════════════════════════════════════════════════════
# Single 3×3 figure
# ════════════════════════════════════════════════════════════════════════════
LW_A, LW_B = 2.5, 2               # Linewidth for each node
CLR_A, CLR_B = 'black', 'limegreen'   # Color for each node

# Shift spectrogram time axes to absolute time
t_a = t_a + t_start
t_b = t_b + t_start

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(f"Wall-Pressure Comparison: {label_a} vs. {label_b}", fontsize=13, fontweight='bold')

# ── Row 0: Spectrograms (A and B fixed -60 to 0 dB; diff auto) ──────────────
spectrogram_panels = [
    (Sxx_a_dB, t_a, label_a,             'inferno', -60, 0),
    (Sxx_b_dB, t_b, label_b,             'inferno', -60, 0),
    (diff_dB,  t_a, "|Difference| (dB)", 'viridis', None, None),
]
for col, (Sxx_dB, t, title, cmap, vmin_c, vmax_c) in enumerate(spectrogram_panels):
    ax = axes[0, col]
    kwargs = dict(shading='gouraud', cmap=cmap)
    if vmin_c is not None:
        kwargs.update(vmin=vmin_c, vmax=vmax_c)
    im = ax.pcolormesh(t, f_a, Sxx_dB, **kwargs)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Freq (Hz)", fontsize=8)
    fig.colorbar(im, ax=ax, pad=0.02, label="dB")

# ── Pre-compute time-varying quantities ──────────────────────────────────────
centroid_a = centroid_over_time(f_a, Sxx_a)
centroid_b = centroid_over_time(f_b, Sxx_b)

rms_a = windowed_rms(sig_a, nperseg, noverlap)
rms_b = windowed_rms(sig_b, nperseg, noverlap)
n_min = min(len(rms_a), len(rms_b), len(t_a), len(t_b))
t_rms = t_a[:n_min]

rolloff_a = rolloff_over_time(f_a, Sxx_a)
rolloff_b = rolloff_over_time(f_b, Sxx_b)

peak_freq_a = f_a[np.argmax(Sxx_a, axis=0)]
peak_freq_b = f_b[np.argmax(Sxx_b, axis=0)]

t_sig = np.linspace(t_start, t_end, len(sig_a))

# ── Row 1: Pressure signals | Signal RMS | PSD ───────────────────────────────
ax = axes[1, 0]
ax.plot(t_sig, sig_a/133.3,       color=CLR_A, linewidth=LW_A, label='Node A')
ax.plot(t_sig, sig_b/133.3, '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("Pressure Signals", fontsize=9)
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("Pressure (mmHg)", fontsize=8)
ax.legend(fontsize=7)

ax = axes[1, 1]
ax.plot(t_rms, rms_a[:n_min],       color=CLR_A, linewidth=LW_A, label='Node A')
ax.plot(t_rms, rms_b[:n_min], '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("Signal RMS (Pa)", fontsize=9)
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("RMS (Pa)", fontsize=8)
ax.legend(fontsize=7)

ax = axes[1, 2]
ax.semilogy(f_psd_a, psd_a,       color=CLR_A, linewidth=LW_A, label='Node A')
ax.semilogy(f_psd_b, psd_b, '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("PSD — Welch", fontsize=9)
ax.set_xlabel("Freq (Hz)", fontsize=8)
ax.set_ylabel("PSD (Pa²/Hz)", fontsize=8)
ax.legend(fontsize=7)

# ── Row 2: Spectral Centroid | Rolloff | Peak Frequency ──────────────────────
ax = axes[2, 0]
ax.plot(t_a, centroid_a,       color=CLR_A, linewidth=LW_A, label='Node A')
ax.plot(t_b, centroid_b, '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("Spectral Centroid", fontsize=9)
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("Centroid (Hz)", fontsize=8)
ax.legend(fontsize=7)

ax = axes[2, 1]
ax.plot(t_a, rolloff_a,       color=CLR_A, linewidth=LW_A, label='Node A')
ax.plot(t_b, rolloff_b, '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("Spectral Rolloff 85% (Hz)", fontsize=9)
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("Freq (Hz)", fontsize=8)
ax.legend(fontsize=7)

ax = axes[2, 2]
ax.plot(t_a, peak_freq_a,       color=CLR_A, linewidth=LW_A, label='Node A')
ax.plot(t_b, peak_freq_b, '--', color=CLR_B,  linewidth=LW_B, label='Node B')
ax.set_title("Instantaneous Peak Frequency (Hz)", fontsize=9)
ax.set_xlabel("Time (s)", fontsize=8)
ax.set_ylabel("Freq (Hz)", fontsize=8)
ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

import numpy as np
from pathlib import Path
from scipy import signal
from scipy.io import wavfile

"""
spectrogram_data = {'power_avg_dB': S_avg_dB,
            'bins': bins,
            'freqs': freqs,
            'sampling_rate': sampling_rate,
            'n_fft': n_fft,
            'window_length': window_length,
            'overlap_frac': overlap_frac}
"""


def griffin_lim_from_magnitude(sig_amplitude: np.ndarray,
                               sampling_rate: float,
                               n_seg: int,
                               n_overlap: int,
                               n_fft: int,
                               window_type: str = "hann",
                               n_iter: int = 80,
                               seed: int = 0):
    """
    Griffin–Lim phase reconstruction from magnitude spectrogram.

    signal_amplitude: shape (n_freq, n_frames), magnitude (not power)
    Returns: reconstructed time signal (float64)
    """
    rng = np.random.default_rng(seed)
    angles = np.exp(1j * 2.0 * np.pi * rng.random(sig_amplitude.shape))
    sig_complex = sig_amplitude * angles

    for _ in range(n_iter):
        # Inverse STFT to time domain
        _, sig = signal.istft(sig_complex, fs=sampling_rate, window=window_type, nperseg=n_seg, noverlap=n_overlap, nfft=n_fft, input_onesided=True, boundary=True)

        # Forward STFT to update phase
        _, _, sig_phase = signal.stft(sig, fs=sampling_rate, window=window_type, nperseg=n_seg, noverlap=n_overlap, nfft=n_fft, boundary="zeros", padded=True, return_onesided=True)

        # Keep the target magnitude, update phase
        sig_complex = sig_amplitude * np.exp(1j * np.angle(sig_phase))

    # Final reconstruction
    _, sig_final = signal.istft(sig_complex, fs=sampling_rate, window=window_type, nperseg=n_seg, noverlap=n_overlap, nfft=n_fft, input_onesided=True, boundary=True)
    return sig_final


def spectrogram_npz_to_audio(input_path_npz: str, output_path_wav: str = "spectrogram_audio.wav", n_iter: int = 80):
    """
    Load your saved spectrogram_data (.npz), map it to an audio-like STFT,
    and synthesize a waveform.

    Notes:
    - audio_fs is the audio sampling rate you want (Hz).
    - The time scaling will follow your spectrogram frame spacing.
      If your spectrogram was computed from CFD time steps, this gives a
      meaningful time axis; otherwise it's just a sonification.
    """
    data = np.load(input_path_npz, allow_pickle=True)
    
    spec = data["arr_0"].item()

    bins          = spec["bins"]
    signal_dB     = spec["power_avg_dB"]
    window_length = int(spec["window_length"])
    overlap_frac  = float(spec["overlap_frac"])
    n_fft         = spec.get("n_fft") or window_length

    # Convert your normalized dB spectrogram -> relative power -> magnitude
    signal_power     = 10.0 ** (signal_dB / 10.0)                # relative power
    signal_amplitude = np.sqrt(np.maximum(signal_power, 1e-12))  # magnitude (avoid zeros)


    # --- Time-scale mapping to audio ---
    # Your STFT frame times are encoded in `bins` (seconds in your computation)
    # We will "resample time" by adjusting the audio sampling rate target.
    # But your bins spacing encodes hop timing. To preserve the hop timing, we pick an audio_fs so that hop/audio_fs ~= dt_frame.

    n_seg = window_length
    n_overlap = int(round(overlap_frac * n_seg))
    n_hop = n_seg - n_overlap

    # Time spacing between consecutive spectrogram frames (in original units)
    dt_frame = float(bins[1] - bins[0]) #[s]
    
    # Desired audio sampling rate
    audio_fs = n_hop / dt_frame

    signal = griffin_lim_from_magnitude(sig_amplitude = signal_amplitude, sampling_rate=audio_fs, n_seg=n_seg, n_overlap=n_overlap, n_fft=n_fft, window_type="hann", n_iter=n_iter, seed=0)

    # Clean the signal
    t_start = 1 #[s]
    t_end   = 5 #[s]
    ind_start = int(t_start * audio_fs)
    ind_end   = int(t_end * audio_fs)

    signal = signal[ind_start:ind_end]    # remove the ends of the signal
    signal = signal - np.mean(signal)     # removing DC offset from signal
    peak = np.max(np.abs(signal))
    signal = 0.98 * signal / peak         # normalizing signal to its peak amplitude (used 0.98 instead of 1 to avoid overflowing when rounding)
    signal_int16 = signal * (2**15 - 1)   # convert signal to int16 range

    # Writing the audio from the reconstructed signal
    wavfile.write(output_path_wav, int(audio_fs), signal_int16.astype(np.int16))



if __name__ == "__main__":

    BASE_DIR    = "/Users/rojin/Dropbox/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/Spectrogram_WallPressure/run2_centerline_fine/window2732_overlap0.9_ROIcylinder_multiROITrue/"
    INPUT_PATH  = Path(BASE_DIR) / "files"
    OUTPUT_PATH = Path(BASE_DIR) / "audios"
    
    # Create output folders
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Loop over all spectrogram files and generate audios
    for npz_file in INPUT_PATH.glob("*.npz"):
        output_file = OUTPUT_PATH / (npz_file.stem + ".wav")
        
        print(f"Processing: {npz_file.name}")
        spectrogram_npz_to_audio(npz_file, output_file, n_iter=80)
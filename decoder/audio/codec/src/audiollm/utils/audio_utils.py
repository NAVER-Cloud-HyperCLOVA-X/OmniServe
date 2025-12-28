# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import Union

import numpy as np
import scipy.signal
import torch
from librosa.filters import mel as librosa_mel_fn

VOLUME_LEVEL_DB = -26
VOLUME_LEVEL = 10 ** (VOLUME_LEVEL_DB / 20)


# Global caches for mel filter banks and Hann windows.
mel_basis = {}
hann_window = {}


def hpf_normalize(
    pcm: np.ndarray, sr: Union[int, float], volume_level: float
) -> np.ndarray:
    assert (pcm**2).mean() > 0, "Error in the wav file"
    assert np.issubdtype(pcm.dtype, np.floating)

    # highpass filter
    filter_ = scipy.signal.butter(2, 70, "highpass", fs=sr, output="sos")
    pcm = scipy.signal.sosfilt(filter_, pcm)
    pcm = pcm.astype(np.float32)

    # volume normalize
    gain = min(volume_level / (pcm**2).mean() ** 0.5, 1 / np.max(np.abs(pcm)))
    pcm *= gain
    return pcm


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def compute_mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute the mel spectrogram from an audio waveform.

    This function calculates the mel spectrogram from an input audio tensor `y`
    using the Short-Time Fourier Transform (STFT) and applies mel filter banks.
    The process includes reflective padding of the waveform, computation of the STFT,
    mapping of the frequency bins to the mel scale, and dynamic range compression
    for spectral normalization.

    Parameters:
        y (torch.Tensor): Audio waveform tensor.
        n_fft (int): FFT size.
        num_mels (int): Number of mel filter banks.
        sampling_rate (int): Sampling rate of the audio.
        hop_size (int): Hop size for the STFT.
        win_size (int): Window size for the STFT.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        center (bool, optional): If True, pads the signal so that the t-th frame is centered at time t.

    Returns:
        torch.Tensor: The computed mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print("min value is", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is", torch.max(y))

    global mel_basis, hann_window
    # Create a unique key based on fmax and device
    key = f"{fmax}_{y.device}"
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Pad the signal for STFT
    pad_amount = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect"
    ).squeeze(1)

    # Compute the Short-Time Fourier Transform (STFT)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Compute the magnitude spectrogram with a small epsilon to avoid log(0)
    spec = torch.sqrt(torch.real(spec * spec.conj() + 1e-9))

    # Map the linear-frequency spectrogram to the mel scale
    spec = torch.matmul(mel_basis[key], spec)

    # Apply spectral normalization (dynamic range compression)
    spec = spectral_normalize_torch(spec)

    return spec

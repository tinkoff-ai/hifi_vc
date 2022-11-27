from dataclasses import dataclass
from typing import Optional

import librosa
import torch
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}


@dataclass
class AudioFeaturesParams:
    win_size: int = 1024
    hop_size: int = 240
    num_mel_bins: int = 80
    sampling_rate: int = 24000
    fmin: float = 50.0
    fmax: Optional[float] = 11025.0
    n_fft: int = 2048
    central_padding: bool = False


def load_and_preprocess_audio(audio_file: str, sr: int, trim=False) -> torch.Tensor:
    audio, _ = librosa.load(audio_file, sr=sr)

    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=10)

    audio = torch.FloatTensor(audio).squeeze()
    audio /= torch.abs(audio).max()

    audio = audio.unsqueeze(0)
    return audio


def dynamic_range_compression_torch(
    x: torch.Tensor, C: int = 1, clip_val: float = 1e-5
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
    audio: torch.Tensor, audio_params: AudioFeaturesParams
) -> torch.Tensor:
    """Computes melspectrogram according to parameters

    Args:
        audio [wav_frames] or [1, wav_frames]: input wav
        audio_params : parameters of input audio and desired mel
        loss : min and max frequencies depend of that argument

    Returns:
        [num_bins, mel_frames] : melspectrogram
    """
    if audio.ndim < 2:
        audio = audio.unsqueeze(0)

    fmax = audio_params.fmax
    fmin = audio_params.fmin

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            audio_params.sampling_rate,
            audio_params.n_fft,
            audio_params.num_mel_bins,
            fmin,
            fmax,
        )
        mel_basis[str(fmin) + "_" + str(fmax) + "_" + str(audio.device)] = (
            torch.from_numpy(mel).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(
            audio_params.win_size
        ).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        [
            int((audio_params.n_fft - audio_params.hop_size) / 2),
            int((audio_params.n_fft - audio_params.hop_size) / 2),
        ],
        mode="reflect",
    )
    audio = audio.squeeze(1)

    spec = torch.stft(
        audio,
        audio_params.n_fft,
        hop_length=audio_params.hop_size,
        win_length=audio_params.win_size,
        window=hann_window[str(audio.device)],
        center=audio_params.central_padding,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(
        mel_basis[str(fmin) + "_" + str(fmax) + "_" + str(audio.device)], spec
    )
    spec = spectral_normalize_torch(spec)

    return spec

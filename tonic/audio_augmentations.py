import os
import random
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import torch

from tonic.audio_transforms import FixLength

# import torchaudio
# from qut_noise import QUTNoise
# from torchaudio.utils import download_asset


__all__ = [
    "RandomTimeStretch",
    "RandomPitchShift",
    "RandomAmplitudeScale",
    "AddWhiteNoise",
    "AddHomeNoise",
    "EmbeddedHomeNoise",
    "RIR",
]


@dataclass
class RandomTimeStretch:
    """Time-stretch an audio sample by a fixed rate.
    Parameters:
        samplerate (float): sample rate of the sample
        sample_length (int): sample length in seconds
        factors (float): range of desired factors for time stretch
        aug_index (int): index of the chosen factor for time stretch. It will be randomly chosen from the desired range (if not passed while initilization)
        caching (bool): if we are caching the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform
        fix_length (bool): if True, time stretched signal will be returned in a fixed length (samplerate * sample_length )
    Args:
        audio (np.ndarray): data sample
    Returns:
        np.ndarray: time stretched data sample
    """

    samplerate: float
    sample_length: int
    factors: list = field(default_factory=lambda: [0.8, 0.9, 1.1, 1.2])
    aug_index: int = 0
    caching: bool = False
    fix_length: bool = True

    def __call__(self, audio: np.ndarray):
        if not self.caching:
            self.aug_index = random.choice(range(0, len(self.factors)))
        factor = self.factors[self.aug_index]
        stretched = librosa.effects.time_stretch(audio, rate=factor)
        if self.fix_length:
            fix_length = FixLength(length=self.sample_length * self.samplerate)
            stretched = fix_length(stretched)

        return stretched


@dataclass
class RandomPitchShift:
    """Shift the pitch of a waveform by n_steps steps .

    Parameters:
        samplerate (float): sample rate of the sample
        sample_length (int): sample length in seconds
        factors (float): range of desired factors for pitch shift
        aug_index (int): index of the chosen factor for pitchshift. It will be randomly chosen from the desired range (if not passed while initilization)
        caching (bool): if we are caching, the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform

    Args:
        audio (np.ndarray): data sample
    Returns:
        np.ndarray: pitch shifted data sample
    """

    samplerate: float
    sample_length: int
    factors: list = field(
        default_factory=lambda: list(range(-5, 0)) + list(range(1, 6))
    )

    aug_index: int = 0
    caching: bool = False

    def __call__(self, audio: np.ndarray):
        if not self.caching:
            self.aug_index = random.choice(range(0, len(self.factors)))

        semitone = self.factors[self.aug_index]
        return librosa.effects.pitch_shift(audio, sr=self.samplerate, n_steps=semitone)


@dataclass
class RandomAmplitudeScale:
    """Scales the maximum amplitude of the incoming signal to a random amplitude chosen from a
    range.

    Parameters:
        samplerate (float): sample rate of the sample
        sample_length (int): sample length in seconds
        min_amp (float): minimum of the amplitude range in volts
        max_amp (float): maximum of the amplitude range in volts
        factors (float): range of desired factors for amplitude scaling
        aug_index (int): index of the chosen factor for pitchshift. It will be randomly chosen from the desired range (if not passed while initilization)
        caching (bool): if we are caching, the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform


    Args:
        data (np.ndarray): input (single-or multi-channel) signal.

    Returns:
        np.ndarray: scaled version of the signal.
    """

    samplerate: float
    sample_length: float
    min_amp: float = 0.058
    max_amp: float = 0.159
    factors: list = field(
        default_factory=lambda: list(range(int(1000 * 0.058), int(1000 * 0.159), 10))
    )
    aug_index: int = 0
    caching: bool = False

    def __post_init__(self):
        # converted to mVolt
        self.factors = list(
            range(int(1000 * self.min_amp), int(1000 * self.max_amp), 10)
        )

    def __call__(self, audio: np.ndarray):
        if not self.caching:
            self.aug_index = random.choice(range(0, len(self.factors)))

        max_amplitude = self.factors[self.aug_index] / 1000
        max_data_amplitude = np.max(np.abs(audio.ravel()))
        if max_data_amplitude == 0:
            max_data_amplitude = 1.0

        return audio / max_data_amplitude * max_amplitude


@dataclass
class AddWhiteNoise:
    """Add white noise to the data sample with a known ratio.

    Parameters:
        samplerate (float): sample rate of the sample
        factors (float): range of desired ratios for added noise
        aug_index (int): index of the chosen factor for noise. It will be randomly chosen from the desired range (if not passed while initilization)
        caching (bool): if we are caching the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform

    Args:
        audio (np.ndarray): data sample
    Returns:
        np.ndarray: data sample with added noise
    """

    samplerate: float
    factors: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    aug_index: int = 0
    caching: bool = False

    def __call__(self, audio: np.ndarray):
        if not self.caching:
            self.aug_index = random.choice(range(0, len(self.factors)))

        noise = np.random.normal(0, audio.std(), audio.size)
        noise_factor = self.factors[self.aug_index]
        noisy_audio = audio + noise_factor * noise
        return noisy_audio


# @dataclass
# class AddHomeNoise:
#     """Add a home background noise (from QUTNOise dataset) to the audio sample with a known snr
#     (signal to noise ratio).

#     Parameters:
#         sample_length (int): sample length in seconds
#         target_sr (float): the target sample rate of the mixed final signal (default to the higher sample rate, between sample rates of noise and data )
#         params_dataset (dict): containing other parameters of the noise dataset
#         orig_sr (float): original sample rate of data
#         factors (float): range of desired snrs
#         partition (str): partition of the QUTNoise dataset that is used for noise augmentation
#         aug_index (int): index of the chosen factor for snr. It will be randomly chosen from the desired range (if not passed while initilization)
#         caching (bool): if we are caching the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform
#         seed (int): a fixed seed for reproducibility
#     Args:
#         audio (np.ndarray): data sample
#     Returns:
#         np.ndarray: data sample with added noise
#     """

#     sample_length: int
#     params_dataset: dict
#     target_sr: float = 48000
#     orig_sr: float = 16000
#     factors: list = field(default_factory=lambda: [0, 10, 20])
#     partition: str = "test"
#     aug_index: int = 0
#     caching: bool = False
#     seed: int = 123

#     def __post_init__(self):
#         random.seed(self.seed)

#         noises = QUTNoise(
#             classes=["HOME"],
#             create_splits=False,
#             duration_split=[self.sample_length],
#             partition=self.partition,
#             **self.params_dataset,
#         )

#         split_qutnoise_path = noises.config_path

#         self.wave_files_path = (
#             str(split_qutnoise_path)
#             + "/splits_"
#             + str(self.sample_length)
#             + "s"
#             + "/"
#             + self.partition
#             + "/"
#         )

#         self.home_noises = os.listdir(self.wave_files_path)

#     def resample(self, audio):
#         audio_resampled = librosa.resample(
#             audio, orig_sr=self.orig_sr, target_sr=self.target_sr
#         )
#         return audio_resampled

#     def get_noise(self):
#         self.noise_wave = random.choice(self.home_noises)

#         noise, _ = librosa.core.load(
#             self.wave_files_path + self.noise_wave, sr=self.target_sr
#         )
#         self.noise = noise[0 : int(self.target_sr) * self.sample_length]
#         return self.noise

#     def add_noise(
#         self,
#         waveform: torch.Tensor,
#         noise: torch.Tensor,
#         snr: torch.Tensor,
#     ) -> torch.Tensor:
#         """Scales and adds noise to waveform per signal-to-noise ratio.

#         Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
#         :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as
#         .. math::
#             y = x + a n \, \text{,}
#         where
#         .. math::
#             a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}
#         with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.
#         Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
#         consistent with the above formulae and PyTorch's broadcasting semantics.
#         .. devices:: CPU CUDA
#         .. properties:: Autograd TorchScript
#         Args:
#             waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
#             noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
#             snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
#         Returns:
#             torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
#             (same shape as ``waveform``).
#         """

#         L = waveform.size(-1)

#         if L != noise.size(-1):
#             raise ValueError(
#                 f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)})."
#             )

#         # compute scale, second by second
#         noisy_audio = torch.zeros_like(waveform)
#         for i in range(0, self.sample_length):
#             start, end = int(i * self.target_sr), int((i + 1) * self.target_sr)
#             sig, noise_ = waveform[:, start:end], noise[:, start:end]

#             energy_signal = torch.linalg.vector_norm(sig, ord=2, dim=-1) ** 2  # (*,)
#             energy_noise = torch.linalg.vector_norm(noise_, ord=2, dim=-1) ** 2  # (*,)
#             original_snr_db = 10 * (
#                 torch.log10(energy_signal) - torch.log10(energy_noise)
#             )
#             scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

#             # scale noise
#             self.scaled_noise = scale.unsqueeze(-1) * noise_  # (*, 1) * (*, L) = (*, L)
#             noisy_audio[:, start:end] = sig + self.scaled_noise

#         return noisy_audio

#     def __call__(self, audio: np.ndarray):
#         if not self.caching:
#             self.aug_index = random.choice(range(0, len(self.factors)))
#         snr_db = torch.tensor([self.factors[self.aug_index]])
#         self.noise = torch.from_numpy(self.get_noise())
#         self.noise = torch.unsqueeze(self.noise, dim=0)
#         self.resampled_audio = torch.from_numpy(self.resample(audio))
#         noisy_audio = self.add_noise(self.resampled_audio, self.noise, snr_db)

#         return noisy_audio.detach().numpy()


# @dataclass
# class EmbeddedHomeNoise(AddHomeNoise):
#     """Add a home background noise (from QUTNOise dataset) to the data sample with a known snr_db
#     (signal to noise ratio).

#     The difference with AddHomeNoise is that a leading (/and trainling) noise will be added to the augmented sample.
#     Parameters:
#         noise_length (int): the length of noise (in seconds) that will be added to the sample
#         two_sided (bool): if True the augmented signal will be encompassed between leading and trailing noises
#     Args:
#         audio (np.ndarray): data sample
#     Returns:
#         np.ndarray: data sample with added noise at the begining
#     """

#     noise_length: int = None
#     two_sided: bool = False

#     def __post_init__(self):
#         super().__post_init__()

#         if self.noise_length is None:
#             raise ValueError("noise length is not specified")
#         elif self.noise_length > self.sample_length:
#             raise ValueError(
#                 "in the current implementation length of noise can't exceed sample length"
#             )

#     def __call__(self, audio: np.ndarray):
#         if not self.caching:
#             self.aug_index = random.choice(range(0, len(self.factors)))
#         snr_db = torch.tensor([self.factors[self.aug_index]])

#         self.noise = torch.from_numpy(self.get_noise())
#         self.noise = torch.unsqueeze(self.noise, dim=0)
#         self.resampled_audio = torch.from_numpy(self.resample(audio))
#         noisy_audio = (
#             self.add_noise(self.resampled_audio, self.noise, snr_db).detach().numpy()
#         )

#         initial_noise = self.scaled_noise[
#             :, 0 : int(self.target_sr * self.noise_length)
#         ]
#         if self.two_sided:
#             noise_then_audio = np.concatenate(
#                 (initial_noise, noisy_audio, initial_noise), axis=1
#             )
#         else:
#             noise_then_audio = np.concatenate((initial_noise, noisy_audio), axis=1)

#         return noise_then_audio


# @dataclass
# class RIR:
#     """Convolves a RIR (room impluse response, sound of hand clapping in an empty room) to the data
#     sample.

#     Parameters:
#         samplerate (float): sample rate of the sample
#         caching (bool): if we are caching the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform

#     Args:
#         audio (np.ndarray): data sample
#     Returns:
#         np.ndarray: data sample convolved with RIR
#     """

#     samplerate: float
#     caching: bool = False

#     def __call__(self, audio):
#         SAMPLE_RIR = download_asset(
#             "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
#         )
#         rir_raw, rir_sample_rate = torchaudio.load(SAMPLE_RIR)
#         rir = rir_raw[:, int(rir_sample_rate * 1.01) : int(rir_sample_rate * 1.3)]
#         rir = rir / torch.norm(rir, p=2)
#         RIR = torch.flip(rir, [1])

#         t_audio = torch.nn.functional.pad(
#             torch.from_numpy(audio), (RIR.shape[1] - 1, 0)
#         )
#         rir_augmented = torch.nn.functional.conv1d(t_audio[None, ...], RIR[None, ...])[
#             0
#         ].numpy()

#         return rir_augmented

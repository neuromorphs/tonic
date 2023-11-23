import os
import random
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.utils import download_asset

from tonic.audio_transforms import FixLength

__all__ = [
    "RandomTimeStretch",
    "RandomPitchShift",
    "RandomAmplitudeScale",
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
        caching (bool): if we are caching the DiskCached dataset will dynamically pass copy index of data item to the transform (to set aug_index). Otherwise the aug_index will be chosen randomly in every call of transform

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

import torch
from typing import Tuple, List, Union, Iterator
import torch.nn.functional as F
from dataclasses import dataclass
from scipy.signal import butter
from torchaudio.functional import lfilter
import numpy as np


@dataclass
class FixLength:
    """
    Fix the length of a sample along a specified axis to a given length.

    Parameters:
        length:
            Desired length of the sample
        axis:
            Dimension along which the length needs to be fixed.
    Args:
        data: torch.Tensor
    Returns:
        torch.Tensor of the same dimension
    """
    length: int
    axis: int = 1

    def __call__(self, data: torch.Tensor):
        data_length = data.shape[self.axis]
        if data_length == self.length:
            return data
        elif data_length > self.length:
            data_splits = torch.split(data, self.length, self.axis)
            return data_splits[0]
        else:
            padding = []
            for cur_axis, axis_len in enumerate(data.shape):
                if cur_axis == self.axis:
                    padding = [0, self.length - data_length] + padding
                else:
                    padding = [0, 0] + padding
            return F.pad(data, padding)


@dataclass
class Bin:
    """
    Bin the given data along a specified axis at the specivied new frequency

    Parameters:
        orig_freq: float
            Sampling frequency of the given data stream
        new_freq: float
            Desired frequency after binning
        axis: int
            Axis along which the data needs to be binned

    Args:
         data: torch.Tensor
            data to be binned

    Returns:
        torch.Tensor binned data

    """
    orig_freq: float
    new_freq: float
    axis: int

    def __call__(self, data: torch.Tensor):
        splits = torch.split(data, int(self.orig_freq/self.new_freq), dim=self.axis)
        data = [torch.sum(split, dim=self.axis, keepdim=True) for split in splits]
        return torch.cat(data, self.axis)


@dataclass
class LFilter:
    """
    See documentation of torchaudio.functional.lfilter for detailed explanation of parameters
    """
    a_coeffs: torch.Tensor
    b_coeffs: torch.Tensor
    clamp: bool

    def __call__(self, data):
        return lfilter(data, a_coeffs=self.a_coeffs, b_coeffs=self.b_coeffs, clamp=False)


@dataclass
class ButterFilter:
    order: int
    freq: Union[float, Tuple[float, float]]
    analog: bool
    btype: str
    clamp: bool

    def __post_init__(self):
        b_coeffs, a_coeffs = butter(self.order, self.freq, analog=self.analog, btype=self.btype, output="ba")
        b_coeffs = torch.tensor(b_coeffs)
        a_coeffs = torch.tensor(a_coeffs)
        self.filter = LFilter(a_coeffs, b_coeffs, self.clamp)

    def __call__(self, data):
        return self.filter(data)


@dataclass
class ButterFilterBank:
    order: int
    freq: List[Tuple[float, float]]
    clamp: bool

    def __post_init__(self):
        self.filters = [ButterFilter(self.order, freq, analog=False, btype="band", clamp=self.clamp) for freq in self.freq]

    def __call__(self, data):
        return torch.cat([filt(data) for filt in self.filters], dim=0)


@dataclass
class LinearButterFilterBank:
    order: int = 2
    low_freq: float = 100
    sampling_freq: float = 16000
    num_filters: int = 64
    clamp: bool = False

    def compute_freq_bands(self):
        filter_bandwidth = 2 / self.num_filters
        nyquist = self.sampling_freq / 2

        high_freq = self.sampling_freq / 2 / (1 + filter_bandwidth) - 1
        freqs = np.linspace(self.low_freq, high_freq, self.num_filters)

        return torch.tensor([freqs, freqs * (1 + filter_bandwidth)]).T/ nyquist

    def __post_init__(self):
        freq_bands = self.compute_freq_bands()
        self.filterbank = ButterFilterBank(order=self.order, freq=freq_bands, clamp=self.clamp)

    def __call__(self, data):
        return self.filterbank(data)



@dataclass
class MelButterFilterBank(LinearButterFilterBank):
    @staticmethod
    def hz2mel(freq):
        return 2595 * np.log10(1 + freq / 700)

    @staticmethod
    def mel2hz(freq):
        return 700 * (10 ** (freq / 2595) - 1)

    def compute_freq_bands(self):
        filter_bandwidth = 2 / self.num_filters
        nyquist = self.sampling_freq / 2

        high_freq = self.sampling_freq / 2 / (1 + filter_bandwidth) - 1
        freqs = np.linspace(self.low_freq, high_freq, self.num_filters)

        freq_bands = np.array([freqs, freqs * (1 + filter_bandwidth)])/ nyquist

        low_freq = self.hz2mel(self.low_freq)
        high_freq = self.hz2mel(self.sampling_freq / 2 / (1 + filter_bandwidth) - 1)
        freqs = self.mel2hz(np.linspace(low_freq, high_freq, self.num_filters))

        return torch.tensor([freqs, freqs * (1 + filter_bandwidth)]).T/nyquist


@dataclass
class AddNoise:
    """
    Add nose to data

    Params:
        dataset:
            A dataset object that returns a tuple when iterated over the first element of which is the audio signal to be used for noise.
        snr:
            Desired signal to noise ratio in dB
        normed:
            If set to false, the signal max value will not be normalized. True by default.
    """
    dataset: Iterator
    snr: float
    normed: bool = True

    def get_noise_sample(self, sample_len: int) -> torch.Tensor:
        """Get a random noise sample from the dataset"""
        print(sample_len)
        # Find noise sample of minimum length
        while True:
            noise_idx = torch.randint(0, len(self.dataset), (1,)).item()
            noise = self.dataset[noise_idx][0]
            print(noise.shape)
            if noise.shape[1] >= sample_len:
                break
        # Sample a random part of the data recording
        noise_signal_len = noise.shape[1]
        if noise_signal_len > sample_len:
            start_t = torch.randint(0, noise_signal_len - sample_len, (1,)).item()
            noise = noise[:, start_t:start_t + sample_len]
        return noise

    def __call__(self, signal):
        # randomly pick a piece of noise data
        noise = self.get_noise_sample(sample_len=signal.shape[1])

        # mix signal with noise with given SNR
        signal_power = (signal ** 2).mean()
        noise_power = (noise ** 2).mean()
        noise_scale = (signal_power / noise_power) * 10 ** (-self.snr / 10)
        signal_with_snr = signal + noise_scale * noise

        # Normalize if specified
        if self.normed:
            return self.normalize(signal_with_snr)
        else:
            return signal_with_snr

    @staticmethod
    def normalize(signal):
        """Normalize the signal"""
        signal -= signal.mean()
        max_val = torch.max(torch.abs(signal))
        if max_val > 0:
            signal /= max_val
        return signal

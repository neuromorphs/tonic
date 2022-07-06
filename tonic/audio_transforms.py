import librosa
import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfilt
from typing import Tuple, List, Union, Iterator


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

    def __call__(self, data: np.ndarray):
        return librosa.util.fix_length(data, self.length, self.axis)


@dataclass
class Bin:
    """
    Bin the given data along a specified axis at the specified new frequency

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

    def __call__(self, data: np.ndarray):
        data_len = data.shape[self.axis]
        n_splits = int(data_len / (self.orig_freq / self.new_freq))
        splits = np.array_split(data, n_splits, axis=self.axis)
        data = [np.sum(split, axis=self.axis, keepdims=True) for split in splits]
        return np.concatenate(data, self.axis)


@dataclass
class SOSFilter:
    """
    SOS filter

    Parameters
    ----------

    coeffs:
        coefficients of the second order filter
    axis:
        Axis along with the filter needs to be applied

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html for more details
    """

    coeffs: np.ndarray
    axis: int

    def __call__(self, signal):
        return sosfilt(self.coeffs, signal, axis=self.axis)


@dataclass
class ButterFilter:
    """
    Butter filter

    Parameters
    ----------

    order:
        Order of filter to be used
    freq:
        Frequency for the filter (float or (float, float))
    analog:
        True if analog filter
    btype:
        Filter type, {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    rectify:
        If true, the output is the absolute value of the filtered output
    axis:
        Axis along which the filter needs to be applied

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter for more details on parameters.
    """

    order: int
    freq: Union[float, Tuple[float, float]]
    analog: bool
    btype: str
    rectify: bool
    axis: int

    def __post_init__(self):
        coeffs = butter(
            self.order, self.freq, analog=self.analog, btype=self.btype, output="sos"
        )
        self.filter = SOSFilter(coeffs, axis=self.axis)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        out = self.filter(data)
        if self.rectify:
            return np.abs(out)
        else:
            return out


@dataclass
class ButterFilterBank:
    """
    Butter filter bank

    Parameters
    ----------

    order:
        Order of filter to be used
    freq:
        Frequency for the filter (float or (float, float))
    rectify:
        If true, the output is the absolute value of the filtered output
    axis:
        Axis along which the filter needs to be applied
    analog:
        If true, the filter will be analog. False by default

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter for more details on parameters.
    """

    order: int
    freq: List[Tuple[float, float]]
    rectify: bool
    axis: int
    analog: bool = False

    def __post_init__(self):
        self.filters = [
            ButterFilter(
                self.order,
                freq,
                analog=self.analog,
                btype="band",
                rectify=self.rectify,
                axis=self.axis,
            )
            for freq in self.freq
        ]

    def __call__(self, data):
        return np.concatenate([filt(data) for filt in self.filters], axis=0)


@dataclass
class LinearButterFilterBank:
    """
    Butter filter bank

    Parameters
    ----------

    order:
        Order of filter to be used
    low_freq:
        Lower/cutoff frequency the filter (float or (float, float))
    sampling_freq:
        Sampling frequency of the signal, also serves as higher frequency of the filter bank.
    analog:
        True if analog filter
    rectify:
        If true, the output is the absolute value of the filtered output
    axis:
        Axis along which the filter needs to be applied

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter for more details on parameters.
    """

    order: int = 2
    low_freq: float = 100
    sampling_freq: float = 16000
    analog: bool = False
    num_filters: int = 64
    rectify: bool = True
    axis: int = -1

    def compute_freq_bands(self):
        filter_bandwidth = 2 / self.num_filters
        nyquist = self.sampling_freq / 2

        high_freq = self.sampling_freq / 2 / (1 + filter_bandwidth) - 1
        freqs = np.linspace(self.low_freq, high_freq, self.num_filters)

        return np.array([freqs, freqs * (1 + filter_bandwidth)]).T / nyquist

    def __post_init__(self):
        freq_bands = self.compute_freq_bands()
        self.filterbank = ButterFilterBank(
            order=self.order, freq=freq_bands, rectify=self.rectify, axis=self.axis
        )

    def __call__(self, data):
        return self.filterbank(data)


@dataclass
class MelButterFilterBank(LinearButterFilterBank):
    """
    Butter filter bank with frequencies along the mel scale

    Parameters
    ----------

    order:
        Order of filter to be used
    low_freq:
        Lower/cutoff frequency the filter (float or (float, float))
    sampling_freq:
        Sampling frequency of the signal, also serves as higher frequency of the filter bank.
    analog:
        True if analog filter
    rectify:
        If true, the output is the absolute value of the filtered output
    axis:
        Axis along which the filter needs to be applied

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter for more details on parameters.
    """

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

        freq_bands = np.array([freqs, freqs * (1 + filter_bandwidth)]) / nyquist

        low_freq = self.hz2mel(self.low_freq)
        high_freq = self.hz2mel(self.sampling_freq / 2 / (1 + filter_bandwidth) - 1)
        freqs = self.mel2hz(np.linspace(low_freq, high_freq, self.num_filters))

        return np.array([freqs, freqs * (1 + filter_bandwidth)]).T / nyquist


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

    def get_noise_sample(self, sample_len: int) -> np.ndarray:
        """Get a random noise sample from the dataset"""
        # Find noise sample of minimum length
        while True:
            noise_idx = np.random.randint(0, len(self.dataset), (1,)).item()
            noise = self.dataset[noise_idx][0]
            if noise.shape[1] >= sample_len:
                break
        # Sample a random part of the data recording
        noise_signal_len = noise.shape[1]
        if noise_signal_len > sample_len:
            start_t = np.random.randint(0, noise_signal_len - sample_len, (1,)).item()
            noise = noise[:, start_t : start_t + sample_len]
        return noise

    def __call__(self, signal):
        # randomly pick a piece of noise data
        noise = self.get_noise_sample(sample_len=signal.shape[1])

        # mix signal with noise with given SNR
        signal_power = (signal**2).mean()
        noise_power = (noise**2).mean()
        noise_scale = (signal_power / noise_power) * 10 ** (-self.snr / 10)
        signal_with_snr = signal + noise_scale * noise

        # Normalize if specified
        if self.normed:
            return normalize(signal_with_snr)
        else:
            return signal_with_snr


def normalize(signal):
    """Normalize the signal"""
    signal -= signal.mean()
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val
    return signal


# @dataclass
# class DivisiveNormalization:
#    frame_dt: float  # Frame clock step
#    num_frames_avg: int  # Number of frames to average over
#    gating_clock_dt: float  # Clock frequency of gating E(t)
#    dt: float = 1  # Global clock step
#
#    def __call__(self, events: np.ndarray):
#        raise NotImplementedError

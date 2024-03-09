import numpy as np
import pytest


def test_standardize_data_length():
    from tonic.audio_transforms import FixLength

    sdl = FixLength(100, 1)

    # Data is longer
    data = np.ones((1, 120))
    assert sdl(data).shape == (1, 100)

    # Data is shorter
    data = np.ones((1, 80))
    assert sdl(data).shape == (1, 100)


def test_bin():
    import numpy as np

    from tonic.audio_transforms import Bin

    bin_transform = Bin(orig_freq=16_000, new_freq=100, axis=1)

    data = np.random.random((1, 8 * 16_000))
    data_binned = bin_transform(data)

    assert data_binned.shape == (1, 8 * 100)
    assert data.sum() == pytest.approx(data_binned.sum(), 1e-7)


def test_linear_butter_filter_bank():
    import numpy as np

    from tonic.audio_transforms import LinearButterFilterBank

    fb = LinearButterFilterBank(
        order=2, low_freq=100, sampling_freq=16000, num_filters=16
    )
    data = np.random.random((1, 16_000))

    filter_out = fb(data)
    assert filter_out.shape == (16, 16_000)


def test_mel_butter_filter_bank():
    import numpy as np

    from tonic.audio_transforms import MelButterFilterBank

    fb = MelButterFilterBank(order=2, low_freq=100, sampling_freq=16000, num_filters=16)
    data = np.random.random((1, 16_000))

    filter_out = fb(data)
    assert filter_out.shape == (16, 16_000)


def test_add_noise():
    import numpy as np

    class DummyNoiseDataset:
        def __len__(self):
            return 1000

        def __getitem__(self, item):
            sig_len = np.random.randint(12000, 20000, (1,)).item()
            return np.random.random((1, sig_len)), 0

    data = np.sin(np.arange(0, 16_000 * 0.001, 0.001))[None, ...]
    print(data.shape)

    from tonic.audio_transforms import AddNoise

    noise_dataset = DummyNoiseDataset()

    print(noise_dataset[0][0].shape)

    add_noise = AddNoise(noise_dataset, 10, normed=True)

    signal = add_noise(data)
    assert signal.shape == (1, 16_000)


def test_swap_axes():
    """Tests SwapAxes transform with synthetic data."""
    from tonic.audio_transforms import SwapAxes

    np.random.seed(123)
    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)
    ax1, ax2 = 0, 1
    swap_ax = SwapAxes(ax1=ax1, ax2=ax2)
    swaped = swap_ax(data)

    assert swaped.shape[0] == data.shape[1]
    assert swaped.shape[1] == data.shape[0]


def test_amplitude_scale():
    """Tests the amplitude scaling transform with synthetic data."""
    from tonic.audio_transforms import AmplitudeScale

    np.random.seed(123)
    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)
    max_amps = np.random.rand(10)

    for amp in max_amps:
        AmpScale = AmplitudeScale(max_amplitude=amp)
        transformed = AmpScale(data)
        assert data.shape[1] == transformed.shape[1]
        assert transformed.max() == amp


def test_robust_amplitude_scale():
    """Tests robust amplitude scaling transform with a synthetic data."""
    from tonic.audio_transforms import RobustAmplitudeScale

    np.random.seed(123)
    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)
    max_amps = np.random.rand(10)
    percent = 0.01
    for amp in max_amps:
        RobustAmpScale = RobustAmplitudeScale(
            max_robust_amplitude=amp, outlier_percent=percent
        )
        transformed = RobustAmpScale(data)
        sorted_transformed = np.sort(np.abs(transformed.ravel()))
        non_outlier = sorted_transformed[
            0 : int(np.floor(len(sorted_transformed)) * (1 - percent))
        ]
        print(non_outlier)
        assert data.shape[1] == transformed.shape[1]
        assert np.all(non_outlier <= amp)

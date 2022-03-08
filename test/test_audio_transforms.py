import pytest
import numpy as np


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
    from tonic.audio_transforms import LinearButterFilterBank
    import numpy as np

    fb = LinearButterFilterBank(
        order=2, low_freq=100, sampling_freq=16000, num_filters=16
    )
    data = np.random.random((1, 16_000))

    filter_out = fb(data)
    assert filter_out.shape == (16, 16_000)


def test_mel_butter_filter_bank():
    from tonic.audio_transforms import MelButterFilterBank
    import numpy as np

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

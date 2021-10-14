import torch


def test_standardize_data_length():
    import torch
    from tonic.audio_transforms import FixLength
    sdl = FixLength(100, 1)

    # Data is longer
    data = torch.ones((1, 120))
    assert sdl(data).shape == (1, 100)

    # Data is shorter
    data = torch.ones((1, 80))
    assert sdl(data).shape == (1, 100)


def test_bin():
    from tonic.audio_transforms import Bin

    bin_transform = Bin(orig_freq=16_000, new_freq=100, axis=1)

    data = torch.rand((1, 8 * 16_000))
    data_binned = bin_transform(data)

    assert data_binned.shape == (1, 8 * 100)
    assert data.sum() == data_binned.sum()


def test_linear_butter_filter_bank():
    from tonic.audio_transforms import LinearButterFilterBank
    import torch

    fb = LinearButterFilterBank(order=2, low_freq=100, sampling_freq=16000, num_filters=16, clamp=False)
    data = torch.rand((1, 16_000)).double()

    filter_out = fb(data)
    assert filter_out.shape == (16, 16_000)


def test_mel_butter_filter_bank():
    from tonic.audio_transforms import MelButterFilterBank
    import torch

    fb = MelButterFilterBank(order=2, low_freq=100, sampling_freq=16000, num_filters=16, clamp=False)
    data = torch.rand((1, 16_000)).double()

    filter_out = fb(data)
    assert filter_out.shape == (16, 16_000)

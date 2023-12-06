import numpy as np
import pytest


def test_random_time_stretch():
    """Tests the time_stretch transform with synthetic data for 2 scenarions: slowing down and
    speeding up.

    - verifies if the output of transform is different than the input data
    - verifies if the length of signal has chanched according to the stretch factor (and it should remain fixed if fix_length flag is True)
    """
    from tonic.audio_augmentations import RandomTimeStretch

    np.random.seed(123)

    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)

    for fix_length in [False, True]:
        # verify length of stretched signal
        slowing_down = RandomTimeStretch(
            samplerate=sr, sample_length=sl, factors=[0.5], fix_length=fix_length
        )
        slow = slowing_down(data)

        assert slow is not data

        if fix_length:
            assert slow.shape[1] == data.shape[1]
        else:
            assert np.allclose(
                slow.shape[1],
                data.shape[1] / (slowing_down.factors[0]),
                rtol=1e-2,
                atol=1e-3,
            )

        speeding_up = RandomTimeStretch(
            samplerate=sr, sample_length=sl, factors=[1.5], fix_length=fix_length
        )
        fast = speeding_up(data)

        assert fast is not data

        if fix_length:
            assert fast.shape[1] == data.shape[1]
        else:
            assert np.allclose(
                fast.shape[1],
                data.shape[1] / (speeding_up.factors[0]),
                rtol=1e-2,
                atol=1e-3,
            )


def test_random_pitch_shift():
    """Tests the pitch_shift transform with synthetic data.

    - verifies if the output of transform is different than the input data
    - verifies that the size has not changed
    """
    from tonic.audio_augmentations import RandomPitchShift

    np.random.seed(123)

    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)

    aug = RandomPitchShift(samplerate=sr)
    pitch_shifted = aug(data)

    assert pitch_shifted is not data

    assert pitch_shifted.shape[1] == data.shape[1]


def test_random_amplitude_scale():
    """Tests the amplitude_scale transform with synthetic data.

    - verifies if the output of transform is different than the input data
    - verifies that the size has not changed
    - verifies that maximum amplitude is in the defined range
    """
    from tonic.audio_augmentations import RandomAmplitudeScale

    np.random.seed(123)

    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.ones((1, sr * sl))
    min_amp, max_amp = 0.05, 0.15

    aug = RandomAmplitudeScale(samplerate=sr, min_amp=min_amp, max_amp=max_amp)
    amp_scaled = aug(data)

    assert amp_scaled is not data
    assert amp_scaled.shape[1] == data.shape[1]
    assert amp_scaled.max() <= max_amp


def test_add_white_noise():
    """Tests the add_white_noise transform with synthetic data.

    - verifies if the output of transform is different than the input data
    - verifies that the size has not changed
    """
    from tonic.audio_augmentations import AddWhiteNoise

    np.random.seed(123)

    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl)

    aug = AddWhiteNoise(samplerate=sr)
    noisy = aug(data)
    assert noisy is not data
    assert noisy.shape[1] == data.shape[1]


def test_RIR():
    """Tests the RIR transform with a synthetic data.

    - verifies if the output of transform is different than the input data
    - verifies that the size has not changed
    """
    from tonic.audio_augmentations import RIR

    np.random.seed(123)

    sr = 16_000  # sample rate
    sl = 1  # sample length
    data = np.random.rand(1, sr * sl).astype("float32")
    rir_audio_path = (
        "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav"
    )
    aug = RIR(samplerate=sr, rir_audio=rir_audio_path)
    RIR_augmented = aug(data)
    assert RIR_augmented is not data
    assert RIR_augmented.shape[1] == data.shape[1]

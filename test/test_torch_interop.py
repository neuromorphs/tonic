"""Tests for NumPy <-> PyTorch interoperability.

PyTorch builds compiled against the NumPy 1.x C API fail at runtime under NumPy 2
("Numpy is not available" / "_ARRAY_API not found"). These tests exercise the
code paths in tonic that cross the NumPy/PyTorch boundary.
"""

import numpy as np
import pytest
from utils import create_random_input

import tonic
import tonic.transforms as transforms

torch = pytest.importorskip("torch")


def test_torch_numpy_bridge():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = torch.from_numpy(array)
    assert tensor.shape == (3, 4)
    np.testing.assert_array_equal(tensor.numpy(), array)
    np.testing.assert_array_equal(torch.as_tensor(array).numpy(), array)
    np.testing.assert_array_equal(np.asarray(torch.ones(3)), np.ones(3))


def test_sparse_tensor_matches_frames():
    sensor_size = (40, 30, 2)
    events, sensor_size = create_random_input(sensor_size=sensor_size)

    sparse = transforms.ToSparseTensor(sensor_size=sensor_size, n_time_bins=5)(events)
    frames = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=5)(events)

    assert sparse.is_sparse
    np.testing.assert_array_equal(sparse.to_dense().numpy(), frames)


def test_dataloader_pad_tensors_with_numpy_samples():
    sensor_size = (40, 30, 2)
    lengths = [3, 5, 4]

    class FrameDataset:
        def __len__(self):
            return len(lengths)

        def __getitem__(self, index):
            frames = np.full(
                (lengths[index], *sensor_size[::-1]), index, dtype=np.int16
            )
            return frames, np.int64(index)

    dataloader = torch.utils.data.DataLoader(
        FrameDataset(),
        batch_size=len(lengths),
        collate_fn=tonic.collation.PadTensors(batch_first=True),
    )
    frames, targets = next(iter(dataloader))

    assert frames.shape == (len(lengths), max(lengths), *sensor_size[::-1])
    assert targets.tolist() == list(range(len(lengths)))
    for index, length in enumerate(lengths):
        assert (frames[index, :length] == index).all()
        assert (frames[index, length:] == 0).all()


def test_rir_from_local_file(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    from tonic.audio_augmentations import RIR

    rng = np.random.default_rng(123)
    rir_sr = 8_000
    rir_raw = rng.uniform(-1, 1, size=int(1.5 * rir_sr)).astype(np.float32)
    rir_path = tmp_path / "rir.wav"
    soundfile.write(rir_path, rir_raw, rir_sr, subtype="FLOAT")

    audio = rng.random((1, 16_000), dtype=np.float32)
    augmented = RIR(samplerate=16_000, rir_audio=str(rir_path))(audio)

    rir = rir_raw[int(rir_sr * 1.01) : int(rir_sr * 1.3)]
    rir = rir / np.linalg.norm(rir)
    expected = np.convolve(audio[0], rir)[: audio.shape[1]]

    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == audio.shape
    np.testing.assert_allclose(augmented[0], expected, rtol=1e-4, atol=1e-4)

import os
import tempfile

import numpy as np

from tonic.audio_augmentations import RandomPitchShift
from tonic.audio_transforms import AmplitudeScale, FixLength
from tonic.cached_dataset import Aug_DiskCachedDataset, load_from_disk_cache


class mini_dataset:
    def __init__(self) -> None:
        np.random.seed(0)
        self.data = np.random.rand(10, 16000)
        self.transform = None
        self.target_transform = None

    def __getitem__(self, index):
        sample = self.data[index]
        label = 1
        if sample.ndim == 1:
            sample = sample[None, ...]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label


def test_aug_disk_caching():
    from torchvision.transforms import Compose

    with tempfile.TemporaryDirectory() as cache_dir:
        cache_path = os.path.join(cache_dir, "cache")
        all_transforms = {}
        all_transforms["pre_aug"] = [AmplitudeScale(max_amplitude=0.150)]
        all_transforms["augmentations"] = [
            RandomPitchShift(samplerate=16000, caching=True)
        ]
        all_transforms["post_aug"] = [FixLength(16000)]
        # number of copies is set to number of augmentation params (factors)
        n = len(RandomPitchShift(samplerate=16000, caching=True).factors)
        Aug_cach = Aug_DiskCachedDataset(
            dataset=mini_dataset(),
            cache_path=cache_path,
            all_transforms=all_transforms,
            num_copies=n,
        )

        sample_index = 0
        Aug_cach.generate_all(sample_index)

        for i in range(n):
            transform = Compose(
                [
                    AmplitudeScale(max_amplitude=0.150),
                    RandomPitchShift(samplerate=16000, caching=True, aug_index=i),
                    FixLength(16000),
                ]
            )
            ds = mini_dataset()
            ds.transform = transform
            augmented_sample = ds[sample_index][0]
            loaded_sample, targets = load_from_disk_cache(
                os.path.join(cache_path, f"0_{i}.hdf5")
            )
            assert np.allclose(augmented_sample, loaded_sample, rtol=1e-5, atol=1e-5)

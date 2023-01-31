import tonic.prototype.datasets as datasets


def test_prophesee_minidataset():
    dp = datasets.MiniDataset("/home/gregorlenz/datasets")

    data, target = next(iter(dp))

    assert len(data) == 386431662

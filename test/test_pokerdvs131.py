import sys
import types

import pytest

from tonic.datasets import POKERDVS131


class _FakeEvent:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.x = 1
        self.y = 2
        self.polarity = 1


class _FakeLegacyAedatFile:
    """Stands in for dv.LegacyAedatFile: two bursts of events per recording,
    separated by a pause longer than POKERDVS131.idle_threshold_us."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _read_event_v2(self):
        for t in [0, 10, 20, 50_000, 50_010]:
            yield _FakeEvent(t)


def _create_recordings(save_to):
    folder = save_to / "POKERDVS131"
    folder.mkdir()
    for file in POKERDVS131.files:
        (folder / file).touch()


def test_pokerdvs131_segments_recordings(tmp_path, monkeypatch):
    fake_dv = types.SimpleNamespace(LegacyAedatFile=_FakeLegacyAedatFile)
    monkeypatch.setitem(sys.modules, "dv", fake_dv)
    _create_recordings(tmp_path)

    dataset = POKERDVS131(save_to=str(tmp_path))

    # 2 samples per recording, labelled cl, he, ... in recording order
    assert len(dataset) == 6
    assert dataset.targets == [0, 1, 0, 1, 0, 1]
    events, target = dataset[1]
    assert target == 1
    assert list(events["t"]) == [50_000, 50_010]


def test_pokerdvs131_requires_dv(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "dv", None)

    with pytest.raises(ImportError, match="pip install dv"):
        POKERDVS131(save_to=str(tmp_path))

from tonic.download_utils import _normalize_download_url


def test_normalize_figshare_download_url():
    url = "https://figshare.com/ndownloader/files/38022171"
    assert (
        _normalize_download_url(url)
        == "https://ndownloader.figshare.com/files/38022171"
    )

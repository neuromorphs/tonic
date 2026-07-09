from unittest.mock import patch

from tonic.download_utils import download_url


def test_figshare_url_rewritten_before_download(tmp_path):
    figshare_url = "https://figshare.com/ndownloader/files/38022171"
    with (
        patch(
            "tonic.download_utils._get_redirect_url", side_effect=lambda u, **kw: u
        ) as redirect,
        patch("tonic.download_utils._urlretrieve"),
        patch("tonic.download_utils.check_integrity", side_effect=[False, True]),
        patch("tonic.download_utils._get_google_drive_file_id", return_value=None),
    ):
        download_url(figshare_url, str(tmp_path), filename="test.bin")
    assert "ndownloader.figshare.com" in redirect.call_args[0][0]

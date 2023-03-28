# YOLOv5 Reproduction 🚀 by GuoQuanhao, GPL-3.0 license
"""
Download utils
"""

import logging
import os
import subprocess
import urllib
from pathlib import Path
from tqdm import tqdm
from urllib.request import urlopen, Request
import tempfile
import requests
import shutil
import paddle


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    output = subprocess.check_output(['gsutil', 'du', url], shell=True, encoding='utf-8')
    if output:
        return int(output.split()[0])
    return 0


def url_getsize(url='http://182.61.54.236/yolov5/bus.jpg'):
    # Return downloadable file size in bytes
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    req = Request(url, headers={"User-Agent": "paddle.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a url to a filename using curl.
    """
    silent_option = 'sS' if silent else ''  # silent
    proc = subprocess.run([
        'curl',
        '-#',
        f'-{silent_option}L',
        url,
        '--output',
        filename,
        '--retry',
        '9',
        '-C',
        '-',])
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f'ERROR: {assert_msg}\n{error_msg}')
        LOGGER.info('')


def attempt_download(file, url='http://182.61.54.236/yolov5/'):
    # Download weights from Baidu Cloud
    from utils.general import LOGGER

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        url = url + str(file)
        safe_download(file=str(file), url=url, min_bytes=1E5)
        return str(file)
    return str(file)

"""Restore selected retired DVC objects from pinned Kaggle notebook outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


def md5_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def restore(session, url, target, row):
    expected_size = row.get('size')
    if target.exists():
        if (expected_size and target.stat().st_size != expected_size) or md5_file(target) != row['md5']:
            raise ValueError('Existing cache object differs; retained: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.kaggle-restore.part')
    with session.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with part.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    if (expected_size and part.stat().st_size != expected_size) or md5_file(part) != row['md5']:
        part.unlink()
        raise ValueError('Kaggle size/MD5 mismatch: ' + row['md5'])
    os.replace(part, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--md5', nargs='+', required=True)
    parser.add_argument('--cache', type=Path, default=Path('.dvc/cache/files/md5'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    index = json.loads((repo / 'KAGGLE_OUTPUT_INDEX.json').read_text())
    objects = {row['md5']: row for row in index['objects']}
    rows = [objects[value] for value in args.md5]
    if any(not row.get('recovery_verified') for row in rows):
        parser.error('Selected output has no successful Kaggle recovery check')
    owners = {row['kernel'].split('/')[0] for row in rows}
    if len(owners) != 1:
        parser.error('Run separately through kaggle_as.py for each kernel owner')
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    if not os.environ.get('KAGGLE_API_TOKEN'):
        parser.error('Run through account-scoped kaggle_as.py')
    listings = {}
    with KaggleClient() as client, requests.Session() as session:
        for row in rows:
            key = row['kernel'], row.get('version')
            if key not in listings:
                owner, slug = row['kernel'].split('/')
                version = row.get('version')
                if version is None:
                    metadata = ApiGetKernelRequest()
                    metadata.user_name, metadata.kernel_slug = owner, slug
                    version = client.kernels.kernels_api_client.get_kernel(metadata).metadata.current_version_number
                    if not version:
                        raise ValueError('Notebook version unavailable: ' + row['kernel'])
                files, token = {}, None
                while True:
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    request.version_label, request.page_size = f"v{version}", 200
                    if token:
                        request.page_token = token
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                    files.update({entry.file_name: entry.url for entry in response.files or []})
                    token = response.next_page_token
                    if not token:
                        break
                listings[key] = files
            target = args.cache / row['md5'][:2] / row['md5'][2:]
            restore(session, listings[key][row['file']], target, row)
            print('MD5 verified:', row['md5'], str(target), flush=True)


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        raise SystemExit(type(error).__name__) from None

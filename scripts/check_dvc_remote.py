"""Probe configured DVC push/pull with a tiny new object; never touch project manifests.

Run from the checkout with the nckh interpreter. --cloud-root verifies the object through
rclone directly, independently of its mounted VFS cache. The tiny probe object is retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-root', required=True, help='rclone remote:path to DVC storage')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dvc = str(Path(sys.executable).with_name('dvc'))

    def run(argv, cwd=ROOT):
        return subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()

    remote = run([dvc, 'remote', 'default'])
    url = Path(run([dvc, 'config', f'remote.{remote}.url']))
    mount = Path.home() / 'Cloud/OneDrive'
    source = run(['findmnt', '-n', '-o', 'SOURCE,FSTYPE', '--mountpoint', str(mount)])
    if not source.startswith('onedrive:') or 'fuse.rclone' not in source or not url.is_relative_to(mount):
        raise SystemExit('Expected mounted onedrive: filesystem; refusing to write to an unmounted folder')
    expected_cloud = 'onedrive:' + url.relative_to(mount).as_posix()
    if args.cloud_root.rstrip('/') != expected_cloud:
        raise SystemExit('--cloud-root does not match the configured mounted remote')
    if not url.is_dir():
        raise SystemExit('DVC storage directory is unavailable')

    payload = b'MobiHet DVC push/pull probe\n' + os.urandom(64)
    digest = hashlib.md5(payload).hexdigest()
    obj = f'files/md5/{digest[:2]}/{digest[2:]}'
    with tempfile.TemporaryDirectory(prefix='mobihet-dvc-probe-') as tmp:
        work = Path(tmp)
        run([dvc, 'init', '--no-scm'], work)
        run([dvc, 'remote', 'add', '-d', 'probe', str(url)], work)
        (work / 'probe.bin').write_bytes(payload)
        run([dvc, 'add', 'probe.bin'], work)
        pushed = run([dvc, 'push', 'probe.bin.dvc'], work)
        cloud = args.cloud_root.rstrip('/') + '/' + obj
        for attempt in range(12):
            result = subprocess.run(['rclone', 'cat', cloud], capture_output=True, timeout=30)
            if result.returncode == 0 and result.stdout == payload:
                break
            if attempt == 11:
                raise SystemExit('Mount push completed; direct cloud verification did not pass within probe budget')
            time.sleep(5)
        (work / 'probe.bin').unlink()
        shutil.rmtree(work / '.dvc/cache')
        pulled = run([dvc, 'pull', 'probe.bin.dvc'], work)
        assert (work / 'probe.bin').read_bytes() == payload, 'DVC pull content mismatch'
    proof = dict(checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), remote=remote, url=str(url),
                 mount_source=source, cloud_root=args.cloud_root, probe_md5=digest,
                 probe_bytes=len(payload), push=pushed, cloud_content_verified=True,
                 pull_after_empty_cache=pulled, restored_content_verified=True,
                 scope='One new tiny object; project manifests and existing objects unchanged. Probe object retained.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()

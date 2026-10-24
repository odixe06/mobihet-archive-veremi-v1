"""Upload a flat stage directory (checkpoint/parent/source files + dataset-metadata.json) as a PRIVATE dataset of its owner,
wait until ready, then verify every file name over all listing pages (sizes and SHA are checked by the notebook at runtime).

    python upload_stage.py <stage dir> [<owner/slug>]   # slug from <stage>/dataset-metadata.json unless given
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HELPER = Path(__file__).resolve().parents[5] / '.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py'


def main():
    stage = Path(sys.argv[1]); meta = stage / 'dataset-metadata.json'
    dataset = sys.argv[2] if len(sys.argv) > 2 else json.loads(meta.read_text())['id']
    owner = dataset.split('/')[0]
    files = {p.name: p.stat().st_size for p in stage.iterdir() if p.is_file() and p.name != 'dataset-metadata.json'}
    assert files and not any(p.is_dir() for p in stage.iterdir()), 'flat stage expected'
    env = dict(os.environ, PATH=f'{Path(sys.executable).parent}{os.pathsep}{os.environ.get("PATH", "")}')   # kaggle CLI of this interpreter's env
    kg = lambda *args: subprocess.run([sys.executable, str(HELPER), owner, '--', 'kaggle', *args], capture_output=True, text=True, env=env)
    with tempfile.TemporaryDirectory() as tmp:   # per-owner metadata without touching the shared stage
        work = Path(tmp)
        for name in files:
            try: (work / name).symlink_to((stage / name).resolve())
            except OSError: shutil.copyfile(stage / name, work / name)   # Windows without Developer Mode
        (work / 'dataset-metadata.json').write_text(json.dumps(dict(id=dataset, title=dataset.split('/')[1], licenses=[dict(name='CC0-1.0')])))
        if 'ready' not in kg('datasets', 'status', dataset).stdout:
            created = kg('datasets', 'create', '-p', str(work), '-q', '-t')
            print(created.stdout[-300:], created.stderr[-300:], flush=True)
            assert created.returncode == 0, 'dataset create failed'
    for _ in range(120):
        status = kg('datasets', 'status', dataset)
        if 'ready' in status.stdout: break
        time.sleep(15)
    assert 'ready' in status.stdout, status.stdout + status.stderr
    seen, page = {}, None
    while True:   # CLI pages at 200 entries
        args = ['datasets', 'files', dataset, '--csv', '--page-size', '200'] + (['--page-token', page] if page else [])
        out = kg(*args).stdout.splitlines()
        rows = [l.split(',') for l in out if l and not l.startswith(('name,', 'Next Page Token'))]
        seen.update({r[0]: r[1] for r in rows})
        token = [l.split('=', 1)[-1].strip() for l in out if l.startswith('Next Page Token')]
        if not token: break
        page = token[0]
    missing = sorted(set(files) - set(seen))
    assert not missing, ('missing files', missing[:5])
    print('VERIFIED', dataset, len(files), 'files', sum(files.values()), 'bytes', flush=True)


if __name__ == '__main__':
    main()

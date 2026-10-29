"""Merge a directory manifest (.dvc with path: artifacts/<name>) with the local workspace and push new objects to the
filesystem DVC remote without `dvc add` (artifacts/ is git-ignored). Old remote entries are kept; local files override the
same relpath. Objects are copied tmp -> MD5 verify -> rename. The .dvc file is updated last.

    python scripts/dvc_merge_push.py <build>/<name>.dvc [...]
"""
import configparser, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_config = configparser.ConfigParser()
_config.read(ROOT / '.dvc/config')
REMOTE = Path(_config[f'\'remote "{_config["core"]["remote"]}"\'']['url']) / 'files/md5'
CACHE = ROOT / '.dvc/cache/files/md5'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def obj(root, h): return root / h[:2] / h[2:]


def put(src, h):
    dst = obj(REMOTE, h)
    if dst.exists(): return 0
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + '.tmp')
    shutil.copyfile(src, tmp); assert md5(tmp) == h, src; os.replace(tmp, dst); return src.stat().st_size


def merge(dvcfile):
    dvcfile = ROOT / dvcfile; text = dvcfile.read_text()
    old_md5 = next(l.split(':', 1)[1].strip() for l in text.splitlines() if l.strip().startswith('- md5:'))
    rel = next(l.split(':', 1)[1].strip() for l in text.splitlines() if l.strip().startswith('path:'))
    src = obj(REMOTE, old_md5) if obj(REMOTE, old_md5).exists() else obj(CACHE, old_md5)
    entries = {e['relpath']: e['md5'] for e in json.loads(src.read_bytes())}
    work = dvcfile.parent / rel; pushed = n = 0
    for p in sorted(work.rglob('*')):
        if not p.is_file() or p.is_symlink() or p.name.endswith(('.tmp', '.part', '.partial')) or 'private_push' in p.parts: continue
        r = p.relative_to(work).as_posix(); h = md5(p); entries[r] = h; pushed += put(p, h); n += 1
    listing = [{'md5': h, 'relpath': r} for r, h in sorted(entries.items())]
    blob = json.dumps(listing, sort_keys=True).encode(); dh = hashlib.md5(blob).hexdigest() + '.dir'
    for root in (REMOTE, CACHE):
        o = obj(root, dh); o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists(): o.write_bytes(blob)
    missing = [r for r, h in entries.items() if not obj(REMOTE, h).exists()]
    assert not missing, ('objects absent on remote', missing[:5])
    size = sum(obj(REMOTE, h).stat().st_size for h in set(entries.values()))
    lines = []
    for l in text.splitlines():
        s = l.strip()
        if s.startswith('- md5:'): l = l.split('md5:')[0] + 'md5: ' + dh
        elif s.startswith('size:'): l = l.split('size:')[0] + f'size: {size}'
        elif s.startswith('nfiles:'): l = l.split('nfiles:')[0] + f'nfiles: {len(entries)}'
        lines.append(l)
    dvcfile.write_text('\n'.join(lines) + '\n')
    print(f'{dvcfile.relative_to(ROOT)}: {old_md5} -> {dh}; entries {len(entries)} (local {n}); pushed {pushed / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    mount = Path.home() / 'Cloud/OneDrive'
    if REMOTE.is_relative_to(mount) and not os.path.ismount(mount):
        raise SystemExit('OneDrive rclone mount unavailable; refusing local fallback writes')
    for f in sys.argv[1:]: merge(f)

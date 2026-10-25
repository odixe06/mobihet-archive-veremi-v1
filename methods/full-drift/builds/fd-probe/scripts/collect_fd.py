"""One-shot terminal collection + audit of FullDrift sessions (no polling loop).

Audit: executed notebook source == pushed (key redacted); runtime input manifest/fd.py SHA; data manifest = full holdout
(source/test rows, group overlap 0); per arm config fingerprint, marker chain, epoch-6 weights SHA + strict DAGSNet
load + finite; every predictions npz SHA and recomputed per-client confusion -> ten metrics == marker eval.
The shared proxy is stored as one prediction row (one model after aggregation) and scored for every client.

    python methods/full-drift/builds/fd-probe/scripts/collect_fd.py [--only 'fd-s42g-*'] [--jobs 16]
    # several terminal sessions: one process each, e.g. 2 sessions --jobs 16, 3-4 sessions --jobs 8 (~32 connections total)
"""
import argparse
import ast
try:
    import fcntl
except ImportError:  # Windows: current collector is intentionally run one session at a time.
    fcntl = None
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import nbformat
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT / '.compat'))
from mh.dagsnet import build_model, CFG
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion

HELPER = ROOT / '.agents/skills/kaggle-training-notebook/scripts'
KAGGLE_PYTHON = os.environ.get('KAGGLE_PYTHON', sys.executable)
PATTERN = (r'^(source/|runs/fd_summary\.json|runs/[^/]+/(complete/|config\.json|result\.json|metrics\.csv|log\.txt|worker[0-9]+\.log|'
           r'predictions/|personalized/|weights/round_0(06|15|50)\.pt))')   # last epoch of 6-, 15- and 50-epoch arms
FINAL_PATTERN = r'^(source/|runs/fd_summary\.json|runs/[^/]+/(complete/|config\.json|runtime\.json|result\.json|metrics\.csv|log\.txt|worker[0-9]+\.log|predictions/|weights/|resume/))'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_personalized(run, result, K):
    folder = run / 'personalized'
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert result['personalized'] == metrics
    assert metrics['method'] == 'Fed-FN' and metrics['epochs'] == 10 and metrics['max_rows'] == 30_000
    assert len(metrics['clients']) == K and [c['client'] for c in metrics['clients']] == list(range(K))
    assert all(c['epochs'] == 10 and 0 < c['rows'] <= 30_000 and c['applied'] > 0 and c['skipped'] <= 16
               for c in metrics['clients'])
    predictions = folder / 'predictions.npz'; weights = folder / 'weights.pt'
    assert sha(predictions) == metrics['predictions_sha'] and sha(weights) == metrics['weights_sha']
    arrays = np.load(predictions); checks = 0
    for hold, rules in metrics['eval'].items():
        y = arrays[hold + '_y'].astype(np.int64)
        for rule, ev in rules.items():
            pred = arrays[f'personalized_{hold}_{rule}'].astype(np.int64)
            assert len(pred) == K
            cms = [np.bincount(y * 16 + p, minlength=256).reshape(16, 16) for p in pred]
            assert all(np.array_equal(c, np.array(e)) for c, e in zip(cms, ev['confusion_clients']))
            mm = [metrics_from_confusion(c) for c in cms]
            assert set(ev['mean_metrics']) == set(METRIC_KEYS)
            for key, value in ev['mean_metrics'].items():
                assert abs(float(np.mean([m[key] for m in mm])) - value) < 1e-12, (run.name, hold, rule, key)
            pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, range(16))] for c in cms], axis=0)
            assert np.allclose(pc, ev['f1_per_class'], atol=1e-12, rtol=0)
            assert abs(float(pc[0]) - ev['benign_f1']) < 1e-12
            assert abs(ev['mean_metrics']['f1_macro'] - ev['f1_macro']) < 1e-12
            checks += 1
    saved = torch.load(weights, map_location='cpu', weights_only=True)
    assert len(saved['clients']) == K
    model = build_model(saved['architecture'], latent=True)
    assert sum(p.numel() for p in model.parameters()) == 395024
    for state in saved['clients']:
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    return dict(method=metrics['method'], clients=K, metric_checks=checks,
                level='personalized hashes; 20 weights strict/finite; predictions->confusion->10 metrics')


def audit_arm(run):
    cfg = json.loads((run / 'config.json').read_text())
    K = cfg['config'].get('clients', 20)
    assert hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != 'fingerprint'}, sort_keys=True).encode()).hexdigest() == cfg['fingerprint']
    rows = [json.loads(m.read_text()) for m in sorted((run / 'complete').glob('round_*.json'))]
    parent = cfg['config'].get('parent'); start = parent['round'] if parent else 0   # round 17 --parent: continues an audited run
    if parent:
        marker = next(json.loads((c.parent / 'complete' / f'round_{start:03d}.json').read_text())
                      for c in sorted((BASE / 'artifacts/kaggle').glob('*/runs/*/config.json'))
                      if json.loads(c.read_text())['fingerprint'] == parent['fingerprint'] and (c.parents[2] / 'AUDITED.json').exists())
        assert marker['weights_sha'] == parent['weights_sha'], ('parent weights', run.name)
    previous, checks, kept = parent['weights_sha'] if parent else None, 0, []
    for i, row in enumerate(rows):
        assert row['round'] == start + i + 1 and row['prev_sha'] == previous and row['fingerprint'] == cfg['fingerprint']
        previous = row['weights_sha']
        wp = run / 'weights' / f'round_{row["round"]:03d}.pt'
        if cfg['config'].get('final'): assert wp.exists(), ('final round missing weights', row['round'])
        if wp.exists():
            assert sha(wp) == row['weights_sha'], ('weights sha', run.name, row['round'])
            w = torch.load(wp, map_location='cpu', weights_only=True)
            assert w['round'] == row['round'] and w['fingerprint'] == cfg['fingerprint']
            if cfg['config'].get('final'):
                assert len(w['private']) == len(w['ema_private']) == K and len(w['proxy']) == 1
                assert w['architecture'] == dict(CFG, n_features=66, num_classes=16)
            architecture = w.get('architecture', dict(CFG, n_features=66, num_classes=16))
            model = build_model(architecture, latent=True)
            for sd in w['private'] + w['proxy'] + w['ema_private']:
                model.load_state_dict(sd, strict=True)
                if architecture.get('n_features', 66) == 66:
                    assert sum(p.numel() for p in model.parameters()) == 395024
                assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
            kept.append(row['round'])
        if 'resume_sha' in row and cfg['config'].get('final'):
            rp = run / 'resume' / f'round_{row["round"]:03d}.pt'
            assert sha(rp) == row['resume_sha']
            z = torch.load(rp, map_location='cpu', weights_only=True)
            assert z['round'] == row['round'] and z['fingerprint'] == cfg['fingerprint']
            assert all(torch.isfinite(v).all() for field in ('exact', 'ef') for v in z[field].values() if v.is_floating_point())
        if row.get('evaluation_pending'):
            assert cfg['config'].get('defer_eval') and row['eval'] == {}
            continue
        pf = run / 'predictions' / f'round_{row["round"]:03d}.npz'
        assert sha(pf) == row['predictions_sha'], ('predictions sha', run.name, row['round'])
        arrays = np.load(pf)
        for role, holds in row['eval'].items():
            for hold, rules in holds.items():
                y = arrays[hold + '_y'].astype(np.int64)
                for rule, ev in rules.items():
                    assert set(ev['mean_metrics']) == set(METRIC_KEYS), 'all ten metrics required'
                    pred = arrays[f'{role}_{hold}_{rule}'].astype(np.int64)
                    cms = [np.bincount(y * 16 + p, minlength=256).reshape(16, 16) for p in pred]
                    if len(cms) == 1: cms = cms * K          # shared proxy: one model, every client
                    assert len(cms) == len(ev['confusion_clients']) == K
                    assert all(np.array_equal(c, np.array(e)) for c, e in zip(cms, ev['confusion_clients']))
                    mm = [metrics_from_confusion(c) for c in cms]
                    for key, value in ev['mean_metrics'].items():
                        assert abs(float(np.mean([m[key] for m in mm])) - value) < 1e-12, (run.name, role, hold, rule, key)
                    checks += 1
    target = (50 if run.name.endswith('_e50') else 15 if run.name.endswith('_e15') else 6) - start
    result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {}
    personalized = audit_personalized(run, result, K) if result.get('personalized') is not None else None
    return dict(run=run.name, epochs=len(rows), complete=len(rows) == target, kept_weight_rounds=kept, metric_checks=checks, parent=parent,
                session_complete=result.get('session_complete', len(rows) == target),
                evaluation_pending=any(r.get('evaluation_pending') for r in rows),
                personalized=personalized,
                level='marker chain, pulled weights strict/finite; predictions->confusion->10 metrics when evaluated')


# pull connections: skill pull-outputs.md — throughput scales with open connections, ~32 total saturates the link
JOBS = '16'


PUSH_VERSION = {'fd-s42ma-20261015-r13-v1': 2}

def collect(rid, stage_tag='20261015'):
    folder = BASE / 'notebooks' / rid; dest = BASE / 'artifacts/kaggle' / rid; marker = dest / 'AUDITED.json'
    if not (folder / 'PUSHED').exists():
        return None
    if marker.exists():
        return json.loads(marker.read_text())
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    prefix = [KAGGLE_PYTHON, str(HELPER / 'kaggle_as.py'), meta['id'].split('/')[0], '--']
    status = subprocess.run(prefix + ['kaggle', 'kernels', 'status', meta['id']], check=True, capture_output=True, text=True)
    print(status.stdout.strip(), flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE', 'ERROR')):
        return None
    final = json.loads((folder / 'source_manifest.json').read_text()).get('queues', {})
    final = isinstance(final, dict) and final.get('final', False)
    subprocess.run(prefix + [KAGGLE_PYTHON, str(HELPER / 'pull_kernel_output.py'), meta['id'], '-p', str(dest), '-j', JOBS,
                             '--pattern', FINAL_PATTERN if final else PATTERN], check=True)
    private = BASE / 'private_push' / rid; executed = private / 'executed.ipynb'
    version = str(PUSH_VERSION.get(rid, 1))   # ma: v1 cancelled while queued (never ran), same notebook re-pushed as v2
    subprocess.run(prefix + [KAGGLE_PYTHON, str(HELPER / 'download_executed_notebook.py'), meta['id'], '--version', version,
                             '--output', str(executed)], check=True, umask=0o077)
    executed.chmod(0o600)
    pushed = nbformat.read(private / meta['code_file'], as_version=4)
    reference = nbformat.read(folder / meta['code_file'], as_version=4)
    keys = [n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            and any(isinstance(t, ast.Name) and t.id == '_INLINE_WANDB_KEY' for t in n.targets)]
    key = keys[0] if keys else None
    actual_text = executed.read_text()
    if key:
        actual_text = actual_text.replace(key, '[REDACTED]')
    actual = nbformat.reads(actual_text, as_version=4)
    assert len(actual.cells) == len(reference.cells)
    for i, c in enumerate(actual.cells):
        if c.cell_type == 'code':
            expected = pushed.cells[i].source.replace(key, '[REDACTED]') if key else pushed.cells[i].source
            assert c.source == expected, ('executed source drift', i)
    actual.cells[1].source = reference.cells[1].source; actual.metadata['credential_cell_redacted'] = True
    nbformat.write(actual, folder / 'executed.redacted.ipynb')
    stage = BASE / f'artifacts/input/fd-src-{stage_tag}'
    inputs = json.loads((stage / 'input_manifest.json').read_text())
    assert json.loads((dest / 'source/input_manifest.json').read_text()) == inputs
    assert sha(dest / 'source/fd.py') == inputs['fd.py'] == sha(stage / 'fd.py')   # the round's frozen copy, not the live source
    data = json.loads((dest / 'source/data_manifest.json').read_text())
    assert data['source_rows'] == 43045415 and data['test_rows'] == 10761343 and data['group_overlap'] == 0
    assert data['protocol'].startswith('final: every train80%' if final else 'temporal2% whole partition') and data['train_cap'] is None
    logs = ''.join(v.get('data', '') for v in json.loads((dest / f'mobihet-{rid}.log').read_text()))
    assert not key or key not in logs
    # pulls may run in parallel; the audit holds ~4GB (ep6 test predictions), so audits take one shared lock
    with (ROOT / '.compat/.artifact-verification.lock').open('a') as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        audits = [audit_arm(run) for run in sorted((dest / 'runs').glob('fd_*_s4[23]*'))
                  if (run / 'complete').exists() and list((run / 'complete').glob('round_*.json'))]
    summary = dest / 'runs/fd_summary.json'
    record = dict(kernel=meta['id'], version=1, inputs=sha(stage / 'input_manifest.json'), audits=audits,
                  train_rows=sum(sum(data['counts'][f'train_{k}']) for k in data['clients']),
                  summary=json.loads(summary.read_text()) if summary.exists() else None,
                  arms_complete=sorted(a['run'] for a in audits if a['complete']),
                  limit='Training audit only until detached evaluation is collected' if final else
                        'Single seed screening; weights kept for selected epochs only in the pull; test descriptive')
    marker.write_text(json.dumps(record, indent=2) + '\n')
    print('AUDITED fulldrift', rid, 'arms', len(audits), 'complete', len(record['arms_complete']), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='*', help='rid glob: one process per session runs pulls in parallel')
    parser.add_argument('--jobs', default=JOBS, help='pull connections for this process (split ~32 across processes)')
    args = parser.parse_args(); JOBS = args.jobs
    torch.set_num_threads(1)
    sessions = [(f'fd-s42{s}-20261015-v2', '20261015') for s in 'abc']   # v1: production gate TypeError at Run init
    sessions += [(f'fd-s42{s}-20261015-r2-v1', '20261015-r2') for s in 'def']
    sessions += [(f'fd-s42{s}-20261015-r3-v1', '20261015-r3') for s in 'gh']
    sessions += [('fd-s42i-20261015-r4-v1', '20261015-r4'), ('fd-s42j-20261015-r5-v1', '20261015-r5')]
    sessions += [(f'fd-s43{s}-20261015-r6-v1', '20261015-r6') for s in 'kl']
    sessions += [('fd-s43m-20261015-r7-v1', '20261015-r7')]
    sessions += [(f'fd-s42{s}-20261015-r8-v1', '20261015-r8') for s in 'nopq'] + [('fd-s43r-20261015-r8-v1', '20261015-r8')]
    sessions += [(f'fd-s42{s}-20261015-r9-v1', '20261015-r9') for s in 'stuvx'] + [('fd-s43w-20261015-r9-v1', '20261015-r9')]
    sessions += [('fd-s42y-20261015-r10-v1', '20261015-r10')] + [(f'fd-s43{s}-20261015-r10-v1', '20261015-r10') for s in ('z', 'ka', 'kb', 'ga', 'gb')]
    sessions += [(f'fd-s42{s}-20261015-r11-v1', '20261015-r11') for s in ('la', 'lc')]
    sessions += [('fd-s42qa-20261015-r12-v1', '20261015-r12')]
    sessions += [(f'fd-s42{s}-20261015-r13-v1', '20261015-r13') for s in ('ma', 'mb', 'mc', 'md', 'me', 'mf')]
    sessions += [('fd-s42na-20261015-r14-v1', '20261015-r14'), ('fd-s43nb-20261015-r14-v1', '20261015-r14'), ('fd-s42nc-20261015-r14-v1', '20261015-r14')]
    sessions += [(f'fd-s42{s}-20261015-r15-v1', '20261015-r15') for s in ('oa', 'ob', 'oc')]
    sessions += [(f'fd-s{seed}{s}-20261015-r16-v1', '20261015-r16') for seed, s in ((43, 'pa'), (43, 'pb'), (42, 'pc'), (43, 'pd'))]
    sessions += [(f'fd-s42{s}-20261015-r17-v1', '20261015-r17') for s in ('ra', 'rb', 'rc', 'rd')]
    sessions += [(f'fd-s42{s}-20261015-r18-v1', '20261015-r18') for s in ('sa', 'sb', 'sc', 'sd', 'se', 'sf')]
    sessions += [(f'fd-s{seed}{s}-20261015-r19-v1', '20261015-r19') for seed, s in ((42, 'ta'), (42, 'tb'), (42, 'tc'), (43, 'td'), (43, 'te'),
                                                                                   (42, 'ua'), (43, 'ub'), (42, 'uc'), (43, 'ud'))]
    sessions += [(f'fd-s42{s}-20261015-r20-v1', '20261015-r20') for s in ('va', 'vb', 'vc', 'vd', 've', 'wa', 'wb', 'wc')]
    sessions += [(f'fd-s{seed}{s}-20261015-r22-v1', '20261015-r22') for seed, s in ((42, 'xa'), (42, 'xb'), (42, 'xc'), (43, 'xd'))]
    sessions += [(f'fd-s{seed}{s}-20261015-r23-v1', '20261015-r23') for seed, s in ((43, 'ya'), (42, 'yb'))]
    sessions += [('fd-s43ya-20261015-r23-v2', '20261015-r23')]   # ya-v1 stopped by owner (slow host, prepare 9929s)
    sessions += [(p.name, tag) for tag in ('20261015-r24', '20261015-r25', '20261015-r28', '20261015-r30', '20261015-r31')
                 for p in sorted((BASE / 'notebooks').glob(f'fd-s4[23]*-{tag}-*')) if (p / 'source_manifest.json').exists()]
    sessions += [('fd-s42fa-20261015-r32-v1', '20261015-r32')]
    sessions += [(f'fd-s42{s}-20261015-r37-v1', '20261015-r37') for s in ('fc', 'fe')]
    sessions += [(f'fd-s42{s}-20261015-r37-v3', '20261015-r37') for s in ('fg', 'fh', 'pa', 'pb')]
    sessions += [(f'fd-s42{s}-20261015-r37-v4', '20261015-r37') for s in ('pc', 'ta', 'tc', 'ca', 'cb', 'cc')]
    sessions += [('fd-s42tb-20261015-r37-v5', '20261015-r37')]
    sessions += [(f'fd-s42{s}-20261015-r38-v1', '20261015-r38') for s in ('na', 'nb', 'nc')]
    if args.only != '*':
        selected = [rid for rid, tag in sessions if fnmatch.fnmatch(rid, args.only)]
        assert selected, ('unknown session filter', args.only)
        assert all((BASE / 'notebooks' / rid / 'PUSHED').exists() for rid in selected), 'selected session missing PUSHED; verify remote identity first'
    for rid, tag in sessions:
        if fnmatch.fnmatch(rid, args.only): collect(rid, tag)

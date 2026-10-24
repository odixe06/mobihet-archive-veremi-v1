"""Evaluate one committed round on both GPUs; never optimize or change its weights.

Training config/data identity and the marker SHA are checked before model reconstruction.
Each process scores disjoint clients, including recalibrated private EMA. Rank 0 scores
the shared proxy, merges predictions and writes the evaluation marker last.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from fd import Run, Method, deterministic, fixture_path
from mh import pilot as P
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion
from mh.new_fl import digest, save
from mh.v2_methods import calibrate


def checked_weights(run, r):
    cfg = json.loads((run / 'config.json').read_text())
    row = json.loads((run / 'complete' / f'round_{r:03d}.json').read_text())
    assert row['round'] == r and row['fingerprint'] == cfg['fingerprint'], 'round/config mismatch'
    path = run / 'weights' / f'round_{r:03d}.pt'
    assert digest(path) == row['weights_sha'], 'weights sha'
    w = torch.load(path, map_location='cpu', weights_only=True)
    assert w['round'] == r and w['fingerprint'] == cfg['fingerprint']
    expected = dict(P.CFG, n_features=53, num_classes=P.C, input_pad=1) if 'drop53' in cfg['config']['arm'].split('_') else \
               dict(P.CFG, n_features=P.D, num_classes=P.C)
    assert w['architecture'] == expected
    return cfg, row, w


def score_part(a, cfg, w):
    config = dict(cfg['config'])
    config.pop('parent', None)
    args = SimpleNamespace(**{**config, 'clients': config.get('clients', 20), 'workers': a.workers,
                              'rank': a.rank, 'evaluate_only': True, 'wandb_online': False})
    # Evaluation is deterministic even when training uses --fast.
    deterministic()
    R = Run(args)
    fx = fixture_path(args)
    assert digest(fx / 'manifest.json') == cfg['fixture'], 'evaluation fixture differs from training'
    m = Method(R, initialize=False)
    m.prior()  # same SecureSum prior arithmetic as inline evaluation; no Gaussian fitting
    assert len(w['private']) == len(w['ema_private']) == R.K and len(w['proxy']) == 1
    assert sum(p.numel() for p in R.init.parameters()) == 395024
    for model, state in zip(R.priv, w['private'], strict=True):
        model.load_state_dict(state, strict=True)
    for state in w['private'] + w['ema_private'] + w['proxy']:
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), 'nonfinite weights'
    arrays = {'test_y': R.H['test'][1].cpu().numpy().astype('uint8')}
    confusions = {}
    def put(role, k, model, rules):
        values = R.score(model, k, 'test', rules)
        for rule, (cm, pred) in values.items():
            key = f'{role}_{rule}_{k}'
            arrays[key] = pred
            confusions[key] = cm.tolist()
    for k in R.own:
        put('private', k, R.priv[k], R.rules)
        model = copy.deepcopy(R.priv[k])
        model.load_state_dict(w['ema_private'][k], strict=True)
        calibrate(model, R.mask(R.X[k][R.cal[k]].float()))
        put('private_ema', k, model, ('head', 'gp'))
        del model
        save(a.out / f'progress_rank{a.rank}.json', dict(client=k, clients_done=R.own.index(k) + 1, clients_total=len(R.own), timestamp=time.time()))
    if a.rank == 0:
        m.proxy[0].load_state_dict(w['proxy'][0], strict=True)
        put('proxy', 0, m.proxy[0], ('head',))
    part = a.out / 'parts' / f'round_{a.round:03d}_rank{a.rank}'
    part.parent.mkdir(parents=True, exist_ok=True)
    with part.with_suffix('.npz').open('wb') as f:
        np.savez_compressed(f, **arrays)
    save(part.with_suffix('.json'), dict(confusions=confusions, clients=R.own, rules=R.rules, K=R.K))


def merge(a, cfg, row):
    arrays, cms, clients = {}, {}, []
    for rank in range(a.workers):
        part = a.out / 'parts' / f'round_{a.round:03d}_rank{rank}'
        meta = json.loads(part.with_suffix('.json').read_text())
        clients.extend(meta['clients']); cms.update(meta['confusions'])
        with np.load(part.with_suffix('.npz')) as data:
            if 'test_y' in arrays: assert np.array_equal(arrays['test_y'], data['test_y'])
            arrays.update({key: data[key] for key in data.files})
    K = meta['K']
    assert sorted(clients) == list(range(K)), 'eval client coverage'
    predictions, evaluated = {'test_y': arrays['test_y']}, {}
    for role, rules in [('private', meta['rules']), ('private_ema', ('head', 'gp')), ('proxy', ('head',))]:
        result = {}
        for rule in rules:
            keys = [f'{role}_{rule}_{k}' for k in range(1 if role == 'proxy' else K)]
            cc = [np.array(cms[key], dtype=np.int64) for key in keys]
            for key, cm in zip(keys, cc, strict=True):
                actual = np.bincount(arrays['test_y'].astype(np.int64) * 16 + arrays[key], minlength=256).reshape(16, 16)
                assert np.array_equal(actual, cm) and cm.sum() == len(arrays['test_y'])
            if role == 'proxy': cc *= K
            mm = [metrics_from_confusion(c) for c in cc]
            pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cc], axis=0)
            result[rule] = dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
            predictions[f'{role}_test_{rule}'] = np.stack([arrays[key] for key in keys])
        evaluated[role] = {'test': result}
    pf = a.out / 'predictions' / f'round_{a.round:03d}.npz'
    pf.parent.mkdir(exist_ok=True)
    with pf.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f, **predictions)
    os.replace(pf.with_suffix('.tmp'), pf)
    cf = a.out / 'metrics' / f'round_{a.round:03d}.csv'; cf.parent.mkdir(exist_ok=True)
    with cf.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'role', 'holdout', 'rule', *METRIC_KEYS]); writer.writeheader()
        for role, holds in evaluated.items():
            for rule, v in holds['test'].items():
                writer.writerow(dict(round=a.round, role=role, holdout='test', rule=rule, **v['mean_metrics']))
    save(a.out / 'eval' / f'round_{a.round:03d}.json',
         dict(round=a.round, fingerprint=cfg['fingerprint'], weights_sha=row['weights_sha'],
              predictions_sha=digest(pf), eval=evaluated, eval_source_sha=digest(Path(__file__))))
    print('EVALUATED', a.round, cfg['fingerprint'], 'clients', K, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--round', type=int, required=True); p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    p.add_argument('--rank', type=int, default=0); p.add_argument('--part', action='store_true')
    p.add_argument('--wandb_online', action='store_true')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    assert not (a.out / 'eval' / f'round_{a.round:03d}.json').exists(), 'evaluation already committed'
    cfg, row, w = checked_weights(a.run, a.round)
    if a.part: return score_part(a, cfg, w)
    started = time.monotonic(); procs = []; wb = None
    if a.wandb_online:
        import wandb
        assert wandb.login(verify=True)
        wb = wandb.init(project='mobihet-fulldrift', id=f'eval-{a.out.name}-r{a.round:03d}-{cfg["fingerprint"][:8]}',
                        name=f'eval-{a.out.name}-r{a.round:03d}', config=dict(phase='evaluation', round=a.round, fingerprint=cfg['fingerprint']))
        save(a.out / 'wandb_run.json', dict(entity=wb.entity, project=wb.project, id=wb.id, url=wb.url))
        wb.log({'eval/started': 1})
    for rank in range(a.workers):
        log = (a.out / f'round_{a.round:03d}_worker{rank}.log').open('w')
        command = [sys.executable, str(Path(__file__)), '--run', str(a.run), '--out', str(a.out),
                   '--round', str(a.round), '--workers', str(a.workers), '--rank', str(rank), '--part']
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[rank % len(visible)])
        procs.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    while any(proc.poll() is None for proc, log in procs):
        if any(proc.poll() not in (None, 0) for proc, log in procs):
            for proc, log in procs:
                if proc.poll() is None: proc.terminate()
            raise RuntimeError('evaluation worker failed; inspect worker logs')
        if wb:
            progress = [json.loads(f.read_text()) for f in a.out.glob('progress_rank*.json')]
            wb.log({'eval/clients_done': sum(v['clients_done'] for v in progress), 'eval/seconds': time.monotonic() - started})
        time.sleep(5)
    for proc, log in procs:
        rc = proc.wait(); log.close()
        assert rc == 0, ('evaluation worker failed', rc)
    merge(a, cfg, row)
    if wb:
        result = json.loads((a.out / 'eval' / f'round_{a.round:03d}.json').read_text())
        wb.log({f'{role}/test/{rule}/{key}': value for role, holds in result['eval'].items()
                for rule, ev in holds['test'].items() for key, value in ev['mean_metrics'].items()})
        wb.summary['complete'] = True; wb.finish()
    print('evaluation_seconds', round(time.monotonic() - started, 1), flush=True)


if __name__ == '__main__': main()

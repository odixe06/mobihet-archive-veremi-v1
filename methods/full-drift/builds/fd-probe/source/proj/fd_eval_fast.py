"""Fast evaluation of committed rounds (owner 21/10: score as fast as possible). Tolerance class against fd_eval.py.

Same checks and outputs as fd_eval.py (eval/, metrics/, predictions/ per round, audited by collect_eval.py). Changes:
- DAGSNet inference runs as GEMMs over (rows, positions, channels) with eval BatchNorm folded in (FastDAGSNet), compiled
  once per process. Every scored model is gated against its own reference module on the first GATE_ROWS test rows.
  EMA BatchNorm recalibration, the prior and the decision rules are the fd_eval.py / fd.py code.
- One persistent process per GPU scores every requested round; clients are claimed from a shared queue (no idle GPU).
- The controller merges round r (confusion checks, compression, metrics, W&B) while the workers score the next round.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from fd import Run, Method, deterministic, fixture_path
from fd_eval import checked_weights
from mh import pilot as P
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion
from mh.new_fl import digest, save
from mh.v2_methods import calibrate

BATCH, GATE_ROWS = 16384, 8192
# r27: gate 0.1 -> 0.25 (r26 T4 re-scoring: fp16 max logit error up to 0.077 over 8 rounds x 41-101 models; a wrong weight load is O(1))
# r29: absolute bound -> relative to the reference's max |logit| (ze 50c r34: 0.2508 > 0.25; logits grow ~3x r5 -> r48 and the fp16
# error with them, relative max 0.2-0.5 % on 255 local models r5-r48, decisive flips 0; a wrong weight load is O(10 %))
GATE_MAX_REL, GATE_MARGIN = 0.02, 0.25


class FastDAGSNet(torch.nn.Module):
    """Inference-only DAGSNet (mh/dagsnet.py) in (rows, positions, channels) layout with eval BatchNorm folded in. A Conv1d of
    width k is one GEMM to k*out channels followed by a shifted sum over positions; 1x1 convs reading the same tensor share
    one GEMM. `load` copies another model's folded weights into fixed buffers (CUDA-graph safe). T4 21/10: 1.7x the
    compiled module (18.9 vs 29.9-34.3 s per 10.76M rows), im2col GEMMs 1.3x."""
    SHARED = {'stem': ['stem0', 'stem1', 'stem2', 'stem3'], 'g0in': ['g0b1', 'g0b3a', 'g0b5a'], 'g1in': ['g1b1', 'g1b3a', 'g1b5a'],
              's0ex': ['s0e1', 's0e3'], 's1ex': ['s1e1', 's1e3'], 's2ex': ['s2e1', 's2e3']}

    def __init__(self, model):
        super().__init__()
        shared = {n for names in self.SHARED.values() for n in names}
        self.groups = {**{n: [n] for n, _ in self.layers(model) if n not in shared}, **self.SHARED}
        self.shapes = {}
        for key, (w, b, shapes) in self.folded(model).items():
            self.register_buffer(key + '_w', torch.empty_like(w)); self.register_buffer(key + '_b', torch.empty_like(b)); self.shapes[key] = shapes
        for name, t in self.head(model): self.register_buffer(name, torch.empty_like(t))
        self.eps, self.positions, self.channels = model.head[0].eps, model.k, model.patch_len
        self.input_pad = model.input_pad
        self.load(model)

    @staticmethod
    def layers(m):
        out = [(f'stem{i}', s) for i, s in enumerate(m.stems)] + [(f'dense{i}', b) for i, b in enumerate(m.dense.blocks)]
        for i, inc in enumerate(m.google.net):
            out += [(f'g{i}b1', inc.b1), (f'g{i}b3a', inc.b3[0]), (f'g{i}b3b', inc.b3[1]), (f'g{i}b5a', inc.b5[0]),
                    (f'g{i}b5b', inc.b5[1]), (f'g{i}bp', inc.bp[1])]
        out += [(f'alex{i}', m.alex.net[j]) for i, j in enumerate((0, 2, 4))]
        for i, fire in enumerate(m.squeeze.net): out += [(f's{i}q', fire.squeeze), (f's{i}e1', fire.e1), (f's{i}e3', fire.e3)]
        return out

    def folded(self, m):
        """key -> (weight (sum k*out, in): one block of `out` rows per shift, bias (sum out), [(k, out)] per layer)."""
        layers = dict(self.layers(m)); result = {}
        for key, names in self.groups.items():
            ws, bs, shapes = [], [], []
            for name in names:
                conv, bn, _ = layers[name]
                scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
                w = conv.weight.float() * scale[:, None, None]                            # (out, in, k)
                ws.append(w.permute(2, 0, 1).reshape(-1, w.shape[1])); bs.append(bn.bias.float() - bn.running_mean.float() * scale)
                shapes.append((w.shape[2], w.shape[0]))
            result[key] = (torch.cat(ws), torch.cat(bs), shapes)
        return result

    @staticmethod
    def head(m):
        h = m.head   # LayerNorm, Dropout, Linear, ReLU, Dropout, Linear
        return [('ln_w', h[0].weight), ('ln_b', h[0].bias), ('fc1_w', h[2].weight), ('fc1_b', h[2].bias), ('fc2_w', h[5].weight), ('fc2_b', h[5].bias)]

    @torch.no_grad()
    def load(self, model):
        assert not model.training
        for key, (w, b, _) in self.folded(model).items(): getattr(self, key + '_w').copy_(w); getattr(self, key + '_b').copy_(b)
        for name, t in self.head(model): getattr(self, name).copy_(t)

    def cbr(self, key, x):
        """ReLU(BN(conv)) of every layer in the group, each (rows, positions, out); zero padding k // 2 on the position axis."""
        y = F.linear(x, getattr(self, key + '_w')); b = getattr(self, key + '_b'); n = x.shape[1]; out, col, bcol = [], 0, 0
        for k, o in self.shapes[key]:
            v = y[..., col:col + k * o]
            if k > 1:
                v = F.pad(v, (0, 0, k // 2, k // 2)).view(len(x), n + k - 1, k, o)
                v = sum(v[:, j:j + n, j] for j in range(k))   # out[l] = sum_j W_j x[l + j - k // 2]
            out.append(F.relu(v + b[bcol:bcol + o])); col += k * o; bcol += o
        return out

    @staticmethod
    def maxpool3(x):   # MaxPool1d(3, 1, 1)
        xp = F.pad(x, (0, 0, 1, 1), value=float('-inf'))
        return torch.maximum(torch.maximum(xp[:, :-2], xp[:, 1:-1]), xp[:, 2:])

    @staticmethod
    def maxpool2(x):   # MaxPool1d(2, ceil_mode=True)
        if x.shape[1] % 2: x = F.pad(x, (0, 0, 0, 1), value=float('-inf'))
        return x.view(x.shape[0], -1, 2, x.shape[2]).amax(2)

    def forward(self, x):
        if self.input_pad: x = F.pad(x, (0, self.input_pad))
        x = x.view(x.shape[0], self.positions, self.channels)   # DAGSNet.combined: view(B, k, p).transpose(1, 2) in (B, C, L)
        d, g, a, q = self.cbr('stem', x)
        for i in range(3): d = torch.cat([d, *self.cbr(f'dense{i}', d)], dim=-1)
        for i in range(2):
            b1, b3, b5 = self.cbr(f'g{i}in', g)
            g = torch.cat([b1, *self.cbr(f'g{i}b3b', b3), *self.cbr(f'g{i}b5b', b5), *self.cbr(f'g{i}bp', self.maxpool3(g))], dim=-1)
        a, = self.cbr('alex0', a); a, = self.cbr('alex1', self.maxpool2(a)); a, = self.cbr('alex2', self.maxpool2(a))
        for i in range(3): q = torch.cat(self.cbr(f's{i}ex', self.cbr(f's{i}q', q)[0]), dim=-1)
        z = torch.cat([d.mean(1), g.mean(1), a.mean(1), q.mean(1)], dim=-1)
        z = F.layer_norm(z.float(), (z.shape[-1],), self.ln_w, self.ln_b, self.eps)
        return F.linear(F.relu(F.linear(z, self.fc1_w, self.fc1_b)), self.fc2_w, self.fc2_b)


class Scorer:
    """Replaces Run.logits: masked test rows through the compiled FastDAGSNet, padded tail; gate against the module itself."""
    def __init__(self, R):
        self.R, self.fast = R, FastDAGSNet(copy.deepcopy(R.init).eval()).to(R.dev)
        def forward(x):
            with torch.autocast('cuda', dtype=torch.float16): return self.fast(R.mask(x.float())).float()
        self.compiled = torch.compile(forward, mode='reduce-overhead', dynamic=False); self.gates = []

    @torch.no_grad()
    def __call__(self, model, X):
        model.eval(); self.fast.load(model); out = []
        for start in range(0, len(X), BATCH):
            x = X[start:start + BATCH]; n = len(x)
            if n < BATCH: x = F.pad(x, (0, 0, 0, BATCH - n))
            torch.compiler.cudagraph_mark_step_begin(); out.append(self.compiled(x)[:n].clone())
        z = torch.cat(out)
        with torch.autocast('cuda', dtype=torch.float16): ref = model(self.R.mask(X[:GATE_ROWS].float()))[0].float()
        error = float((z[:GATE_ROWS] - ref).abs().max()); rel = error / float(ref.abs().max())
        top = ref.topk(2, dim=1).values; decisive = (top[:, 0] - top[:, 1]) > GATE_MARGIN
        flips = int((z[:GATE_ROWS].argmax(1) != ref.argmax(1)).sum())
        assert rel <= GATE_MAX_REL and torch.equal(z[:GATE_ROWS].argmax(1)[decisive], ref.argmax(1)[decisive]), ('fast gate', error, rel)
        self.gates.append(dict(max_error=error, max_rel=rel, flips=flips))
        return z


def claim(folder, job):
    try: os.close(os.open(folder / str(job), os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError: return False


def worker(a):
    deterministic()   # reproducible run to run; no cuDNN autotune
    cfg, _, w = checked_weights(a.run, a.rounds[0])
    config = dict(cfg['config']); config.pop('parent', None)
    args = SimpleNamespace(**{**config, 'clients': config.get('clients', 20), 'workers': 2, 'rank': a.rank, 'evaluate_only': True,
                              'wandb_online': False})
    R = Run(args)
    assert digest(fixture_path(args) / 'manifest.json') == cfg['fixture'], 'evaluation fixture differs from training'
    m = Method(R, initialize=False); m.prior()   # same SecureSum prior arithmetic as inline evaluation
    assert sum(p.numel() for p in R.init.parameters()) == 395024
    scorer = Scorer(R); R.logits = scorer
    y = R.H['test'][1]
    for i, r in enumerate(a.rounds):
        if i >= 2:   # at most two rounds of parts on disk: wait for the merge of round i-2
            while not (a.out / 'eval' / f'round_{a.rounds[i - 2]:03d}.json').exists(): time.sleep(5)
        cfg_r, _, w = checked_weights(a.run, r); assert cfg_r == cfg
        assert len(w['private']) == len(w['ema_private']) == R.K and len(w['proxy']) == 1
        for state in w['private'] + w['ema_private'] + w['proxy']:
            assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()), 'nonfinite weights'
        part = a.scratch / f'round_{r:03d}'; claims = part / 'claims'; claims.mkdir(parents=True, exist_ok=True)
        confusions, jobs, gates = {}, [], {}
        def put(role, k, model, rules):
            scorer.gates.clear()
            for rule, (cm, pred) in R.score(model, k, 'test', rules).items():
                np.save(part / f'{role}_{rule}_{k}.npy', pred); confusions[f'{role}_{rule}_{k}'] = cm.tolist()
            gates[f'{role}_{k}'] = scorer.gates[-1]
        for job in ['proxy', *range(R.K)]:
            if not claim(claims, job): continue
            jobs.append(job)
            if job == 'proxy':
                m.proxy[0].load_state_dict(w['proxy'][0], strict=True); put('proxy', 0, m.proxy[0], ('head',)); continue
            model = R.priv[job]; model.load_state_dict(w['private'][job], strict=True); put('private', job, model, R.rules)
            model = copy.deepcopy(R.priv[job]); model.load_state_dict(w['ema_private'][job], strict=True)
            calibrate(model, R.mask(R.X[job][R.cal[job]].float())); put('private_ema', job, model, ('head', 'gp')); del model
            save(a.out / f'progress_rank{a.rank}.json', dict(round=r, client=job, clients_done=len(jobs), timestamp=time.time()))
        if a.rank == 0: np.save(part / 'test_y.npy', y.cpu().numpy().astype('uint8'))
        save(part / f'rank{a.rank}.json', dict(jobs=jobs, confusions=confusions, gates=gates, rules=list(R.rules), K=R.K))


def merge(a, r, cfg, row):
    part = a.scratch / f'round_{r:03d}'
    metas = [json.loads((part / f'rank{rank}.json').read_text()) for rank in range(2)]
    K, rules = metas[0]['K'], metas[0]['rules']; cms = {}; gates = {}
    for meta in metas: cms.update(meta['confusions']); gates.update(meta['gates'])
    assert sorted(j for meta in metas for j in meta['jobs'] if j != 'proxy') == list(range(K)), 'eval client coverage'
    assert sum(meta['jobs'].count('proxy') for meta in metas) == 1
    test_y = np.load(part / 'test_y.npy'); yy = test_y.astype(np.int64) * 16
    predictions, evaluated = {'test_y': test_y}, {}
    for role, role_rules in [('private', rules), ('private_ema', ('head', 'gp')), ('proxy', ('head',))]:
        result = {}
        for rule in role_rules:
            n = 1 if role == 'proxy' else K
            stacked = np.empty((n, len(test_y)), dtype=np.uint8); cc = []
            for k in range(n):
                stacked[k] = np.load(part / f'{role}_{rule}_{k}.npy'); cm = np.array(cms[f'{role}_{rule}_{k}'], dtype=np.int64)
                actual = np.bincount(yy + stacked[k], minlength=256).reshape(16, 16)
                assert np.array_equal(actual, cm) and cm.sum() == len(test_y); cc.append(cm)
            if role == 'proxy': cc *= K
            mm = [metrics_from_confusion(c) for c in cc]
            pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cc], axis=0)
            result[rule] = dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
            predictions[f'{role}_test_{rule}'] = stacked
        evaluated[role] = {'test': result}
    pf = a.out / 'predictions' / f'round_{r:03d}.npz'; pf.parent.mkdir(exist_ok=True)
    with pf.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f, **predictions)
    os.replace(pf.with_suffix('.tmp'), pf)
    cf = a.out / 'metrics' / f'round_{r:03d}.csv'; cf.parent.mkdir(exist_ok=True)
    with cf.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'role', 'holdout', 'rule', *METRIC_KEYS]); writer.writeheader()
        for role, holds in evaluated.items():
            for rule, v in holds['test'].items():
                writer.writerow(dict(round=r, role=role, holdout='test', rule=rule, **v['mean_metrics']))
    gate = dict(max_error=max(g['max_error'] for g in gates.values()), max_rel=max(g['max_rel'] for g in gates.values()), flips=sum(g['flips'] for g in gates.values()), rows=GATE_ROWS, models=len(gates))
    save(a.out / 'eval' / f'round_{r:03d}.json',
         dict(round=r, fingerprint=cfg['fingerprint'], weights_sha=row['weights_sha'], predictions_sha=digest(pf), eval=evaluated,
              eval_source_sha=digest(Path(__file__)), evaluator='fast-gemm', fast_gate=gate))
    shutil.rmtree(part)
    print('EVALUATED', r, cfg['fingerprint'], 'clients', K, 'gate', gate, flush=True)
    return evaluated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', required=True, help='comma list of committed rounds, scored in this order')
    p.add_argument('--scratch', type=Path, required=True, help='uncompressed per-round parts, removed after each merge')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--worker', action='store_true')
    p.add_argument('--wandb_online', action='store_true')
    a = p.parse_args(); a.rounds = [int(v) for v in a.rounds.split(',')]; a.out.mkdir(parents=True, exist_ok=True)
    if a.worker: return worker(a)
    for r in a.rounds: assert not (a.out / 'eval' / f'round_{r:03d}.json').exists(), ('evaluation already committed', r)
    checked = {r: checked_weights(a.run, r)[:2] for r in a.rounds}
    if a.wandb_online:
        import wandb
        assert wandb.login(verify=True)
    started = time.monotonic(); procs = []
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count())))).split(',')
    for rank in range(2):
        log = (a.out / f'worker{rank}.log').open('a')
        command = [sys.executable, str(Path(__file__)), '--run', str(a.run), '--out', str(a.out), '--rounds', ','.join(map(str, a.rounds)),
                   '--scratch', str(a.scratch), '--rank', str(rank), '--worker']
        procs.append((subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible[rank % len(visible)]), stdout=log,
                                       stderr=subprocess.STDOUT), log))
    def alive():
        if any(proc.poll() not in (None, 0) for proc, _ in procs):
            for proc, _ in procs:
                if proc.poll() is None: proc.terminate()
            raise RuntimeError('evaluation worker failed; inspect worker logs')
    for r in a.rounds:
        cfg, row = checked[r]; t = time.monotonic(); wb = None
        if a.wandb_online:
            wb = wandb.init(project='mobihet-fulldrift', id=f'evalfast-{a.out.name}-r{r:03d}-{cfg["fingerprint"][:8]}',
                            name=f'evalfast-{a.out.name}-r{r:03d}', config=dict(phase='evaluation', evaluator='fast-gemm', round=r,
                                                                                 fingerprint=cfg['fingerprint']))
            save(a.out / f'wandb_run_r{r:03d}.json', dict(entity=wb.entity, project=wb.project, id=wb.id, url=wb.url))
            wb.log({'eval/started': 1})
        part = a.scratch / f'round_{r:03d}'
        while not all((part / f'rank{rank}.json').exists() for rank in range(2)):
            alive()
            if wb:
                progress = [json.loads(f.read_text()) for f in a.out.glob('progress_rank*.json')]
                wb.log({'eval/clients_done': sum(v['clients_done'] for v in progress if v['round'] == r), 'eval/seconds': time.monotonic() - t})
            time.sleep(5)
        evaluated = merge(a, r, cfg, row)
        if wb:
            wb.log({f'{role}/test/{rule}/{key}': value for role, holds in evaluated.items()
                    for rule, ev in holds['test'].items() for key, value in ev['mean_metrics'].items()})
            wb.summary['complete'] = True; wb.finish()
        print('evaluation round', r, 'sec', round(time.monotonic() - t, 1), 'total', round(time.monotonic() - started, 1), flush=True)
    for proc, log in procs:
        rc = proc.wait(); log.close(); assert rc == 0, ('evaluation worker failed', rc)


if __name__ == '__main__': main()

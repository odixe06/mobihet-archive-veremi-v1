"""Local code test (pilot fixture, 1024 rows/client, eager): fd_dml/fd_gaussian epoch == frozen full_run epoch bit-exact;
every arm runs a 2-epoch smoke. Run via scripts/run_local_guarded.py.

    python methods/full-drift/builds/fd-probe/scripts/test_fd_local.py --work <scratch> --part checks|smoke [--manifest m.json]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
FULL = ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj'


def one_epoch(fd, arm, use_frozen, batch=512, epochs=1, hook=None):
    import torch
    from full_run import Run as FullRun
    aux = 128 if 'g512' in arm.split('_') else 64   # round 18 g512: main() doubles the smoke aux_batch
    a = SimpleNamespace(arm=arm, seed=42, rounds=50, batch=batch, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=aux, aux_weight=1.,
                        backend='eager', smoke=True, wandb_online=False)
    R = fd.Run(a)
    if hook: hook(R)
    m = fd.Method(R); m.prior()
    for r in range(1, epochs + 1):
        m.start_round()
        if getattr(R, 'cv', None): R.cv_round(m, r)   # round 25: as train()
        for k in range(R.K):
            if m.kd: m.prepare(k, r)
            if use_frozen: FullRun.private_epoch(R, m, k, r)
            else: R.private_epoch(m, k, r)
            if m.kd:
                packet, _ = m.proxy_epoch(k, r)
                if packet is not None: m.aggregate_sum.add(k, packet)
        m.aggregate(r)
    state = [v.clone() for model in R.priv for v in model.state_dict().values()]
    if m.kd: state += [v.clone() for p in (m.proxy if m.own else [m.global_state]) for v in (p.state_dict() if m.own else p).values()]
    return state


def round8(fd, work):
    """Round 8 hooks: old arms bit-exact vs the pushed round-5 fd.py (KD diagnostics/probe have no side effects); kdcap with an
    infinite cap == m4b; frz epoch 1 == wu epoch 1 (no private KD before a snapshot); FedDyn server/client algebra; JS/two views."""
    import importlib.util
    import torch
    stage = BASE / 'artifacts/input/fd-src-20261015-r5/fd.py'
    spec = importlib.util.spec_from_file_location('fd_r5', stage); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b', 'fd_dml_swappos', 'fd_gaussian_m4b'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs round-5 fd.py, 2 epochs:', arm, flush=True)
    m4b = one_epoch(fd, 'fd_dml_m4b', False, batch=128, epochs=2)
    cap, fd.KD_CAP = fd.KD_CAP, float('inf')
    assert eq(one_epoch(fd, 'fd_dml_m4b_kdcap', False, batch=128, epochs=2), m4b); fd.KD_CAP = cap
    kdcap = one_epoch(fd, 'fd_dml_m4b_kdcap', False, batch=128, epochs=2); assert not eq(kdcap, m4b)
    print('kdcap: infinite cap == m4b; cap 1 differs', flush=True)
    wu1, frz1 = one_epoch(fd, 'fd_dml_m4b_kdcap_wu', False, batch=128), one_epoch(fd, 'fd_dml_m4b_kdcap_frz', False, batch=128)
    assert eq(wu1, frz1) and not eq(wu1, one_epoch(fd, 'fd_dml_m4b_kdcap', False, batch=128))
    wu2, frz2 = (one_epoch(fd, a, False, batch=128, epochs=2) for a in ('fd_dml_m4b_kdcap_wu', 'fd_dml_m4b_kdcap_frz'))
    assert not eq(wu2, frz2); print('wu/frz: epoch 1 identical (private KD off before a snapshot), epoch 2 differs', flush=True)
    # own: proxies stay local (never averaged), private differs from the pooled arm
    own = one_epoch(fd, 'fd_dml_m4b_own', False, batch=128); assert not eq(own[:len(own) // 2], m4b[:len(own) // 2])
    print('own: one epoch runs with 20 local proxies', flush=True)
    # FedDyn server: w1 = m1 + (m1 - w0); w2 = m2 - h2/alpha with h2 = -alpha (m1 - w0) - alpha (m2 - w1)
    a = SimpleNamespace(arm='fd_dml_m4b_dyn', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
                        aux_weight=1., backend='eager', smoke=True, wandb_online=False)
    R = fd.Run(a); m = fd.Method(R); m.prior(); names = [n for n, _ in m.proxy[0].named_parameters()]
    w0 = {n: m.global_state[n].clone() for n in names}; g = torch.Generator(device=R.dev).manual_seed(0)
    def push(step, base):
        m.start_round(); means = {n: torch.zeros_like(base[n]) for n in names}
        for k in range(R.K):
            m.prepare(k, step)
            with torch.no_grad():
                for n, p in m.proxy[k].named_parameters():
                    d = 1e-2 * torch.randn(p.shape, device=R.dev, generator=g); p.add_(d); means[n] += (base[n] + d) / R.K
            m.logs[k] = {}; packet, _ = m.proxy_epoch(k, step); m.aggregate_sum.add(k, packet)
        m.aggregate(step); return means
    m1 = push(1, w0); w1 = {n: m.global_state[n].clone() for n in names}
    err1 = max(float((w1[n] - (2 * m1[n] - w0[n])).abs().max()) for n in names)
    m2 = push(2, w1); err2 = max(float((m.global_state[n] - (m2[n] + (m1[n] - w0[n]) + (m2[n] - w1[n]))).abs().max()) for n in names)
    assert err1 < 1e-5 and err2 < 1e-5, (err1, err2); print('FedDyn server algebra PASS', err1, err2, flush=True)
    # FedDyn client: a_k = -alpha (w_k - w_t) after one local epoch from a_k = 0
    R = fd.Run(a); m = fd.Method(R); m.prior(); m.start_round(); m.prepare(0, 1); R.private_epoch(m, 0, 1)
    errs = [float((ak + fd.DYN_ALPHA * (p.float() - m.global_state[n].float())).abs().max())
            for ak, (n, p) in zip(m.dyn_a[0], m.proxy[0].named_parameters())]
    assert max(errs) < 1e-7 and any(float(x.abs().max()) > 0 for x in m.dyn_a[0]); print('FedDyn client a_k PASS', flush=True)
    dyn = one_epoch(fd, 'fd_dml_m4b_dyn', False, batch=128, epochs=2)
    assert not eq(dyn, m4b) and all(torch.isfinite(v.float()).all() for v in dyn); print('dyn 2 epochs finite, differs', flush=True)
    # JS and the two views
    za, zb = torch.randn(32, 16), torch.randn(32, 16); p, q = za.softmax(1), zb.softmax(1); mm = (p + q) / 2
    ref = 0.5 * ((p * (p / mm).log()).sum(1) + (q * (q / mm).log()).sum(1)).mean()
    assert abs(float(fd.js(za, zb)) - float(ref)) < 1e-6 and float(fd.js(za, za)) < 1e-7
    R = fd.Run(SimpleNamespace(arm='fd_dml_swappos_cons', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
                               aux_weight=1., backend='eager', smoke=True, wandb_online=False))
    index = torch.arange(64, device=R.dev); x = R.X[0][index].float()
    va, vb = R.two_views(x, index, 0, torch.Generator(device=R.dev).manual_seed(1))
    other = [c for c in range(x.shape[1]) if c not in R.block.tolist()]
    assert torch.equal(va, R.mask(x)) and torch.equal(va[:, other], vb[:, other]) and not torch.equal(va, vb) and (va[:, R.zero] == 0).all()
    cons, cons0 = (one_epoch(fd, arm, False, batch=128) for arm in ('fd_dml_swappos_cons', 'fd_dml_swappos_cons0'))
    assert not eq(cons, cons0); print('JS == reference; view b differs only in the 9-feature block; cons != cons0', flush=True)
    for arm in ('fd_dml_m4b_kdcap', 'fd_dml_m4b_kdcap_wu', 'fd_dml_m4b_kdcap_frz', 'fd_dml_m4b_own', 'fd_gaussian_m4b_own',
                'fd_dml_m4b_dyn', 'fd_dml_swappos_cons', 'fd_dml_swappos_cons0'):
        state = one_epoch(fd, arm, False, batch=128); assert all(torch.isfinite(v.float()).all() for v in state)
    print('round 8: every new arm one epoch finite', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True)
    p.add_argument('--manifest', help='data_manifest.json pulled from a Kaggle setup: exercises the production gate')
    # Owner RAM limit: run 'checks' and 'smoke' as separate invocations so only one CUDA process is alive at a time.
    p.add_argument('--part', choices=('checks', 'smoke', 'r8', 'r9', 'r11', 'r12', 'r13', 'r15', 'r17', 'r18', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r32', 'r33', 'r34', 'final', 'resume', 'fast', 'parent', 'par'), required=True)
    p.add_argument('--arms', help='smoke only these comma-separated arms')
    a = p.parse_args()
    work = Path(a.work); proj = work / 'proj'
    if proj.exists(): shutil.rmtree(proj)
    proj.mkdir(parents=True)
    for f in ('full_run.py', 'full_method.py', 'full_driver.py'): shutil.copyfile(FULL / f, proj / f)
    shutil.copyfile(BASE / 'source/proj/fd.py', proj / 'fd.py')
    sys.path[:0] = [str(proj), str(ROOT)]
    if a.part == 'smoke':
        import ast
        tree = ast.parse((proj / 'fd.py').read_text())
        arms = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'ARMS')
        return smoke([x for x in arms if not a.arms or x in a.arms.split(',')], proj, work)
    if a.part == 'par':
        return parallel(proj, work, a.arms.split(',') if a.arms else None)
    if a.part == 'parent':
        return parent(proj, work)
    import torch
    import fd
    if a.part == 'r8':
        return round8(fd, work)
    if a.part == 'r9':
        return round9(fd, work)
    if a.part == 'r11':
        return round11(fd, work)
    if a.part == 'r12':
        return round12(fd, work)
    if a.part == 'r13':
        return round13(fd, work)
    if a.part == 'r15':
        return round15(fd, work)
    if a.part == 'r17':
        return round17(fd, work)
    if a.part == 'r18':
        return round18(fd, work)
    if a.part == 'r20':
        return round20(fd, work)
    if a.part == 'r21':
        return round21(fd, work)
    if a.part == 'r22':
        return round22(fd, work)
    if a.part == 'r23':
        return round23(fd, work)
    if a.part == 'r24':
        return round24(fd, work)
    if a.part == 'r25':
        return round25(fd, work)
    if a.part == 'r26':
        return round26(fd, work)
    if a.part == 'r32':
        return round32(fd)
    if a.part == 'r33':
        return round33(fd)
    if a.part == 'r34':
        return round34(fd)
    if a.part == 'resume':
        return resume(proj, work)
    if a.part == 'final':
        return final(fd, proj, work)
    if a.part == 'fast':
        return fast(fd, proj, work)
    if a.manifest:
        import json
        manifest = json.loads(Path(a.manifest).read_text()); n = [sum(manifest['counts'][f'train_{k}']) for k in range(20)]
        fd.check_full_manifest(manifest, n, 10761343)
        for bad in (n[:-1] + [n[-1] - 1], ):
            try: fd.check_full_manifest(manifest, bad, 10761343)
            except AssertionError: pass
            else: raise SystemExit('production gate accepted wrong row counts')
        print('production manifest gate PASS on', a.manifest, 'train rows', sum(n), flush=True)
    for arm in ('fd_dml', 'fd_gaussian'):
        ours, frozen = one_epoch(fd, arm, False), one_epoch(fd, arm, True)
        assert len(ours) == len(frozen) and all(torch.equal(x, y) for x, y in zip(ours, frozen)), arm
        print('BIT-EXACT vs frozen full_run epoch:', arm, len(ours), 'tensors', flush=True)
    ctl, q4 = one_epoch(fd, 'fd_dml', False, batch=128), one_epoch(fd, 'fd_dml_q4', False, batch=128)
    rel = max(float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)) for x, y in zip(q4, ctl) if y.is_floating_point())
    assert 0 < rel < .5, rel; print('q4 differs from dense but stays close: max rel', round(rel, 4), flush=True)
    # QPacket: decode == quantize() tensors; corrupted bytes refused; real bytes vs the old counter (32 + payload)
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b_q4', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
                               aux_weight=1., backend='eager', smoke=True, wandb_online=False))
    m = fd.Method(R); m.prior(); g = torch.Generator(device=R.dev).manual_seed(5); entries, expect, counter = {}, {}, 32
    for name, v in m.proxy[0].state_dict().items():
        if not v.is_floating_point(): entries[name] = expect[name] = v; counter += 8 * v.numel()
        elif fd.is_buffer(name): entries[name] = v; expect[name] = v.half().float(); counter += 2 * v.numel()
        else:
            wire, q, scale, n = fd.quantize(0.01 * torch.randn(v.shape, device=R.dev, generator=g), 4, g)
            entries[name] = (q, scale); expect[name] = wire; counter += n
    packet = m.qpacket.encode(entries, 3, 4); got = m.qpacket.decode(packet, R.dev, 3)
    assert got.keys() == expect.keys() and all(torch.equal(got[n], expect[n]) for n in expect)
    bad = bytearray(packet); bad[-1] ^= 1
    for wrong, base in ((bytes(bad), 3), (packet, 2)):
        try: m.qpacket.decode(wrong, R.dev, base)
        except AssertionError: pass
        else: raise SystemExit('QPacket accepted a corrupted/stale packet')
    print('QPacket int4: decode == quantize(); CRC/base checked; bytes', len(packet), 'counter', counter, flush=True)
    zs, zr, y = torch.randn(64, 16), torch.randn(64, 16), torch.randint(0, 16, (64,))
    kl = lambda a, b: float(torch.nn.functional.kl_div(fd.not_true(a, y), fd.not_true(b, y), log_target=True, reduction='batchmean'))
    moved = zr.clone(); moved[torch.arange(64), y] += 9.
    keep = ~torch.nn.functional.one_hot(y, 16).bool()
    ref = sum(torch.nn.functional.kl_div(zs[i][keep[i]].log_softmax(0), zr[i][keep[i]].log_softmax(0), log_target=True, reduction='sum')
              for i in range(64)) / 64
    assert abs(kl(zs, zr) - kl(zs, moved)) < 1e-6 and abs(kl(zs, zr) - float(ref)) < 1e-5
    print('ntd KL ignores the teacher true-class logit and equals the 15-class KL', flush=True)
    base = dict(seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64, aux_weight=1., backend='eager',
                smoke=True, wandb_online=False)
    for arm, n in (('fd_dml_m4b', 13), ('fd_dml_swappos', 4), ('fd_dml_ntd', 0), ('fd_gaussian_m4b', 13)):
        R = fd.Run(SimpleNamespace(arm=arm, **base)); x = R.mask(R.X[0][:8].float())
        assert len(R.zero) == n and (x[:, R.zero] == 0).all() and R.swap == (arm == 'fd_dml_swappos') and R.ntd == (arm == 'fd_dml_ntd')
        del R
    print('round-3 hooks: m4b zeroes 13 receiver/position features, swappos swaps + zeroes 4, ntd masks none', flush=True)
    for arm in ('fd_dml_ntd', 'fd_dml_hkd'):
        state = one_epoch(fd, arm, False, batch=128)
        assert any(not torch.equal(x, y) for x, y in zip(ctl, state)); print(arm, 'differs from control', flush=True)
    for arm in ('fd_dml', 'fd_dml_avgm', 'fd_fedavg', 'fd_dml_m4b', 'fd_dml_swappos', 'fd_dml_ntd', 'fd_dml_q4', 'fd_dml_hkd', 'fd_dml_m4b_q4'):
        state = one_epoch(fd, arm, False, batch=128)
        assert all(torch.isfinite(v.float()).all() for v in state); print('one epoch finite:', arm, flush=True)


def round9(fd, work):
    """Round 9 hooks leave round-5/8 arms bit-exact; scaf epoch 1 == sgd epoch 1 (c = c_i = 0) then differs; SCAFFOLD Option II
    client/server control algebra; private-only Gaussian/replay rows (replay labels match the drawn rows)."""
    import importlib.util
    import torch
    load = lambda tag: importlib.util.spec_from_file_location('fd_' + tag, BASE / f'artifacts/input/fd-src-20261015-{tag}/fd.py')
    olds = {}
    for tag in ('r5', 'r8'):
        spec = load(tag); olds[tag] = importlib.util.module_from_spec(spec); spec.loader.exec_module(olds[tag])
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for tag, arm in (('r5', 'fd_dml_m4b'), ('r5', 'fd_gaussian_m4b'), ('r8', 'fd_dml_m4b_kdcap_frz'), ('r8', 'fd_dml_m4b_dyn'),
                     ('r8', 'fd_dml_swappos_cons'), ('r8', 'fd_dml_m4b_own')):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(olds[tag], arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs', tag, 'fd.py, 2 epochs:', arm, flush=True)
    sgd1, scaf1 = one_epoch(fd, 'fd_dml_m4b_sgd3', False, batch=128), one_epoch(fd, 'fd_dml_m4b_scaf3', False, batch=128)
    assert eq(sgd1, scaf1) and not eq(sgd1, one_epoch(fd, 'fd_dml_m4b', False, batch=128))
    sgd2, scaf2 = (one_epoch(fd, a, False, batch=128, epochs=2) for a in ('fd_dml_m4b_sgd3', 'fd_dml_m4b_scaf3'))
    assert not eq(sgd2, scaf2) and all(torch.isfinite(v.float()).all() for v in scaf2)
    print('scaf epoch 1 == sgd epoch 1 (zero controls), epoch 2 differs; SGD proxy differs from AdamW', flush=True)
    a = SimpleNamespace(arm='fd_dml_m4b_scaf3', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
                        aux_weight=1., backend='eager', smoke=True, wandb_online=False)
    R = fd.Run(a); m = fd.Method(R); m.prior(); m.start_round(); names = [n for n, _ in m.proxy[0].named_parameters()]
    eta = 0.03 * R.lr(1) / R.a.lr; news = []
    for k in range(R.K):
        m.prepare(k, 1); log = R.private_epoch(m, k, 1)
        expect = [(m.global_state[n].float() - p.float()) / (log['applied'] * eta) for n, p in m.proxy[k].named_parameters()]
        assert max(float((x - y).abs().max()) for x, y in zip(m.scaf_ci[k], expect)) < 1e-4; news.append(m.scaf_ci[k])
        packet, _ = m.proxy_epoch(k, 1); m.aggregate_sum.add(k, packet)
    m.aggregate(1)
    mean = [sum(v[i] for v in news) / R.K for i in range(len(names))]
    assert max(float((x - y).abs().max()) for x, y in zip(m.scaf_c, mean)) < 1e-5; print('SCAFFOLD Option II client/server PASS', flush=True)
    R = fd.Run(SimpleNamespace(**dict(vars(a), arm='fd_dml_m4b_replay'))); g = torch.Generator(device=R.dev).manual_seed(3)
    for k in range(R.K):
        order, start, count, present = R.by_class[k]; rows, labels = R.replay_rows(k, 64, g)
        idx = order[start[labels] + 0]; assert set(labels.tolist()) <= set(present.tolist())
        assert torch.equal(R.Y[k][order[start[labels]]], labels) and (rows[:, R.zero] == 0).all()
    print('replay: labels cycle over held classes; rows masked', flush=True)
    states = {arm: one_epoch(fd, arm, False, batch=128) for arm in ('fd_dml_m4b_gpool', 'fd_dml_m4b_gown', 'fd_dml_m4b_replay',
                                                                   'fd_dml_m4b_sgd1', 'fd_dml_m4b_sgd10', 'fd_dml_m4b_scaf10')}
    assert all(torch.isfinite(v.float()).all() for s in states.values() for v in s)
    assert not eq(states['fd_dml_m4b_gpool'], states['fd_dml_m4b_gown']) and not eq(states['fd_dml_m4b_gpool'], states['fd_dml_m4b_replay'])
    print('round 9: every new arm one epoch finite; gpool != gown != replay', flush=True)


def round11(fd, work):
    """Round 11 hooks leave round-5/8/10 arms bit-exact; kd2 with multiplier 1 == gpool, kdpx with infinite cap == gpool,
    gw2 with weight 1 == gpool; each differs at its real value."""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location('fd_r10', BASE / 'artifacts/input/fd-src-20261015-r10/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b', 'fd_dml_m4b_gpool', 'fd_dml_m4b_kdcap', 'fd_dml_m4b_q4', 'fd_dml_m4b_replay', 'fd_gaussian_m4b',
                'fd_dml_swappos'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r10 fd.py, 2 epochs:', arm, flush=True)
    gpool = one_epoch(fd, 'fd_dml_m4b_gpool', False, batch=128, epochs=2)
    up, fd.KD_UP = fd.KD_UP, 1.0
    assert eq(one_epoch(fd, 'fd_dml_m4b_gpool_kd2', False, batch=128, epochs=2), gpool); fd.KD_UP = up
    assert not eq(one_epoch(fd, 'fd_dml_m4b_gpool_kd2', False, batch=128, epochs=2), gpool)
    cap, fd.KD_CAP = fd.KD_CAP, float('inf')
    assert eq(one_epoch(fd, 'fd_dml_m4b_gpool_kdpx', False, batch=128, epochs=2), gpool); fd.KD_CAP = cap
    assert not eq(one_epoch(fd, 'fd_dml_m4b_gpool_kdpx', False, batch=128, epochs=2), gpool)
    gw, fd.GW2 = fd.GW2, 1.0
    assert eq(one_epoch(fd, 'fd_dml_m4b_gpool_gw2', False, batch=128, epochs=2), gpool); fd.GW2 = gw
    assert not eq(one_epoch(fd, 'fd_dml_m4b_gpool_gw2', False, batch=128, epochs=2), gpool)
    print('kd2 (x1) == gpool, kdpx (cap inf) == gpool, gw2 (x1) == gpool; all differ at their real values', flush=True)


def round12(fd, work):
    """Source after round 11 (sync-free loop, QPacket, --workers) leaves every round-11 arm bit-exact over 2 epochs; gpool_q4:
    round-1 privates == gpool (quantization acts only after local training), proxy differs, real QPacket bytes up/down."""
    import importlib.util
    import torch
    from types import SimpleNamespace as NS
    spec = importlib.util.spec_from_file_location('fd_r11', BASE / 'artifacts/input/fd-src-20261015-r11/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b_gpool', 'fd_dml_m4b_q4', 'fd_dml_m4b_gpool_kd2', 'fd_dml_m4b_gpool_kdpx', 'fd_dml_m4b_gpool_gw2',
                'fd_dml_m4b', 'fd_gaussian_m4b', 'fd_dml_swappos', 'fd_dml_m4b_replay', 'fd_dml_m4b_own'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r11 fd.py, 2 epochs:', arm, flush=True)
    gpool, q = (one_epoch(fd, arm, False, batch=128) for arm in ('fd_dml_m4b_gpool', 'fd_dml_m4b_gpool_q4'))
    private = 20 * len(fd.P.make('dagsnet').state_dict())
    assert eq(gpool[:private], q[:private]) and not eq(gpool[private:], q[private:])
    q2 = one_epoch(fd, 'fd_dml_m4b_gpool_q4', False, batch=128, epochs=2)
    assert all(torch.isfinite(v.float()).all() for v in q2) and not eq(q2, one_epoch(fd, 'fd_dml_m4b_q4', False, batch=128, epochs=2))
    R = fd.Run(NS(arm='fd_dml_m4b_gpool_q4', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
                  aux_weight=1., backend='eager', smoke=True, wandb_online=False))
    m = fd.Method(R); m.prior(); m.start_round()
    for k in range(R.K):
        m.prepare(k, 1); R.private_epoch(m, k, 1); m.proxy_epoch(k, 1)
    m.aggregate(1)
    rows = [b for b in R.bytes if b[0] == 1 and b[3] == 'proxy_quantized']
    assert len(rows) == 2 * R.K and {b[4] for b in rows} == {208268} and len(m.latest_packet) == 208268
    print('gpool_q4: round-1 privates == gpool, proxy differs, 2 epochs finite; QPacket 208268 B up and down per client', flush=True)


def round15(fd, work):
    """Round 15 hooks leave round-13 arms bit-exact over 2 epochs; gpool_frz_gkd epoch 1 == gpool_frz (no snapshot yet: hard CE),
    epoch 2 differs and stays finite; frz arms train 2 finite epochs with the teacher compiled path untouched (eager here)."""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location('fd_r13', BASE / 'artifacts/input/fd-src-20261015-r13/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b_gpool', 'fd_dml_m4b', 'fd_dml_m4b_gpool_rej', 'fd_dml_m4b_gpool_p1', 'fd_dml_m4b_kdcap_frz',
                'fd_dml_m4b_gpool_q2', 'fd_dml_m4b_gpool_fill'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r13 fd.py, 2 epochs:', arm, flush=True)
    frz1, frz2 = (one_epoch(fd, 'fd_dml_m4b_gpool_frz', False, batch=128, epochs=e) for e in (1, 2))
    gkd1, gkd2 = (one_epoch(fd, 'fd_dml_m4b_gpool_frz_gkd', False, batch=128, epochs=e) for e in (1, 2))
    assert eq(gkd1, frz1) and not eq(gkd2, frz2) and all(torch.isfinite(v.float()).all() for v in gkd2)
    print('gkd: epoch 1 == gpool_frz (hard CE before the first snapshot), epoch 2 differs, finite', flush=True)
    m4b_frz = one_epoch(fd, 'fd_dml_m4b_frz', False, batch=128, epochs=2)
    assert all(torch.isfinite(v.float()).all() for v in m4b_frz) and not eq(m4b_frz, one_epoch(fd, 'fd_dml_m4b', False, batch=128, epochs=2))
    print('fd_dml_m4b_frz: 2 finite epochs, differs from fd_dml_m4b', flush=True)
    q2g = one_epoch(fd, 'fd_dml_m4b_gpool_q2_frz_gkd', False, batch=128, epochs=2)
    assert all(torch.isfinite(v.float()).all() for v in q2g) and not eq(q2g, gkd2)
    print('fd_dml_m4b_gpool_q2_frz_gkd: 2 finite epochs, differs from the dense-wire gkd', flush=True)


def round17(fd, work):
    """Round 17 hooks (la token, parent import) leave the round-16 arms bit-exact over 2 epochs; la differs from gkd from epoch 1
    (its CE acts before any snapshot), kdpx keeps the epoch-1 privates of gkd (no private KD before the first snapshot) but not
    its proxy; all new arms finite over 2 epochs."""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location('fd_r16', BASE / 'artifacts/input/fd-src-20261015-r16/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    for arm in ('fd_dml_m4b_gpool_frz_gkd', 'fd_dml_m4b_gpool_frz', 'fd_dml_m4b_frz', 'fd_dml_m4b_gpool', 'fd_dml_m4b',
                'fd_dml_m4b_gpool_kdpx', 'fd_dml_m4b_gpool_q2_frz_gkd', 'fd_dml_m4b_gpool_rej'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r16 fd.py, 2 epochs:', arm, flush=True)
    private = 20 * len(fd.P.make('dagsnet').state_dict())
    gkd1 = one_epoch(fd, 'fd_dml_m4b_gpool_frz_gkd', False, batch=128)
    la1 = one_epoch(fd, 'fd_dml_m4b_gpool_frz_gkd_la', False, batch=128)
    px1 = one_epoch(fd, 'fd_dml_m4b_gpool_frz_gkd_kdpx', False, batch=128)
    assert not eq(la1[:private], gkd1[:private]) and eq(px1[:private], gkd1[:private]) and not eq(px1[private:], gkd1[private:])
    print('epoch 1: la privates differ from gkd; kdpx privates == gkd, proxy differs', flush=True)
    for arm in ('fd_dml_m4b_gpool_frz_gkd_la', 'fd_dml_m4b_gpool_frz_gkd_kdpx', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx'):
        assert finite(one_epoch(fd, arm, False, batch=128, epochs=2)), arm
        print('2 finite epochs:', arm, flush=True)


def round18(fd, work):
    """Round 18 (cop, pce, t2, g512, gk2, q4_frz_gkd; la prior from SecureSum; compile probe on own rows): r17 arms bit-exact over
    2 epochs (la: reported, its shift now comes from the aggregate); epoch-1 identities of the new tokens; copula bootstrap bytes,
    pooled histogram mass and sampled marginals; all new arms finite over 2 epochs."""
    import importlib.util
    import torch
    from types import SimpleNamespace as NS
    spec = importlib.util.spec_from_file_location('fd_r17', BASE / 'artifacts/input/fd-src-20261015-r17/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    for arm in ('fd_dml_m4b_gpool_frz_gkd', 'fd_dml_m4b_gpool_frz', 'fd_dml_m4b_frz', 'fd_dml_m4b_gpool', 'fd_dml_m4b',
                'fd_dml_m4b_gpool_frz_gkd_kdpx', 'fd_dml_m4b_gpool_q2_frz_gkd', 'fd_dml_m4b_gpool_rej'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r17 fd.py, 2 epochs:', arm, flush=True)
    la_new, la_old = (one_epoch(m, 'fd_dml_m4b_gpool_frz_gkd_la', False, batch=128, epochs=2) for m in (fd, old))
    print('la (shift from the SecureSum prior) vs r17 in-process prior: bit-exact', eq(la_new, la_old),
          'max abs diff', max(float((x.float() - y.float()).abs().max()) for x, y in zip(la_new, la_old)), flush=True)
    private = 20 * len(fd.P.make('dagsnet').state_dict())
    gkd1, gkd2 = (one_epoch(fd, 'fd_dml_m4b_gpool_frz_gkd', False, batch=128, epochs=e) for e in (1, 2))
    for arm, same1, same_private1 in (('t2', True, True), ('gk2', True, True), ('pce', False, True), ('g512', False, False),
                                      ('cop', False, False)):
        a1, a2 = (one_epoch(fd, f'fd_dml_m4b_gpool_frz_gkd_{arm}', False, batch=128, epochs=e) for e in (1, 2))
        assert eq(a1, gkd1) == same1 and eq(a1[:private], gkd1[:private]) == same_private1 and not eq(a2, gkd2) and finite(a2), arm
        print(f'{arm}: epoch 1 == gkd {same1}, epoch-1 privates == gkd {same_private1}; epoch 2 differs, finite', flush=True)
    q1, q2 = (one_epoch(fd, 'fd_dml_m4b_gpool_q4_frz_gkd', False, batch=128, epochs=e) for e in (1, 2))
    assert eq(q1[:private], gkd1[:private]) and not eq(q1[private:], gkd1[private:]) and finite(q2)
    print('q4_frz_gkd: epoch-1 privates == gkd, proxy differs, 2 finite epochs', flush=True)
    a = NS(arm='fd_dml_m4b_gpool_frz_gkd_cop', seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64,
           aux_weight=1., backend='eager', smoke=True, wandb_online=False)
    R = fd.Run(a); m = fd.Method(R); t = m.copula; d = len(t['keep'])
    kinds = {}
    for _, k, way, kind, n in R.bytes: kinds.setdefault((way, kind), set()).add(n)
    print('copula bytes per client', {k: sorted(v) for k, v in kinds.items()}, flush=True)
    assert d == 53 and {(w, k) for w, k in kinds} == {('up', 'copula_hist'), ('down', 'copula_hist'), ('up', 'copula_scores'),
                                                       ('down', 'copula_corr')}
    assert torch.allclose(t['cum'][:, :, -1], torch.ones(16, d, device=R.dev)) and (t['cum'].diff(dim=2) >= 0).all()
    g = torch.Generator(device=R.dev).manual_seed(5); labels = torch.arange(16, device=R.dev).repeat_interleave(4000)
    x = m.draw(labels, g, 0); assert not m.draw_bad and (x[:, R.zero] == 0).all()
    bins = torch.bucketize(torch.asinh(x[:, t['keep']]).clamp(-fd.COP_LIM, fd.COP_LIM - 1e-6), t['edges'][1:-1])
    worst = 0.
    for c in range(16):
        h = torch.stack([torch.bincount(bins[labels == c][:, j], minlength=fd.COP_BINS) for j in range(d)]).float() / 4000
        worst = max(worst, float((h.cumsum(1) - t['cum'][c, :, 1:]).abs().max()))
    assert worst < .03, worst
    print(f'copula draws: masked features 0, sampled marginal CDF within {worst:.4f} of the pooled histogram CDF', flush=True)


def round20(fd, work):
    """Round 20 (la15/la05/lau/lap/kd2 on la_kdpx): r19 arms bit-exact over 2 epochs; la_shift = tau x (log pi_k - target); epoch-1
    identities (lap: privates == la_kdpx, proxy differs; kd2: whole state == la_kdpx; tau/lau: privates differ); 2 finite epochs."""
    import importlib.util
    import math
    import torch
    spec = importlib.util.spec_from_file_location('fd_r19', BASE / 'artifacts/input/fd-src-20261015-r19/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    base = 'fd_dml_m4b_gpool_frz_gkd_la_kdpx'
    for arm in ('fd_dml_m4b_gpool_frz_gkd', base, 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx', 'fd_dml_m4b_gpool_frz', 'fd_dml_m4b'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r19 fd.py, 2 epochs:', arm, flush=True)
    shifts = {}
    def grab(arm):
        def hook(R):
            inner = R.private_epoch
            def wrapped(m, k, r): shifts[arm] = R.la_shift.clone(); return inner(m, k, r)
            R.private_epoch = wrapped
        return hook
    private = 20 * len(fd.P.make('dagsnet').state_dict())
    b1, b2 = (one_epoch(fd, base, False, batch=128, epochs=e, hook=grab(base)) for e in (1, 2))
    for arm, same1, same_private1 in (('fd_dml_m4b_gpool_frz_gkd_la15_kdpx', False, False), ('fd_dml_m4b_gpool_frz_gkd_la05_kdpx', False, False),
                                      ('fd_dml_m4b_gpool_frz_gkd_lau_kdpx', False, False), ('fd_dml_m4b_gpool_frz_gkd_la_lap_kdpx', False, True),
                                      ('fd_dml_m4b_gpool_frz_gkd_la_kdpx_kd2', True, True)):
        a1, a2 = (one_epoch(fd, arm, False, batch=128, epochs=e, hook=grab(arm)) for e in (1, 2))
        assert eq(a1, b1) == same1 and eq(a1[:private], b1[:private]) == same_private1 and not eq(a2, b2) and finite(a2), arm
        print(f'{arm}: epoch 1 == la_kdpx {same1}, epoch-1 privates == la_kdpx {same_private1}; epoch 2 differs, finite', flush=True)
    la = shifts[base]
    assert torch.allclose(shifts['fd_dml_m4b_gpool_frz_gkd_la15_kdpx'], 1.5 * la) and torch.allclose(shifts['fd_dml_m4b_gpool_frz_gkd_la05_kdpx'], .5 * la)
    lau = shifts['fd_dml_m4b_gpool_frz_gkd_lau_kdpx']
    assert torch.allclose(lau - la, (lau - la)[:1].expand_as(la)) and torch.allclose(torch.logsumexp(lau - math.log(16), 1), torch.zeros(20, device=la.device), atol=1e-5)
    print('la_shift: la15 = 1.5 la, la05 = 0.5 la, lau = log pi_k + log 16 (rows of exp(lau)/16 sum to 1)', flush=True)


def round21(fd, work):
    """Round 21 (zl lossless body, ef client uplink error feedback): zl == q4 la_kdpx state over 2 epochs with fewer wire bytes (up and
    down); ef epoch 1 == q4 la_kdpx (zero residual), epoch 2 differs and finite; q2 ef zl finite."""
    import torch
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    runs = {}
    def keep(arm):
        def hook(R): runs[arm] = R
        return hook
    q4 = 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx'
    base = one_epoch(fd, q4, False, batch=128, epochs=2, hook=keep(q4))
    zl = one_epoch(fd, q4 + '_zl', False, batch=128, epochs=2, hook=keep('zl'))
    wire = lambda R, way: sum(n for _, _, w, kind, n in R.bytes if w == way and kind == 'proxy_quantized')
    up, down = (wire(runs[q4], w) for w in ('up', 'down')); upz, downz = (wire(runs['zl'], w) for w in ('up', 'down'))
    assert eq(zl, base) and upz < up and downz < down, (up, upz, down, downz)
    print(f'zl == q4 la_kdpx state over 2 epochs; wire bytes up {up} -> {upz} ({upz / up:.3f}), down {down} -> {downz} ({downz / down:.3f})', flush=True)
    e1, e2 = (one_epoch(fd, q4 + '_ef_zl', False, batch=128, epochs=e) for e in (1, 2))
    assert eq(e1, one_epoch(fd, q4, False, batch=128, epochs=1)) and not eq(e2, base) and finite(e2)
    print('ef: epoch 1 == q4 la_kdpx (zero residual), epoch 2 differs, finite', flush=True)
    q2 = one_epoch(fd, 'fd_dml_m4b_gpool_q2_frz_gkd_la_kdpx_ef_zl', False, batch=128, epochs=2); assert finite(q2)
    print('q2 ef zl: 2 finite epochs', flush=True)


def round22(fd, work):
    """Round 22 (q3: 3-bit both ways; u2: 2-bit uplink, downlink keeps its bits): the 3-bit packet decodes to exactly what quantize()
    produced (raw and zl); u2 uplink packets are smaller than the int4 ones with the same downlink; every arm 2 finite epochs."""
    import torch
    torch.manual_seed(0); model = torch.nn.Sequential(torch.nn.Linear(37, 64), torch.nn.BatchNorm1d(64), torch.nn.Linear(64, 16))
    g = torch.Generator().manual_seed(1)
    for lossless in (False, True):
        packet = fd.QPacket(model, 'q3-test', 3, lossless=lossless); entries, sent = {}, {}
        for name, value in model.state_dict().items():
            if not value.is_floating_point() or fd.is_buffer(name): entries[name] = value
            else: sent[name], q, scale, _ = fd.quantize(torch.randn(value.shape) * 1e-3, 3, g); entries[name] = (q, scale)
        out = packet.decode(packet.encode(entries, 0, 1), 'cpu', 0)
        assert all(torch.equal(out[n], v.reshape(out[n].shape)) for n, v in sent.items()) and int(max(e[0].abs().max() for e in entries.values() if isinstance(e, tuple))) <= 3
    print('q3 packet: decode == quantize() output (raw and zl), codes within +-3', flush=True)
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    runs = {}
    def keep(arm):
        def hook(R): runs[arm] = R
        return hook
    wire = lambda R, way: sum(n for _, _, w, kind, n in R.bytes if w == way and kind == 'proxy_quantized')
    p = 'fd_dml_m4b_gpool_'
    for arm in ('q4_frz_gkd_la_kdpx_zl', 'q4_frz_gkd_la_kdpx_ef_zl', 'q3_frz_gkd_la_kdpx_zl', 'q4_u2_frz_gkd_la_kdpx_ef_zl', 'q3_u2_frz_gkd_la_kdpx_ef_zl'):
        assert finite(one_epoch(fd, p + arm, False, batch=128, epochs=2, hook=keep(arm))), arm
        print(f'{arm}: 2 finite epochs, wire bytes up {wire(runs[arm], "up")} down {wire(runs[arm], "down")}', flush=True)
    b = {arm: (wire(R, 'up'), wire(R, 'down')) for arm, R in runs.items()}
    assert b['q3_frz_gkd_la_kdpx_zl'][0] < b['q4_frz_gkd_la_kdpx_zl'][0] and b['q3_frz_gkd_la_kdpx_zl'][1] < b['q4_frz_gkd_la_kdpx_zl'][1]
    assert b['q4_u2_frz_gkd_la_kdpx_ef_zl'][0] < b['q4_frz_gkd_la_kdpx_ef_zl'][0] and b['q3_u2_frz_gkd_la_kdpx_ef_zl'][0] < b['q3_frz_gkd_la_kdpx_zl'][0]
    print('q3 smaller than q4 both ways; u2 uplink smaller than its full-bit uplink', flush=True)


def round23(fd, work):
    """Round 23 (rs: RMS Lloyd-Max step; old arms stay bit-exact: --part final): with rs the 2-bit step is 1.224 * row RMS (fp16),
    codes within +-1, the packet decodes to exactly what quantize() produced; q2 rs zl 2 finite epochs, fewer wire bytes than q3 zl."""
    import torch
    g = torch.Generator().manual_seed(1); t = torch.randn(64, 37) * 1e-3
    _, q, scale, _ = fd.quantize(t, 2, None, True)
    assert torch.equal(scale.float(), (1.224 * t.pow(2).mean(1, keepdim=True).sqrt()).half().float()) and int(q.abs().max()) <= 1
    torch.manual_seed(0); model = torch.nn.Sequential(torch.nn.Linear(37, 64), torch.nn.BatchNorm1d(64), torch.nn.Linear(64, 16))
    packet = fd.QPacket(model, 'rs-test', 2, lossless=True); entries, sent = {}, {}
    for name, value in model.state_dict().items():
        if not value.is_floating_point() or fd.is_buffer(name): entries[name] = value
        else: sent[name], q, scale, _ = fd.quantize(torch.randn(value.shape) * 1e-3, 2, g, True); entries[name] = (q, scale)
    out = packet.decode(packet.encode(entries, 0, 1), 'cpu', 0)
    assert all(torch.equal(out[n], v.reshape(out[n].shape)) for n, v in sent.items())
    print('rs: step 1.224 * RMS, codes within +-1, packet decode == quantize() output', flush=True)
    runs = {}
    def keep(arm):
        def hook(R): runs[arm] = R
        return hook
    wire = lambda R: sum(n for _, _, w, kind, n in R.bytes if kind == 'proxy_quantized')
    p = 'fd_dml_m4b_gpool_'
    for arm in ('q3_frz_gkd_la_kdpx_zl', 'q2_rs_frz_gkd_la_kdpx_zl'):
        assert all(torch.isfinite(v.float()).all() for v in one_epoch(fd, p + arm, False, batch=128, epochs=2, hook=keep(arm))), arm
        print(f'{arm}: 2 finite epochs, wire bytes {wire(runs[arm])}', flush=True)
    assert wire(runs['q2_rs_frz_gkd_la_kdpx_zl']) < wire(runs['q3_frz_gkd_la_kdpx_zl'])


def round24(fd, work):
    """Round 24 (xo / xou crossover rows, design §Vòng 24): earlier paths bit-exact with the pushed r25 fd.py over 2 epochs; xo and
    xou epoch 1 == the q3_zl control (no snapshot yet, so no crossover rows), epoch 2 differs, finite, xo != xou, logs the pair
    priority; a crossover row is an own row of class a at rho 0 and of class b at rho 1; pair weights live only on held pairs a != b,
    xo keeps the uniform floor, xou is uniform. Compiled main() --workers 1 == 2: --part par --arms <xo arms>."""
    import importlib.util
    import torch
    sys.dont_write_bytecode = True   # importing from the stage must not write __pycache__ into it (upload expects a flat stage)
    spec = importlib.util.spec_from_file_location('fd_r25', BASE / 'artifacts/input/fd-src-20261015-r25/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    p = 'fd_dml_m4b_gpool_'; control = p + 'q3_frz_gkd_la_kdpx_zl'
    for arm in (control, p + 'frz_gkd', 'fd_dml_m4b_gpool', p + 'rej', 'fd_dml_m4b_replay', p + 'frz_gkd_cop'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r25 fd.py, 2 epochs:', arm, flush=True)
    c1, c2 = one_epoch(fd, control, False, batch=128), one_epoch(fd, control, False, batch=128, epochs=2)
    runs, logs = {}, {}
    def keep(arm):
        def hook(R):
            runs[arm] = R; inner = R.private_epoch
            def logged(m, k, r):
                log = inner(m, k, r); logs[arm, k, r] = log; return log
            R.private_epoch = logged
        return hook
    out = {}
    for tok in ('xo', 'xou'):
        assert eq(one_epoch(fd, f'{control}_{tok}', False, batch=128), c1), tok
        out[tok] = one_epoch(fd, f'{control}_{tok}', False, batch=128, epochs=2, hook=keep(tok))
        assert finite(out[tok]) and not eq(out[tok], c2), tok
        assert 'xo_disagree' not in logs[tok, 0, 1] and 0 <= logs[tok, 0, 2]['xo_disagree'] <= 1 and len(logs[tok, 0, 2]['xo_pairs']) == 3, tok
        print(f'{tok}: epoch 1 == control, epoch 2 differs and finite; client 0 epoch 2 crossover disagreement',
              round(logs[tok, 0, 2]['xo_disagree'], 4), 'top pairs', logs[tok, 0, 2]['xo_pairs'], flush=True)
    assert not eq(out['xo'], out['xou'])
    R = runs['xo']; order, start, count = R.xo_index[0]; held = torch.nonzero(count).squeeze(1); a, b = int(held[0]), int(held[-1])
    w = torch.zeros(256, device=R.dev); w[a * 16 + b] = 1; cdf = w.cumsum(0); rho = fd.XO_RHO
    for value, c in ((0., a), (1., b)):
        fd.XO_RHO = value; rows = R.crossover(0, cdf, torch.Generator(device=R.dev).manual_seed(5))
        pool = R.mask(R.X[0][R.Y[0] == c].float())
        assert len(rows) == R.a.aux_batch and bool((torch.cdist(rows, pool, compute_mode='donot_use_mm_for_euclid_dist').min(1).values == 0).all()), (value, c)
    fd.XO_RHO = rho
    pairs = (count > 0).float(); both = (pairs[:, None] * pairs[None, :] * (1 - torch.eye(16, device=R.dev))).flatten()
    for tok in ('xo', 'xou'):
        R.xo = tok; cdf, info = R.xo_weights(0, R.priv[0], R.priv[1]); w = torch.diff(cdf, prepend=cdf.new_zeros(1))
        assert abs(float(cdf[-1]) - 1) < 1e-5 and float(w[both == 0].abs().max()) == 0, (tok, float(cdf[-1]), float(w[both == 0].abs().max()))
        floor = fd.XO_FLOOR / float(both.sum()) if tok == 'xo' else 1 / float(both.sum())
        assert float(w[both > 0].min()) >= floor - 1e-6 and (tok == 'xo' or float(w[both > 0].max()) <= floor + 1e-6), tok
    R.xo = 'xo'
    print('crossover rows: own class-a rows at rho 0, class-b rows at rho 1; pair weights on held pairs only, xo floor, xou uniform', flush=True)


def round25(fd, work):
    """Round 25 (cvf / cvl calibrated teacher head, design §Vòng 25): earlier paths bit-exact with the pushed r28 fd.py over 2 epochs;
    cvf and cvl epoch 1 == the q3_zl control (no snapshot yet), epoch 2 differs, finite, cvf != cvl; the client sums give the pooled
    class means and covariances exactly (vs a direct pooled computation); a head fitted on virtual features of two well separated
    classes separates them; the calibrated head reaches both teachers (frz and gkd) and nothing else of the snapshot changes;
    diagnostics and bytes logged. Compiled main() --workers 1 == 2: --part par --arms <cv arms>."""
    import importlib.util
    import torch
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('fd_r28', BASE / 'artifacts/input/fd-src-20261015-r28/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    p = 'fd_dml_m4b_gpool_'; control = p + 'q3_frz_gkd_la_kdpx_zl'
    for arm in (control, control + '_xo', p + 'frz_gkd', 'fd_dml_m4b_gpool', p + 'rej'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r28 fd.py, 2 epochs:', arm, flush=True)
    c1, c2 = one_epoch(fd, control, False, batch=128), one_epoch(fd, control, False, batch=128, epochs=2)
    runs, logs = {}, {}
    def keep(arm):
        def hook(R):
            runs[arm] = R; inner = R.private_epoch
            def logged(m, k, r):
                log = inner(m, k, r); logs[arm, k, r] = log; return log
            R.private_epoch = logged
        return hook
    out = {}
    for tok in ('cvf', 'cvl'):
        assert eq(one_epoch(fd, f'{control}_{tok}', False, batch=128), c1), tok
        out[tok] = one_epoch(fd, f'{control}_{tok}', False, batch=128, epochs=2, hook=keep(tok))
        assert finite(out[tok]) and not eq(out[tok], c2), tok
        log = logs[tok, 0, 2]; assert 'cv_teach' not in logs[tok, 0, 1] and len(log['cv_cm_cal']) == 16 and log['cv_teach'][0] >= max(log['cv_teach'][1:]), tok
        rows = [b for b in runs[tok].bytes if b[3] in ('cv_stats', 'cv_head')]
        up = 4 * (16 + 16 * 256 + (16 if tok == 'cvf' else 1) * 256 * 257 // 2)
        assert sorted({(b[0], b[2], b[4]) for b in rows}) == [(2, 'down', 4 * (16 * 256 + 16)), (2, 'up', up)] and len(rows) == 2 * runs[tok].K, (tok, rows[:4])
        print(f'{tok}: epoch 1 == control, epoch 2 differs and finite; up {up} B down {4 * (16 * 256 + 16)} B per client; client 0 own-row teach',
              log['cv_teach'], flush=True)
    assert not eq(out['cvf'], out['cvl'])
    R = runs['cvf']; m = fd.Method(R, initialize=False); state = {n: v.clone() for n, v in R.init.state_dict().items()}
    stats = [R.cv_stats(state, k) for k in range(R.K)]
    hs, ys = zip(*(R.cv_features(state, k) for k in range(R.K))); h, y = torch.cat(hs).double(), torch.cat(ys)
    n = sum(v['n'] for v in stats); mu = sum(v['s'] for v in stats) / n[:, None]
    for c in range(16):
        if n[c] < 3: continue
        hc = h[y == c]; assert torch.allclose(mu[c], hc.mean(0).cpu(), atol=1e-8)
        cov = (stats[0]['ss'][c] * 0 + sum(v['ss'][c] for v in stats) - n[c] * mu[c][:, None] * mu[c][None, :]) / (n[c] - 1)
        assert torch.allclose(cov, torch.cov(hc.T).cpu(), atol=1e-6, rtol=1e-6), c
    print('client sums -> pooled class means and covariances == direct pooled computation', flush=True)
    g = torch.Generator(device=R.dev).manual_seed(3); toy = []
    for c in range(16):
        base = torch.zeros(256, dtype=torch.float64, device=R.dev); base[c] = 6.
        x = base + torch.randn(64, 256, dtype=torch.float64, device=R.dev, generator=g)
        toy.append(dict(n=torch.zeros(16, dtype=torch.float64).index_fill_(0, torch.tensor([c]), 64.),
                        s=torch.zeros(16, 256, dtype=torch.float64).index_copy_(0, torch.tensor([c]), x.sum(0, keepdim=True).cpu()),
                        ss=torch.zeros(16, 256, 256, dtype=torch.float64).index_copy_(0, torch.tensor([c]), (x.T @ x)[None].cpu())))
    for tok in ('cvf', 'cvl'):
        R.cv = tok
        if tok == 'cvl':
            for v in toy: v['ss'] = v['ss'].sum(0) if v['ss'].dim() == 3 else v['ss']
        head = R.cv_calibrate(toy, state, 2)
        probe = torch.zeros(16, 256, device=R.dev); probe[torch.arange(16), torch.arange(16)] = 6.
        pred = (probe @ head['head.5.weight'].float().T + head['head.5.bias'].float()).argmax(1)
        assert torch.equal(pred, torch.arange(16, device=R.dev)), (tok, pred)
    R.cv = 'cvf'
    print('virtual-feature head fit separates 16 separated toy classes (cvf, cvl)', flush=True)
    teacher, syn = R.work_teacher, R.syn_teacher; m.global_state = state; m.cv_head = head; m.version = 1
    for model in (teacher, syn):
        model.load_state_dict(state, strict=True); R.cv_install(model, head)
        for name, v in model.state_dict().items():
            same = torch.equal(v, head[name]) if name in head else torch.equal(v, state[name])
            assert same, name
    print('cv_install: head.5 replaced, every other snapshot tensor unchanged (frz and gkd teachers)', flush=True)


def round26(fd, work):
    """Round 26 (cvp: cv refit in a 48-d projection, probe 36; with cop): earlier paths bit-exact with the pushed r30 fd.py over 2 epochs;
    cvp epoch 1 == the q3_zl control (no snapshot yet), epoch 2 differs and is finite; cvp + cop finite and != cvp; bytes 78,400 B up and
    the 256-d head down per client per calibrated round; the projected client sums give the pooled projected class means/covariances;
    the basis is orthonormal, deterministic and holds the snapshot head (a zero-step fit returns the snapshot head); the calibrated head
    lies in the projection. Compiled main() --workers 1 == 2: --part par --arms <cvp arms>."""
    import importlib.util
    import torch
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('fd_r30', BASE / 'artifacts/input/fd-src-20261015-r30/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    p = 'fd_dml_m4b_gpool_'; control = p + 'q3_frz_gkd_la_kdpx_zl'
    for arm in (control, control + '_cvl', control + '_cvf', p + 'frz_gkd_cop'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r30 fd.py, 2 epochs:', arm, flush=True)
    c1, c2 = one_epoch(fd, control, False, batch=128), one_epoch(fd, control, False, batch=128, epochs=2)
    runs = {}
    def keep(arm):
        def hook(R): runs[arm] = R
        return hook
    assert eq(one_epoch(fd, control + '_cvp', False, batch=128), c1)
    cvp = one_epoch(fd, control + '_cvp', False, batch=128, epochs=2, hook=keep('cvp'))
    assert finite(cvp) and not eq(cvp, c2)
    both = one_epoch(fd, control + '_cvp_cop', False, batch=128, epochs=2, hook=keep('both'))
    assert finite(both) and not eq(both, cvp)
    for tag in ('cvp', 'both'):
        rows = [b for b in runs[tag].bytes if b[3] in ('cv_stats', 'cv_head')]
        assert sorted({(b[0], b[2], b[4]) for b in rows}) == [(2, 'down', 4 * (16 * 256 + 16)), (2, 'up', 78400)] and len(rows) == 2 * runs[tag].K, rows[:4]
    print('cvp: epoch 1 == control, epoch 2 differs and finite; cvp+cop finite and != cvp; up 78400 B down 16448 B per client', flush=True)
    R = runs['cvp']; state = {n: v.clone() for n, v in R.init.state_dict().items()}
    basis = R.cv_basis(state); assert torch.equal(basis, R.cv_basis(state)) and basis.shape == (256, fd.CV_PROJ)
    assert torch.allclose(basis.T @ basis, torch.eye(fd.CV_PROJ, dtype=basis.dtype, device=basis.device), atol=1e-10)
    w = state['head.5.weight'].double(); assert torch.allclose(w @ basis @ basis.T, w, atol=1e-10)
    stats = [R.cv_stats(state, k) for k in range(R.K)]
    hs, ys = zip(*(R.cv_features(state, k) for k in range(R.K))); h, y = (torch.cat(hs).double() @ basis).cpu(), torch.cat(ys).cpu()
    n = sum(v['n'] for v in stats); mu = sum(v['s'] for v in stats) / n[:, None]
    for c in range(16):
        if n[c] < 3: continue
        hc = h[y == c]; assert torch.allclose(mu[c], hc.mean(0), atol=1e-8)
        cov = (sum(v['ss'][c] for v in stats) - n[c] * mu[c][:, None] * mu[c][None, :]) / (n[c] - 1)
        assert torch.allclose(cov, torch.cov(hc.T), atol=1e-6, rtol=1e-6), c
    print('projected client sums -> pooled projected class means and covariances == direct computation; basis orthonormal, holds the head', flush=True)
    steps = fd.CV_STEPS; fd.CV_STEPS = 0
    try: head0 = R.cv_calibrate(stats, state, 2)
    finally: fd.CV_STEPS = steps
    assert torch.allclose(head0['head.5.weight'].float(), state['head.5.weight'].float(), atol=1e-5)
    assert torch.equal(head0['head.5.bias'], state['head.5.bias'])
    head = R.cv_calibrate(stats, state, 2); hw = head['head.5.weight'].double()
    assert torch.isfinite(hw).all() and torch.allclose(hw @ basis @ basis.T, hw, atol=1e-5) and not torch.allclose(hw, w, atol=1e-5)
    print('zero-step cvp fit returns the snapshot head; the fitted head lies in the projection and differs', flush=True)


def round32(fd):
    """m4b validation: full66 stays 66; drop53 removes BLOCK+POSITION and pads only inside DAGSNet."""
    import inspect
    import torch
    names = [*fd.BLOCK, *fd.POSITION, *[f'keep_{i}' for i in range(53)]]
    x = torch.arange(2 * 66, dtype=torch.float32).view(2, 66)
    R = fd.Run.__new__(fd.Run); R.drop53 = True
    dropped = set(fd.BLOCK + fd.POSITION)
    R.keep = torch.tensor([i for i, name in enumerate(names) if name not in dropped])
    R.zero = torch.empty(0, dtype=torch.long); R.zero_cols = torch.zeros(66, dtype=torch.bool)
    z = R.mask(x)
    assert z.shape == (2, 53) and torch.equal(z, x[:, 13:])
    cfg = dict(fd.P.CFG, n_features=53, num_classes=fd.P.C, input_pad=1)
    model = fd.P.build_model(cfg, latent=True).eval()
    with torch.no_grad(): logits, latent = model(z)
    assert logits.shape == (2, 16) and latent.shape == (2, 256)
    assert model.input_features == 53 and model.input_pad == 1 and model.k == 9
    assert sum(p.numel() for p in model.parameters()) == 395024
    assert 'fd_dml_full66_gpool_q3_frz_gkd_la_kdpx_zl' in fd.ARMS
    assert 'fd_dml_drop53_gpool_q3_frz_gkd_la_kdpx_zl' in fd.ARMS
    assert 'if self.drop53 or len(self.zero): x = self.mask(x)' in inspect.getsource(fd.Run.compile_extra)
    print('round 32: full66 registered; drop53 removes 13 inputs; DAGSNet sees 53 then pads to 9x6; forward PASS', flush=True)


def round33(fd):
    """FedAvg feature arms share algorithm/init and differ only in the declared input transform."""
    import torch
    arms = ('fd_fedavg_full66', 'fd_fedavg_drop53', 'fd_fedavg_m4b')
    assert all(arm in fd.ARMS for arm in arms)
    names = [*fd.BLOCK, *fd.POSITION, *[f'keep_{i}' for i in range(53)]]
    x = torch.arange(2 * 66, dtype=torch.float32).view(2, 66)
    dropped = set(fd.BLOCK + fd.POSITION)
    keep = torch.tensor([i for i, name in enumerate(names) if name not in dropped])

    def transform(arm):
        R = fd.Run.__new__(fd.Run); tokens = arm.split('_'); R.drop53 = 'drop53' in tokens
        R.keep = keep; hidden = fd.BLOCK + fd.POSITION if 'm4b' in tokens else []
        R.zero = torch.tensor([names.index(name) for name in hidden], dtype=torch.long)
        R.zero_cols = torch.zeros(66, dtype=torch.bool); R.zero_cols[R.zero] = True
        return R.mask(x)

    full, drop, mask = map(transform, arms)
    assert torch.equal(full, x) and drop.shape == (2, 53) and torch.equal(drop, x[:, 13:])
    assert mask.shape == x.shape and torch.count_nonzero(mask[:, :13]) == 0 and torch.equal(mask[:, 13:], x[:, 13:])
    cfg = dict(fd.P.CFG, n_features=53, num_classes=fd.P.C, input_pad=1)
    model = fd.P.build_model(cfg, latent=True).eval()
    with torch.no_grad(): logits, latent = model(drop)
    assert logits.shape == (2, 16) and latent.shape == (2, 256) and model.input_features == 53 and model.input_pad == 1
    print('round 33: FedAvg full66/drop53/m4b transforms and drop53 forward PASS', flush=True)


def round34(fd):
    """Four remaining legacy methods expose exactly three synchronized feature arms."""
    import inspect
    methods = ('fedprox', 'fedtgp', 'canperfl', 'centralized')
    modes = ('full66', 'drop53', 'm4b')
    expected = {f'fd_{method}_{mode}' for method in methods for mode in modes}
    assert expected <= set(fd.ARMS) and len(expected) == 12
    assert fd.MU == .01 and fd.TGP_DIM == 544 and fd.TGP_LAMBDA == 1.0
    assert (fd.TGP_STEPS, fd.TGP_LR, fd.TGP_TAU) == (100, 1e-3, 100.)
    assert (fd.PERSONAL_EPOCHS, fd.PERSONAL_MAX_ROWS) == (10, 30_000)
    centralized = inspect.getsource(fd.Run.centralized_epoch)
    assert 'batches += 1' in centralized and 'skip_budget = 16 * self.K' in centralized
    print('round 34: 12 arms registered; FedProx/FedTGP/CanPerFL constants locked; centralized registered', flush=True)


def resume(proj, work):
    """Codec resume (round 21): a quantized-wire run with client error feedback continued from its committed epoch 1 (resume bundle:
    server exact model + every client's residual) == the uninterrupted run at epoch 2 (weights, predictions, eval, round-2 bytes),
    serial and --workers 2."""
    import json
    import os
    import numpy as np
    import torch
    arm = 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_ef_zl'
    env = {**os.environ, 'PYTHONPATH': f'{proj}:{ROOT}'}
    def run(out, *extra):
        if (out / f'{arm}_s42').exists(): shutil.rmtree(out / f'{arm}_s42')
        cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--backend', 'eager', '--batch', '128', '--out', str(out), *extra]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:], [(f.name, f.read_text()[-1500:]) for f in (out / f'{arm}_s42').glob('worker*.log')])
        return out / f'{arm}_s42'
    a = run(work / 'resume_a')
    ra = json.loads((a / 'complete/round_001.json').read_text()); assert 'resume_sha' in ra and (a / 'resume/round_001.pt').exists()
    for workers in (1, 2):
        b = run(work / f'resume_b{workers}', '--parent', str(a), '--parent_round', '1', '--workers', str(workers))
        wa, wb = (torch.load(d / 'weights/round_002.pt') for d in (a, b))
        assert all(torch.equal(x, y) for s, t in zip(wa['private'] + wa['proxy'], wb['private'] + wb['proxy'], strict=True)
                   for x, y in zip(s.values(), t.values(), strict=True))
        x, y = (np.load(d / 'predictions/round_002.npz') for d in (a, b))
        assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
        ea, eb = (json.loads((d / 'complete/round_002.json').read_text()) for d in (a, b))
        assert ea['eval'] == eb['eval']
        za, zb = (torch.load(d / 'resume/round_002.pt') for d in (a, b))   # fingerprints differ (the continuation's config names its parent)
        assert za['exact'].keys() == zb['exact'].keys() and all(torch.equal(za['exact'][n], zb['exact'][n]) for n in za['exact'])
        assert za['ef'].keys() == zb['ef'].keys() and len(za['ef']) and all(torch.equal(za['ef'][n], zb['ef'][n]) for n in za['ef'])
        byte_a = [row for row in json.loads((a / 'result.json').read_text())['bytes'] if row[0] == 2]
        assert byte_a == json.loads((b / 'result.json').read_text())['bytes']
        print(f'RESUME q4 ef zl r1 -> r2 (workers {workers}) == uninterrupted: weights, predictions, eval, resume bundle, round-2 bytes', flush=True)


def final(fd, proj, work):
    """--final (owner 19/10, protocol §3): without the flag every path is bit-exact with r20 fd.py over 2 epochs; with it main() scores only
    the official test set after EVERY round (compiled, --fast, --workers 2), config final=True; a --final run continues only a --final
    parent and the chain completes."""
    import importlib.util
    import json
    import os
    import torch
    spec = importlib.util.spec_from_file_location('fd_r20', BASE / 'artifacts/input/fd-src-20261015-r20/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b_gpool_frz_gkd', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_ef_zl'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT without --final vs r20 fd.py, 2 epochs:', arm, flush=True)
    arm = 'fd_dml_m4b_gpool_frz_gkd_la_kdpx'
    env = {**os.environ, 'PYTHONPATH': f'{proj}:{ROOT}', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}
    def run(out, *extra, ok=True):
        if (out / f'{arm}_s42').exists(): shutil.rmtree(out / f'{arm}_s42')
        cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--batch', '128', '--out', str(out), *extra]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, (r.stdout[-2000:], r.stderr[-3000:])
        return out / f'{arm}_s42', r
    a, _ = run(work / 'final_a', '--final', '--fast', '--backend', 'compiled', '--workers', '2')
    cfg, res = (json.loads((a / f).read_text()) for f in ('config.json', 'result.json'))
    assert cfg['config']['final'] is True and res['complete'] and len(res['history']) == 2
    for row in res['history']:
        assert {'private', 'proxy'} <= set(row['eval']) and all(sorted(h) == ['test'] for h in row['eval'].values()), row['eval'].keys()
    assert {line.split(',')[2] for line in (a / 'metrics.csv').read_text().splitlines()[1:]} == {'test'}
    print('--final main(): test scored after every round (rounds 1-2), no group/old, config final=True', flush=True)
    b, _ = run(work / 'final_b', '--final', '--fast', '--backend', 'compiled', '--parent', str(a), '--parent_round', '1')
    rb = json.loads((b / 'result.json').read_text()); assert rb['complete'] and len(rb['history']) == 1 and 'test' in rb['history'][0]['eval']['private']
    _, r = run(work / 'final_c', '--fast', '--backend', 'compiled', '--parent', str(a), '--parent_round', '1', ok=False)
    assert 'parent config differs' in r.stderr
    print('--final parent chain: final -> final continuation completes; non-final continuation of a final parent refused', flush=True)


def fast(fd, proj, work):
    """Round 19 --fast (owner 2026-10-18, tolerance class: fused AdamW + nondeterministic cuDNN). Without the flag every path is
    bit-exact with r18 fd.py over 2 epochs; with it one step per client from the same state changes private logits at rounding level
    (fused alone < 1e-3, + nondet < 3e-2 of max|logit|) with no argmax flip on decisive rows and the same applied/skipped steps;
    2 epochs stay finite; compiled main() --fast --workers 2 completes."""
    import importlib.util
    import json
    import os
    import torch
    spec = importlib.util.spec_from_file_location('fd_r18', BASE / 'artifacts/input/fd-src-20261015-r18/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    finite = lambda s: all(torch.isfinite(v.float()).all() for v in s)
    arms = ('fd_dml_m4b_gpool_frz_gkd', 'fd_dml_m4b_gpool_frz', 'fd_dml_m4b', 'fd_dml_m4b_sgd3', 'fd_dml_m4b_gpool_frz_gkd_cop')
    for arm in arms:
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT without --fast vs r18 fd.py, 2 epochs:', arm, flush=True)
    def run(arm, fused, nondet, batch, epochs):
        logs, keep = [], {}
        def hook(R):
            R.fused = fused or None; keep['R'] = R; inner = R.private_epoch
            def counted(m, k, r):
                log = inner(m, k, r); logs.append((k, r, log['applied'], log['skipped'])); return log
            R.private_epoch = counted
        fd.deterministic(nondet)
        state = one_epoch(fd, arm, False, batch=batch, epochs=epochs, hook=hook); R = keep['R']; logits = []
        with torch.no_grad():
            for k, model in enumerate(R.priv):
                model.eval(); logits.append(model(R.mask(R.X[k][:1024].float()))[0].float()); model.train()
        return state, logs, logits
    def compare(a, b):   # max|logit diff| / max|logit| over clients; argmax flips on decisive rows (top-2 gap > 10 x client max diff)
        rel, flips = 0., 0
        for x, y in zip(a, b):
            d = float((x - y).abs().max()); top = y.topk(2, 1).values; decisive = top[:, 0] - top[:, 1] > 10 * d
            rel = max(rel, d / float(y.abs().max())); flips += int((x.argmax(1) != y.argmax(1))[decisive].sum())
        return rel, flips
    # Tensor-wise relative diffs are no criterion: Adam's first update is about +-lr per coordinate, so BN biases with ~0 true
    # gradient take a rounding-noise sign (local: 0.5-0.7 relative on such biases after one nondet step, logits 0.8 %).
    for arm in arms[:4]:
        _, lb, gb = run(arm, False, False, 1024, 1)   # one step per client from the same state (1024 smoke rows, batch 1024)
        _, lf, gf = run(arm, True, False, 1024, 1); _, lq, gq = run(arm, True, True, 1024, 1)
        (rf, ff), (rq, fq) = compare(gf, gb), compare(gq, gb)
        assert lb == lf == lq and rf < 1e-3 and ff == 0 and rq < 3e-2 and fq == 0, (arm, rf, ff, rq, fq, lb, lq)
        two, l2, _ = run(arm, True, True, 128, 2); assert finite(two), arm
        print(f'--fast {arm}: one step logits fused {rf:.1e} / fused+nondet {rq:.1e} of max|logit|, 0 decisive argmax flips, same '
              f'applied/skipped; 2 epochs x 8 steps finite, skipped {sum(x[3] for x in l2)}', flush=True)
    fd.deterministic(False)
    base = work / 'fastrun'; arm = 'fd_dml_m4b_gpool_frz_gkd'; out = base / f'{arm}_s42'
    if out.exists(): shutil.rmtree(out)
    cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--backend', 'compiled', '--batch', '128',
           '--out', str(base), '--workers', '2', '--fast']
    env = {**os.environ, 'PYTHONPATH': f'{proj}:{ROOT}', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:], [(f.name, f.read_text()[-2000:]) for f in out.glob('worker*.log')])
    cfg, res = (json.loads((out / f).read_text()) for f in ('config.json', 'result.json'))
    assert cfg['config']['fast'] is True and res['complete'] and len(res['history']) == 2
    print('compiled main() --fast --workers 2: complete, config fast=True', flush=True)


def parent(proj, work):
    """--parent: epoch 2 continued from the committed epoch 1 of a run made by the round-16 source == epoch 2 of that run
    (weights, predictions, eval) -- serial and --workers 2 (every worker loads the parent privates)."""
    import json
    import os
    import numpy as np
    import torch
    old = work / 'proj16'
    if old.exists(): shutil.rmtree(old)
    shutil.copytree(proj, old); shutil.copyfile(BASE / 'artifacts/input/fd-src-20261015-r16/fd.py', old / 'fd.py')
    arm = 'fd_dml_m4b_gpool_frz_gkd'
    def run(source, out, *extra):
        cmd = [sys.executable, str(source / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--backend', 'eager', '--batch', '128',
               '--out', str(out), *extra]
        if (out / f'{arm}_s42').exists(): shutil.rmtree(out / f'{arm}_s42')
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'PYTHONPATH': f'{source}:{ROOT}'}, capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-3000:])
        return out / f'{arm}_s42'
    a = run(old, work / 'parent_a')
    for workers in (1, 2):   # local RAM (3 workers + eager parent run exceed ~9 GB)
        b = run(proj, work / f'parent_b{workers}', '--parent', str(a), '--parent_round', '1', '--workers', str(workers))
        assert sorted(p.name for p in (b / 'weights').glob('*.pt')) == ['round_002.pt']
        wa, wb = (torch.load(d / 'weights/round_002.pt') for d in (a, b))
        assert all(torch.equal(x, y) for s, t in zip(wa['private'] + wa['proxy'], wb['private'] + wb['proxy'], strict=True)
                   for x, y in zip(s.values(), t.values(), strict=True))
        x, y = (np.load(d / 'predictions/round_002.npz') for d in (a, b))
        assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
        ra, rb = (json.loads((d / 'complete/round_002.json').read_text()) for d in (a, b))
        assert ra['eval'] == rb['eval'] and rb['prev_sha'] == ra['prev_sha'] and rb['version'] == ra['version'] == 2
        cb = json.loads((b / 'config.json').read_text())['config']
        assert cb['parent']['round'] == 1 and cb['parent']['weights_sha'] == ra['prev_sha']
        res = json.loads((b / 'result.json').read_text()); assert res['complete'] and len(res['history']) == 1
        print(f'PARENT r1 -> r2 (r16 parent, new source, workers {workers}) == uninterrupted r16 run: weights, predictions, eval', flush=True)
    # round 19: --fast continues only a --fast parent (config key 'fast'); a run without the flag is refused by the parent config check
    fa = run(proj, work / 'parent_fa', '--fast')
    fb = run(proj, work / 'parent_fb', '--parent', str(fa), '--parent_round', '1', '--fast', '--workers', '2')
    cb = json.loads((fb / 'config.json').read_text())['config']; res = json.loads((fb / 'result.json').read_text())
    assert cb['fast'] is True and cb['parent']['round'] == 1 and res['complete'] and len(res['history']) == 1
    if (work / 'parent_fc' / f'{arm}_s42').exists(): shutil.rmtree(work / 'parent_fc' / f'{arm}_s42')
    r = subprocess.run([sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--backend', 'eager', '--batch', '128',
                        '--out', str(work / 'parent_fc'), '--parent', str(fa), '--parent_round', '1'],
                       cwd=ROOT, env={**os.environ, 'PYTHONPATH': f'{proj}:{ROOT}'}, capture_output=True, text=True)
    assert r.returncode != 0 and 'parent config differs' in r.stderr, r.stderr[-1500:]
    print('PARENT --fast: fast parent -> fast continuation (workers 2) completes; non-fast continuation of a fast parent refused', flush=True)


def round13(fd, work):
    """Round 13 hooks leave every round-12 arm bit-exact over 2 epochs; rej/p1/p1self/p1cm epoch 1 == gpool (teacher off before
    the first aggregated snapshot) and differ at epoch 2 with sane teacher diagnostics; fill water-fills the labels; QPacket
    2-bit decode == quantize(), corrupted packets refused; gpool_q2 round-1 privates == gpool; gpe prior = (pi + w q) / (1 + w)."""
    import importlib.util
    import torch
    from types import SimpleNamespace as NS
    spec = importlib.util.spec_from_file_location('fd_r12', BASE / 'artifacts/input/fd-src-20261015-r12/fd.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    eq = lambda a, b: len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    for arm in ('fd_dml_m4b_gpool', 'fd_dml_m4b_gpool_q4', 'fd_dml_m4b_gpool_kd2', 'fd_dml_m4b_gpool_gw2', 'fd_dml_m4b',
                'fd_gaussian_m4b', 'fd_dml_swappos', 'fd_dml_m4b_replay', 'fd_dml_m4b_own', 'fd_dml_m4b_gown'):
        assert eq(one_epoch(fd, arm, False, batch=128, epochs=2), one_epoch(old, arm, False, batch=128, epochs=2)), arm
        print('BIT-EXACT vs r12 fd.py, 2 epochs:', arm, flush=True)
    gpool1, gpool2 = (one_epoch(fd, 'fd_dml_m4b_gpool', False, batch=128, epochs=e) for e in (1, 2))
    for arm in ('fd_dml_m4b_gpool_rej', 'fd_dml_m4b_gpool_p1', 'fd_dml_m4b_gpool_p1self', 'fd_dml_m4b_gpool_p1cm'):
        assert eq(one_epoch(fd, arm, False, batch=128), gpool1), arm
        two = one_epoch(fd, arm, False, batch=128, epochs=2)
        assert not eq(two, gpool2) and all(torch.isfinite(v.float()).all() for v in two), arm
        print(arm, 'epoch 1 == gpool (no snapshot yet), epoch 2 differs, finite', flush=True)
    base = dict(seed=42, rounds=50, batch=128, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=64, aux_weight=1., backend='eager',
                smoke=True, wandb_online=False)
    for arm in ('fd_dml_m4b_gpool_rej', 'fd_dml_m4b_gpool_p1', 'fd_dml_m4b_gpool_p1cm'):   # epoch-2 teacher diagnostics
        R = fd.Run(NS(arm=arm, **base)); m = fd.Method(R); m.prior()
        for r in (1, 2):
            m.start_round(); logs = []
            for k in range(R.K):
                m.prepare(k, r); logs.append(R.private_epoch(m, k, r)); packet, _ = m.proxy_epoch(k, r); m.aggregate_sum.add(k, packet)
            m.aggregate(r)
        for log in logs:
            assert 0 <= log['syn_gate'] <= log['syn_accept'] <= 1 and len(log['synacc']) == 16 and all(0 <= v <= 1 for v in log['synacc'])
            assert ('syn_kl' in log) == (arm != 'fd_dml_m4b_gpool_rej') and log.get('syn_kl', 0) >= 0
        print(arm, 'epoch 2 acceptance', round(sum(l['syn_accept'] for l in logs) / R.K, 3), 'gate',
              round(sum(l['syn_gate'] for l in logs) / R.K, 3), 'kl', logs[0].get('syn_kl'), flush=True)
    labels = fd.fill_labels(torch.tensor([.5, .3, .1, .05, .05]), 10)
    assert labels.tolist() == [1, 2, 2, 2, 3, 3, 3, 4, 4, 4], labels.tolist()   # level 0.375: q = 0,.075,.275,.325,.325
    R = fd.Run(NS(arm='fd_dml_m4b_gpool_fill', **base))
    for k in range(R.K):
        pi, n = R.log_pi[k].exp().double(), torch.bincount(R.aux_labels[k], minlength=16).double().cpu()
        assert int(n.sum()) == 64 and (n[pi.argsort()[-1]] == 0)   # the largest local class gets no Gaussian rows
        q = n / 64; pe = (pi.cpu() + q) / 2
        assert torch.allclose(R.log_pe[k].double().cpu(), pe.log(), atol=1e-6)
    gp = fd.Run(NS(arm='fd_dml_m4b_gpool', **base))
    assert torch.allclose(gp.log_pe.exp(), (gp.log_pi.exp() + 1 / 16) / 2, atol=1e-6) and gp.rules == ('head', 'gp', 'gpe')
    assert fd.Run(NS(arm='fd_dml_m4b', **base)).rules == ('head', 'gp')
    fill = one_epoch(fd, 'fd_dml_m4b_gpool_fill', False, batch=128)
    assert not eq(fill, gpool1) and all(torch.isfinite(v.float()).all() for v in fill)
    print('fill: water-filled labels (largest local class 0 rows), differs from gpool; gpe prior checked', flush=True)
    R = fd.Run(NS(arm='fd_dml_m4b_gpool_q2', **base)); m = fd.Method(R); m.prior(); g = torch.Generator(device=R.dev).manual_seed(5)
    entries, expect = {}, {}
    for name, v in m.proxy[0].state_dict().items():
        if not v.is_floating_point(): entries[name] = expect[name] = v
        elif fd.is_buffer(name): entries[name] = v; expect[name] = v.half().float()
        else:
            wire, q, scale, _ = fd.quantize(0.01 * torch.randn(v.shape, device=R.dev, generator=g), 2, g)
            assert set(q.unique().tolist()) <= {-1, 0, 1}; entries[name] = (q, scale); expect[name] = wire
    packet = m.qpacket.encode(entries, 3, 4); got = m.qpacket.decode(packet, R.dev, 3)
    assert got.keys() == expect.keys() and all(torch.equal(got[n], expect[n]) for n in expect)
    bad = bytearray(packet); bad[-1] ^= 1
    for wrong, b in ((bytes(bad), 3), (packet, 2)):
        try: m.qpacket.decode(wrong, R.dev, b)
        except AssertionError: pass
        else: raise SystemExit('QPacket q2 accepted a corrupted/stale packet')
    q4 = fd.QPacket(m.proxy[0], 'x', 4)
    try: q4.decode(packet, R.dev, 3)
    except AssertionError: pass
    else: raise SystemExit('int4 decoder accepted a 2-bit packet')
    q2 = one_epoch(fd, 'fd_dml_m4b_gpool_q2', False, batch=128)
    private = 20 * len(fd.P.make('dagsnet').state_dict())
    assert eq(gpool1[:private], q2[:private]) and not eq(gpool1[private:], q2[private:])
    R = fd.Run(NS(arm='fd_dml_m4b_gpool_q2', **base)); m = fd.Method(R); m.prior(); m.start_round()
    for k in range(R.K):
        m.prepare(k, 1); R.private_epoch(m, k, 1); m.proxy_epoch(k, 1)
    m.aggregate(1)
    rows = [b for b in R.bytes if b[0] == 1 and b[3] == 'proxy_quantized']
    assert len(rows) == 2 * R.K and len({b[4] for b in rows}) == 1 and rows[0][4] == len(m.latest_packet) == len(packet)
    two = one_epoch(fd, 'fd_dml_m4b_gpool_q2', False, batch=128, epochs=2); assert all(torch.isfinite(v.float()).all() for v in two)
    print('QPacket 2-bit: decode == quantize(), CRC/base/bits checked; gpool_q2 round-1 privates == gpool, proxy differs;',
          len(packet), 'B per direction; q_residual_rel', round(m.diag['q_residual_rel'], 4), flush=True)


def parallel(proj, work, only=None):
    """--workers 3 (every rank on the local GPU) == --workers 1 on a 2-epoch smoke main(): weights, predictions, byte log,
    history (timings and file hashes aside) and metrics.csv; compiled backend for the headline, eager for the other paths."""
    import json
    import os
    import numpy as np
    import torch
    skip = {'sec', 'seconds', 'algorithm_seconds', 'predictions_sha', 'weights_sha', 'prev_sha', 'resume_sha'}   # resume: contents below
    skip |= {'gpu_peak_allocated_bytes', 'payload_bytes_per_algorithm_second'}   # round 24: process layout / wall time (cost block)
    strip = lambda v: ({k: strip(x) for k, x in v.items() if k not in skip and not k.endswith('_seconds')} if isinstance(v, dict) else
                       [strip(x) for x in v] if isinstance(v, list) else v)
    def same(x, y):
        if isinstance(x, torch.Tensor): return torch.equal(x, y)
        if isinstance(x, dict): return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        if isinstance(x, list): return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
        return x == y
    ran = set()
    for arm, backend in (('fd_dml_m4b_gpool', 'compiled'), ('fd_dml_m4b_q4', 'eager'), ('fd_gaussian_m4b', 'eager'),
                         ('fd_fedavg', 'eager'), ('fd_dml_m4b_own', 'eager'), ('fd_dml_swappos', 'eager'), ('fd_dml_m4b_replay', 'eager'),
                         ('fd_dml_m4b_gpool_p1self', 'eager'), ('fd_dml_m4b_gpool_rej', 'eager'), ('fd_dml_m4b_gpool_fill', 'eager'),
                         ('fd_dml_m4b_gpool_frz_gkd', 'compiled'), ('fd_dml_m4b_frz', 'eager'),
                         ('fd_dml_m4b_gpool_frz_gkd_la_kdpx', 'compiled'), ('fd_dml_m4b_gpool_frz_gkd_cop', 'compiled'),
                         ('fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx', 'compiled'), ('fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_ef_zl', 'compiled'),
                         ('fd_dml_m4b_gpool_q3_u2_frz_gkd_la_kdpx_ef_zl', 'compiled'), ('fd_dml_m4b_gpool_q2_rs_frz_gkd_la_kdpx_zl', 'compiled'),
                         ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xo', 'compiled'), ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xou', 'compiled'),
                         ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvf', 'compiled'), ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvl', 'compiled'),
                         ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvp', 'compiled'), ('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvp_cop', 'compiled')):
        if only and arm not in only: continue
        ran.add(arm); runs = {}
        for workers in ((1, 2) if {'cop', 'q4', 'q3', 'q2'} & set(arm.split('_')) else (1, 3)):   # local RAM: cop (copula per rank), q4+la_kdpx compiled
            base = work / f'par{workers}'; out = base / f'{arm}_s42'
            if out.exists(): shutil.rmtree(out)
            cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '42', '--smoke', '--backend', backend, '--batch', '128',
                   '--out', str(base), '--workers', str(workers)]
            env = {**os.environ, 'PYTHONPATH': f'{proj}:{ROOT}', 'TORCHINDUCTOR_COMPILE_THREADS': '1'}   # as on Kaggle; RAM
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            assert r.returncode == 0, (arm, workers, r.stdout[-2000:], r.stderr[-3000:],
                                       [(f.name, f.read_text()[-2000:]) for f in out.glob('worker*.log')])
            runs[workers] = out
        a, b = runs[1], runs[max(runs)]
        for f in sorted((a / 'weights').glob('*.pt')):
            assert same(torch.load(f), torch.load(b / 'weights' / f.name)), (arm, f.name)
        for f in sorted((a / 'resume').glob('*.pt')):
            assert same(torch.load(f), torch.load(b / 'resume' / f.name)), (arm, f.name)
        for f in sorted((a / 'predictions').glob('*.npz')):
            x, y = np.load(f), np.load(b / 'predictions' / f.name)
            assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files), (arm, f.name)
        ra, rb = (json.loads((d / 'result.json').read_text()) for d in (a, b))
        assert ra['bytes'] == rb['bytes'] and strip(ra['history']) == strip(rb['history']) and ra['complete'] and rb['complete'], arm
        assert (a / 'metrics.csv').read_text() == (b / 'metrics.csv').read_text(), arm
        print('PARALLEL == SERIAL (weights, predictions, bytes, history, metrics):', arm, backend, flush=True)
    assert not only or set(only) <= ran, ('--arms not in the parallel list (nothing tested)', sorted(set(only) - ran))


def smoke(arms, proj, work):
    for arm in arms:
        cmd = [sys.executable, str(proj / 'fd.py'), '--arm', arm, '--seed', '43', '--smoke', '--backend', 'eager',
               '--batch', '128',
               '--out', str(work / 'runs')]
        out = work / 'runs' / f'{arm}_s43'
        if out.exists(): shutil.rmtree(out)
        r = subprocess.run(cmd, cwd=ROOT, env={**__import__('os').environ, 'PYTHONPATH': f'{proj}:{ROOT}'},
                           capture_output=True, text=True)
        print(arm, 'rc', r.returncode, r.stdout.strip().splitlines()[-3:], r.stderr[-1500:] if r.returncode else '', flush=True)
        assert r.returncode == 0, arm


if __name__ == '__main__':
    main()

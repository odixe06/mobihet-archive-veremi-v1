"""FullDrift probe: full-data internal holdout, epochs 1-6 of the fixed cosine-T50 full recipe, seeds 42/43.

Contract: docs/methods/full-drift/design.md. `fd_dml`/`fd_gaussian` are the frozen full recipe (full_run/full_method)
on the holdout fixture; technique arms only switch one hook. Decisions use the `group` holdout; official test is
evaluated once after the last epoch, descriptive only.

    python fd.py --arm fd_dml_m4b_gpool --seed 42 --wandb_online
"""
import argparse
import copy
import csv
import hashlib
import json
import lzma
import math
import os
import pickle
import re
import secrets
import struct
import subprocess
import sys
import time
import zlib
from multiprocessing.connection import Client, Listener
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mh import pilot as P
from mh.new_fl import digest, save
from mh.v2_methods import calibrate, finite
from mh.v2_math import SecureSum, pack_vector, unpack
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion
from mh.model_packet import Codec
from mh import v7_nystrom as G
from full_method import Method as FullMethod
from full_run import Run as FullRun

# Owner 17/10 ~10:10Z: one aggregation after every full local epoch only (FL_NIDS_10client protocol)
ARMS = ['fd_dml', 'fd_dml_prox', 'fd_dml_avgm', 'fd_fedavg', 'fd_gaussian', 'fd_gaussian_swappos',
        # round 3 (owner 16/10: target = client private gp; zero extra bytes, architecture-agnostic)
        'fd_dml_m4b', 'fd_dml_swappos', 'fd_dml_ntd', 'fd_dml_q4',
        # round 4 (client): hidden-state KD; Gaussian with receiver/position features zeroed (AntiMem m4b)
        'fd_dml_hkd', 'fd_gaussian_m4b',
        # round 5 (owner 16/10 ~08:20Z): m4b with QProxy q4 (R1); m4b:15 horizon reuses fd_dml_m4b
        'fd_dml_m4b_q4',
        # round 8 (owner 17/10, docs/research/algorithm-improvement-plan-2026-10-17.md): how the private absorbs the teacher,
        # FL-vs-local controls, FedDyn on the proxy, two-view receiver consistency
        'fd_dml_m4b_kdcap', 'fd_dml_m4b_kdcap_wu', 'fd_dml_m4b_kdcap_frz', 'fd_dml_m4b_own', 'fd_gaussian_m4b_own',
        'fd_dml_m4b_dyn', 'fd_dml_swappos_cons', 'fd_dml_swappos_cons0',
        # round 9 (same plan): SCAFFOLD-proxy with a plain-SGD proxy (lr 0.01/0.03/0.1 x cosine; sgd = c=0 control) and
        # Hybrid private-only synthetic: pooled / own Gaussian, or class-balanced local replay, CE for the private only
        'fd_dml_m4b_sgd1', 'fd_dml_m4b_sgd3', 'fd_dml_m4b_sgd10', 'fd_dml_m4b_scaf1', 'fd_dml_m4b_scaf3', 'fd_dml_m4b_scaf10',
        'fd_dml_m4b_gpool', 'fd_dml_m4b_gown', 'fd_dml_m4b_replay',
        # round 11 (owner 17/10 ~05Z: headline m4b+gpool; improve it before 50 rounds): private-side KD x2 (kd2),
        # proxy-side KD weight capped only (kdpx), private-only Gaussian CE weight x2 (gw2)
        'fd_dml_m4b_gpool_kd2', 'fd_dml_m4b_gpool_kdpx', 'fd_dml_m4b_gpool_gw2',
        # round 12 (owner 17/10 ~14Z: limited communication budget): headline with the real int4 QPacket proxy exchange
        'fd_dml_m4b_gpool_q4',
        # round 13 (owner 17/10 ~16Z: try A, B, C + the owner's proposals): Gaussian labels water-filled toward locally rare
        # classes (fill); Gaussian rows the received global proxy rejects are dropped (rej); KL to a snapshot teacher on the
        # Gaussian rows -- global proxy (p1), own private (p1self), class-mean target (p1cm); ternary 2-bit proxy codec (q2)
        'fd_dml_m4b_gpool_fill', 'fd_dml_m4b_gpool_rej', 'fd_dml_m4b_gpool_p1', 'fd_dml_m4b_gpool_p1self',
        'fd_dml_m4b_gpool_p1cm', 'fd_dml_m4b_gpool_q2',
        # round 15 (owner 17/10 ~20Z, probes 14-17: the private decision layer drifts to the local data while its DML teacher, the
        # proxy copy, adapts locally): private KD from the aggregated snapshot frozen for the epoch, uncapped (frz); Gaussian rows
        # learn that snapshot by KL instead of hard CE (gkd)
        'fd_dml_m4b_frz', 'fd_dml_m4b_gpool_frz', 'fd_dml_m4b_gpool_frz_gkd',
        # round 16 (design §Vòng 16): the round-15 headline over the ternary 2-bit proxy exchange (R1)
        'fd_dml_m4b_gpool_q2_frz_gkd',
        # round 17 (owner 18/10 ~06Z: small improvements of the round-15 headline, then 50 rounds): local CE on the logits shifted
        # by log pi_k - log pi_g, so the local term trains the same global-prior posterior the frozen teacher gives (la, FedLC);
        # proxy <- private KL weight capped (kdpx: a better aggregated snapshot is a better private teacher); both
        'fd_dml_m4b_gpool_frz_gkd_la', 'fd_dml_m4b_gpool_frz_gkd_kdpx', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx',
        # round 18 (owner 18/10 ~07Z: try every improvement with evidence): class-conditional Gaussian copula rows instead of the class
        # Gaussian (cop: pooled per-class marginal histograms + normal-score correlation, both SecureSum-additive); proxy CE only
        # (pce, FedAvg proxy = the private's teacher); KD temperature 2 (t2); 512 synthetic rows (g512); Gaussian KL weight x2 (gk2);
        # the headline over the int4 wire (q4, R1)
        'fd_dml_m4b_gpool_frz_gkd_cop', 'fd_dml_m4b_gpool_frz_gkd_pce', 'fd_dml_m4b_gpool_frz_gkd_t2', 'fd_dml_m4b_gpool_frz_gkd_g512',
        'fd_dml_m4b_gpool_frz_gkd_gk2', 'fd_dml_m4b_gpool_q4_frz_gkd',
        # round 19 (--fast): the round-17 best (la_kdpx) over the int4 wire that passed round 18 (R1 combination)
        'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx',
        # round 20 (owner 19/10: exhaust the la/kdpx direction before any 50-epoch run): la strength tau 1.5 / 0.5, shift to the uniform
        # prior (lau), la also on the proxy CE (lap: less locally biased aggregated teacher), private <- teacher KD x2 (kd2)
        'fd_dml_m4b_gpool_frz_gkd_la15_kdpx', 'fd_dml_m4b_gpool_frz_gkd_la05_kdpx', 'fd_dml_m4b_gpool_frz_gkd_lau_kdpx',
        'fd_dml_m4b_gpool_frz_gkd_la_lap_kdpx', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx_kd2',
        # round 21 (owner 19/10: smallest communication budget with no loss of performance; parallel to round 20): la_kdpx over the
        # int4 wire with a lossless LZMA2 body (zl), + client-side uplink error feedback (ef), and the 2-bit wire with ef + zl
        'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_zl', 'fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_ef_zl', 'fd_dml_m4b_gpool_q2_frz_gkd_la_kdpx_ef_zl',
        # round 22 (communication, after q2 collapsed through its downlink): the 3-bit wire both ways (q3); the 2-bit uplink only (u2)
        # with client error feedback, downlink int4 or 3-bit
        'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl', 'fd_dml_m4b_gpool_q4_u2_frz_gkd_la_kdpx_ef_zl', 'fd_dml_m4b_gpool_q3_u2_frz_gkd_la_kdpx_ef_zl',
        # round 23: the 2-bit wire both ways with the RMS (Lloyd-Max for a Gaussian) step instead of the row max (rs)
        'fd_dml_m4b_gpool_q2_rs_frz_gkd_la_kdpx_zl',
        # round 24 (owner 21/10: purposeful synthesis; probe 20): rows built from two own rows (class a, features of a class-b row
        # w.p. XO_RHO), learned by KL to the received snapshot; (a, b) drawn where the private and that snapshot disagree on the
        # client's own rows (xo) or uniformly over the class pairs the client holds (xou, priority control)
        'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xo', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xou',
        # round 25 (owner 21/10 after the round-24 analysis: 77 % of the private's errors are errors of its teacher too): the snapshot's
        # last Linear re-fitted by the server on virtual h256 features drawn from every client's class moments of that snapshot
        # (SecureSum-additive), one covariance per class (cvf) or one shared within-class covariance (cvl); that head is the private's
        # teacher (frz KD on real rows and gkd KL on Gaussian rows)
        'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvf', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvl',
        # round 26 (owner 22/10 after probes 32-36): the cv refit in a 48-d projection of h256 that holds every snapshot head direction
        # plus a fixed random complement, one covariance per class there (cvp, probe 23 Project48), alone and with the round-18 copula
        # support (cop): probe 36 frozen-head, both seeds, cvp + cop vs the deployed method +0.0077 / +0.0051
        'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvp', 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvp_cop',
        # round 27 validation: same selected recipe, either all 66 inputs or physically remove the 13 m4b inputs
        'fd_dml_full66_gpool_q3_frz_gkd_la_kdpx_zl', 'fd_dml_drop53_gpool_q3_frz_gkd_la_kdpx_zl',
        # FedAvg feature-input validation: same full-data holdout recipe, optimizer and aggregation; only input handling changes.
        'fd_fedavg_full66', 'fd_fedavg_drop53', 'fd_fedavg_m4b',
        'fd_fedprox_full66', 'fd_fedprox_drop53', 'fd_fedprox_m4b',
        'fd_fedtgp_full66', 'fd_fedtgp_drop53', 'fd_fedtgp_m4b',
        'fd_canperfl_full66', 'fd_canperfl_drop53', 'fd_canperfl_m4b',
        'fd_centralized_full66', 'fd_centralized_drop53', 'fd_centralized_m4b']
MU, SWAP_P, EMA_BETA, CAL_ROWS, SERVER_BETA = 0.01, 0.5, 0.5 ** 0.5, 65536, 0.9
KD_CAP, DYN_ALPHA, CONS_LAMBDA, PROBE_ROWS = 1.0, 0.01, 1.0, 512   # plan §4.2 / §5.1 / §4.5 screening values
KD_UP, GW2 = 2.0, 2.0   # round 11 kd2: private-side KD weight multiplier; gw2: weight of the private-only Gaussian CE
KD_T, GK2, COP_BINS, COP_LIM = 2.0, 2.0, 256, 8.0   # round 18: t2 temperature, gk2 weight, copula bins over asinh(z) in [-8, 8]
P1_LAMBDA, P1_T, P1_CLASSES = 0.25, 1.0, (0, 4)   # round 13 P1: KL weight, temperature, classes (benign, dataReplay)
XO_RHO, XO_FLOOR = 0.3, 0.1   # round 24 xo: donor feature rate; uniform share (over held pairs) mixed into the pair priority
CV_STEPS, CV_BATCH, CV_LR, CV_JITTER = 3000, 4096, 1e-3, 1e-4   # round 25 cv: server head fit on virtual features (probe 22)
CV_PROJ = 48   # round 26 cvp: projected width (probe 23 Project48)
TGP_DIM, TGP_LAMBDA, TGP_STEPS, TGP_LR, TGP_TAU = 544, 1.0, 100, 1e-3, 100.0
PERSONAL_EPOCHS, PERSONAL_MAX_ROWS = 10, 30_000
BLOCK = ['f_rcv_pos_noise_x', 'f_rcv_pos_noise_y', 'f_rcv_spd_noise', 'f_rcv_acl_noise', 'f_rcv_hed_noise',
         'f_rcv_noise_mag', 'f_rcv_profile_normal', 'f_rcv_profile_cautious', 'f_rcv_profile_aggressive']
POSITION = ['f_rcv_x_rel', 'f_rcv_y_rel', 'f_snd_x_rel', 'f_snd_y_rel']
TRAIN_ROWS, TEST_ROWS = None, 10761343   # train rows are read from the holdout manifest


def fixture_path(args):
    return P.ROOT / ('papers/final/fixture' if getattr(args, 'final', False) and not args.smoke else 'papers/pilot/fixture')


class Gaussian(G.Method):
    """Legacy Gaussian constructor with the active fixture in its packet context."""
    def __init__(self, run, initialize=True):
        self.R, self.arm = run, run.a.arm
        self.role, self.proxy, self.version, self.diag = 'private', [], 0, {}
        self.layout = G.Layout(run.init)
        self.global_vector = self.layout.flatten(run.init).clone()
        self.form = self.arm.split('_')[1]
        self.rank = int(self.form[-1]) if self.form.startswith(('nys', 'lr')) else 0
        self.mode = self.arm.split('_')[-1] if self.form.startswith('nys') else 'fl'
        fixture = digest(fixture_path(run.a) / 'manifest.json')
        self.context = hashlib.sha256(json.dumps(dict(arm=self.arm, seed=run.a.seed,
            fixture=fixture, omega=G.OMEGA_SHA, epsilon=G.EPS, schema=1), sort_keys=True).encode()).digest()[:16]
        self.packets, self.tables = {}, []
        if initialize: self.make_tables()


def is_buffer(name):
    return 'running_' in name or 'num_batches_tracked' in name


RS_STEP = {1: 1.224, 3: 0.586, 7: 0.335}   # round 23 rs: Lloyd-Max uniform step / sigma of a Gaussian source, 3 / 7 / 15 levels


def quantize(t, bits, generator=None, rms=False):
    """antimem.quantize: symmetric per-row (dim0) scale, fp16 scales; stochastic rounding when a generator is given.
    rms (round 23): step = RS_STEP * row RMS instead of row max / levels; values beyond the outer level are clipped (error feedback
    keeps the clipped part)."""
    levels = 2 ** (bits - 1) - 1
    rows = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
    scale = ((RS_STEP[levels] * rows.pow(2).mean(1, keepdim=True).sqrt()) if rms else rows.abs().amax(1, keepdim=True) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    v = rows / safe
    if generator is None:
        q = torch.round(v)
    else:
        q = torch.floor(v + torch.rand(v.shape, device=v.device, generator=generator))
    q = q.clamp(-levels, levels) * (scale > 0)
    nbytes = (q.numel() * bits + 7) // 8 + 2 * scale.numel()
    return (q * scale).reshape(t.shape), q.to(torch.int8), scale.half(), nbytes


QHEADER = struct.Struct('<4sBBHIII16sI')   # model_packet.HEADER layout, magic MHQ4 (MHQ2 for 2-bit; MHZ4/MHZ2 lossless-coded body)
LZMA = [dict(id=lzma.FILTER_LZMA2, preset=9 | lzma.PRESET_EXTREME)]   # round 21 zl: raw LZMA2 stream of the packet body


class QPacket:
    """QProxy wire format (int4, 3-bit or 2-bit ternary): per state_dict entry in order -- integer counters int64; BN buffers fp16;
    parameters: per-row fp16 scales, then the codes q+2^(bits-1) (int4: 1..15, 3-bit: 1..7, 2-bit: 1..3) packed 8//bits per byte, lowest
    bits first. decode() returns the tensors quantize() produced (q * scale; a -0.0 code reads back as +0.0), so training is
    unchanged and len(packet) is the real byte count. Header: magic, schema, bits, base/version, body length, context, CRC32."""
    def __init__(self, model, context, bits=4, lossless=False):
        assert bits in (2, 3, 4)
        self.bits, self.magic, self.lossless = bits, (b'MHZ%d' if lossless else b'MHQ%d') % bits, lossless
        self.raw_len = None   # zl: body bytes before the lossless stage (last encode/decode)
        self.context = hashlib.sha256(context.encode()).digest()[:16]
        self.fields = [(name, tuple(v.shape), v.dtype) for name, v in model.state_dict().items()]

    def encode(self, entries, base, version):
        """entries[name]: (q int8, scale fp16) for a parameter, a float tensor (sent fp16) for a BN buffer, an int tensor."""
        chunks, per = [], 8 // self.bits
        for name, shape, dtype in self.fields:
            entry = entries[name]
            if isinstance(entry, tuple):
                q, scale = entry
                codes = (q.reshape(-1).cpu().numpy().astype(np.int16) + 2 ** (self.bits - 1)).astype(np.uint8)
                assert codes.min() >= 1 and codes.max() <= 2 ** self.bits - 1, 'code range'
                if len(codes) % per: codes = np.append(codes, np.zeros(per - len(codes) % per, np.uint8))
                packed = codes[0::per]
                for i in range(1, per): packed = packed | (codes[i::per] << (self.bits * i))
                chunks += [scale.reshape(-1).cpu().numpy().astype('<f2').tobytes(), packed.tobytes()]
            elif dtype.is_floating_point:
                chunks.append(entry.detach().cpu().numpy().astype('<f2').tobytes())
            else:
                chunks.append(entry.detach().cpu().numpy().astype('<i8').tobytes())
        body = b''.join(chunks); self.raw_len = len(body)
        if self.lossless: body = lzma.compress(body, format=lzma.FORMAT_RAW, filters=LZMA)   # round 21: same codes, fewer bytes
        return QHEADER.pack(self.magic, 1, self.bits, 0, base, version, len(body), self.context, zlib.crc32(body)) + body

    def decode(self, packet, device, expected_base):
        magic, schema, bits, reserved, base, version, length, context, crc = QHEADER.unpack_from(packet)
        body = packet[QHEADER.size:]; per = 8 // self.bits
        assert (magic, schema, bits, reserved) == (self.magic, 1, self.bits, 0) and context == self.context, 'QProxy header'
        assert length == len(body) and zlib.crc32(body) == crc and base == expected_base and version == base + 1, 'QProxy packet'
        if self.lossless: body = lzma.decompress(body, format=lzma.FORMAT_RAW, filters=LZMA)
        out, offset = {}, 0; self.raw_len = len(body)
        for name, shape, dtype in self.fields:
            n = int(np.prod(shape))
            if not dtype.is_floating_point:
                out[name] = torch.from_numpy(np.frombuffer(body, '<i8', n, offset).copy()).reshape(shape).to(device); offset += 8 * n
            elif is_buffer(name):
                out[name] = torch.from_numpy(np.frombuffer(body, '<f2', n, offset).astype(np.float32)).reshape(shape).to(device)
                offset += 2 * n
            else:
                rows = shape[0] if len(shape) > 1 else 1
                scale = torch.from_numpy(np.frombuffer(body, '<f2', rows, offset).astype(np.float32)).reshape(rows, 1).to(device)
                offset += 2 * rows; size = (n + per - 1) // per; packed = np.frombuffer(body, np.uint8, size, offset); offset += size
                codes = np.empty(per * size, np.uint8)
                for i in range(per): codes[i::per] = (packed >> (self.bits * i)) & (2 ** self.bits - 1)
                q = torch.from_numpy(codes[:n].astype(np.float32) - 2 ** (self.bits - 1)).reshape(rows, -1).to(device)
                out[name] = (q * scale).reshape(shape)
        assert offset == len(body)
        return out


def not_true(z, y):
    """FedNTD: log-softmax over the not-true classes (true-class logit pushed to -1e4, so it carries no mass)."""
    return F.log_softmax(z.masked_fill(F.one_hot(y, z.shape[1]).bool(), -1e4), 1)


def js(za, zb):
    """Jensen-Shannon divergence between the softmax of two logit batches, mean over rows (both sides get gradients)."""
    la, lb = F.log_softmax(za, 1), F.log_softmax(zb, 1)
    lm = torch.logsumexp(torch.stack([la, lb]), 0) - np.log(2.)
    return 0.5 * ((la.exp() * (la - lm)).sum(1) + (lb.exp() * (lb - lm)).sum(1)).mean()


def sgd_lr(tokens):
    """'sgd3'/'scaf3' -> 0.03: base LR of the plain-SGD proxy (same cosine shape as the AdamW schedule)."""
    hit = [re.fullmatch(r'(sgd|scaf)(\d+)', t) for t in tokens]
    return next((int(h.group(2)) / 100 for h in hit if h), None)


class Mixed:
    """AdamW private (+ local projector) and plain-SGD proxy behind one GradScaler unscale/inf-skip decision."""
    def __init__(self, *optimizers): self.optimizers = optimizers; self.param_groups = [g for o in optimizers for g in o.param_groups]
    def step(self): [o.step() for o in self.optimizers]
    def zero_grad(self, set_to_none=True): [o.zero_grad(set_to_none=set_to_none) for o in self.optimizers]


def check_full_manifest(manifest, n, test_rows):
    """Production coverage gate: full holdout fixture, every non-holdout train row loaded; counts are per-class lists."""
    assert manifest['protocol'].startswith('temporal2% whole partition') and manifest['train_cap'] is None
    assert manifest['group_overlap'] == 0 and manifest['eval_cap'] == 65536
    assert n == [sum(manifest['counts'][f'train_{k}']) for k in range(len(n))]
    assert manifest['source_rows'] == 43045415 and test_rows == TEST_ROWS, 'full holdout coverage gate'
    assert manifest['test_scaler'].startswith('scaler.json')


def check_final_manifest(manifest, n, test_rows):
    """Final protocol gate (owner 19/10, protocol §3): every train80% row trains, no internal holdout; official test20% every round."""
    assert manifest['protocol'].startswith('final: every train80%') and manifest['train_cap'] is None
    assert n == [sum(manifest['counts'][f'train_{k}']) for k in range(len(n))] and sum(n) == manifest['source_rows'] == 43045415
    assert test_rows == TEST_ROWS and manifest['test_scaler'].startswith('scaler.json'), 'final coverage gate'


def fill_labels(pi, n):
    """Round 13 fill: Gaussian label shares q = max(0, L - pi) with sum 1 (water-filling: real + Gaussian label mass as even
    as the n rows allow), rounded to n labels by largest remainder (ties: lower class), sorted by class."""
    pi = pi.double().cpu(); s, _ = torch.sort(pi)
    for j in range(1, len(pi) + 1):
        level = (1 + s[:j].sum()) / j
        if j == len(pi) or level <= s[j]: break
    q = (level - pi).clamp_min(0); q = q / q.sum()
    count = torch.floor(n * q).long(); rest = n * q - count
    count[torch.argsort(-rest, stable=True)[:n - int(count.sum())]] += 1
    return torch.repeat_interleave(torch.arange(len(pi)), count)


def assign(sizes, workers):
    """--workers: clients per process, longest-first onto the least-loaded process (ties: lowest rank); one process -> all."""
    load, own = [0] * workers, [[] for _ in range(workers)]
    for k in sorted(range(len(sizes)), key=lambda k: (-sizes[k], k)):
        rank = min(range(workers), key=lambda w: (load[w], w)); load[rank] += sizes[k]; own[rank].append(k)
    return [sorted(o) for o in own]


class Clients(list):
    """Per-client train rows: this process's clients stay on the GPU; any other client is read from the fixture on access
    (setup only: Gaussian moments, compile probe, EMA calibration on rank 0)."""
    def __init__(self, items, load): super().__init__(items); self.load = load
    def __getitem__(self, k):
        value = super().__getitem__(k)
        return self.load(k) if value is None else value
    def __iter__(self): return (self[k] for k in range(len(self)))


class TrainableGlobalPrototypes(torch.nn.Module):
    """FedTGP server module ported from the reference implementation."""
    def __init__(self, dev):
        super().__init__(); self.raw = torch.nn.Parameter(torch.randn(P.C, TGP_DIM, device=dev) * .02)
        self.processor = torch.nn.Sequential(torch.nn.Linear(TGP_DIM, 256), torch.nn.ReLU(),
                                             torch.nn.Linear(256, TGP_DIM)).to(dev)
    def forward(self): return self.processor(self.raw)

    def update(self, payloads):
        valid = [(torch.as_tensor(p['sum'], device=self.raw.device), torch.as_tensor(p['count'], device=self.raw.device)) for p in payloads]
        local = [(s / n.clamp_min(1)[:, None], n > 0) for s, n in valid]
        local = [(p, mask) for p, mask in local if bool(mask.any())]
        if not local: return self().detach(), 0., 0.
        with torch.no_grad():
            centers = [torch.stack([p[c] for p, mask in local if mask[c]]).mean(0) for c in range(P.C)
                       if any(bool(mask[c]) for _, mask in local)]
            margin = min(float(torch.pdist(torch.stack(centers)).max()) if len(centers) > 1 else 0., TGP_TAU)
        opt = torch.optim.Adam(self.parameters(), lr=TGP_LR)
        for _ in range(TGP_STEPS):
            gp = self(); losses = []
            for proto, mask in local:
                labels = torch.where(mask)[0]; logits = -torch.cdist(proto[labels], gp)
                logits[torch.arange(len(labels), device=labels.device), labels] -= margin
                losses.append(F.cross_entropy(logits, labels))
            loss = torch.stack(losses).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return self().detach(), float(loss.detach()), margin


class Method(FullMethod):
    def __init__(self, run, initialize=True):
        super().__init__(run, initialize)
        self.ema_private = None; self.cv_head = None   # round 25: calibrated head.5 of the current snapshot (None: not calibrated)
        self.fedavg = any(name in self.arm for name in ('fedavg', 'fedprox', 'fedtgp', 'canperfl', 'centralized'))
        self.tgp, self.canperfl, self.centralized = 'fedtgp' in self.arm, 'canperfl' in self.arm, 'centralized' in self.arm
        self.momentum = self.arm.endswith('_avgm'); self.velocity = {}
        self.bits = next((b for b in (4, 3, 2) if f'q{b}' in self.arm.split('_')), None); self.exact = None
        self.ubits = 2 if 'u2' in self.arm.split('_') else self.bits   # round 22: uplink bits (u2: 2-bit up, downlink keeps self.bits)
        self.rs = 'rs' in self.arm.split('_')   # round 23: RMS (Lloyd-Max) step instead of the row max, both directions
        # round 21 (communication): lossless LZMA2 of the packet body (zl); client-side uplink error feedback (ef): each client adds
        # its previous quantization residual to the next delta (state stays on the client, in the process that trains it)
        self.zl, self.ef = 'zl' in self.arm.split('_'), 'ef' in self.arm.split('_'); self.ef_state, self.qraw_up = {}, 0
        assert not (self.zl or self.ef or self.rs) or self.bits
        self.dyn = 'dyn' in self.arm.split('_'); self.dyn_a, self.dyn_h = {}, {}   # FedDyn: client a_i (never sent), server h
        tokens = self.arm.split('_')
        self.scaf = any(t.startswith('scaf') for t in tokens); self.scaf_c, self.scaf_ci, self.scaf_dsum = None, {}, None
        # Hybrid private-only: Gaussian class moments (pooled 'gpool' / client-own 'gown') drawn for the private model only
        self.paux = 'gpool' in tokens or 'gown' in tokens; self.gmode = 'own' if 'gown' in tokens else 'fl'
        self.draw_bad = torch.zeros((), dtype=torch.bool, device=run.dev)   # deferred v7 draw check, asserted per epoch
        self.cop = 'cop' in tokens; assert not self.cop or (self.paux and self.gmode == 'fl')
        if self.cop and initialize: self.make_copula()
        elif self.paux:
            self.gaussian = Gaussian(SimpleNamespace(init=run.init, dev=run.dev, a=SimpleNamespace(arm='n_full', seed=run.a.seed,
                                      final=getattr(run.a, 'final', False), smoke=run.a.smoke)),
                                     initialize=False)
            if initialize:
                self.gaussian.R = run; self.gaussian.mode = self.gmode
                self.gaussian.make_tables(); self.gaussian_state = self.gaussian.state()
        if self.fedavg:   # one shared model trained with CE only: the full DML proxy machinery without private/KD
            self.kd = True
            torch.manual_seed(run.a.seed + 1000)
            initial = P.build_model(run.architecture, latent=True).to(run.dev)
            self.proxy = [copy.deepcopy(initial) for _ in range(run.K)]
            self.global_state = {name: value.clone() for name, value in initial.state_dict().items()}
            self.codec = Codec(initial, f'full:{self.arm}:{run.a.seed}')
            if self.tgp:
                torch.manual_seed(run.a.seed + 2000); self.tgp_server = TrainableGlobalPrototypes(run.dev)
                self.tgp_proto = self.tgp_server().detach()

    def prior(self):
        super().prior()
        if self.R.la:   # round 18: the SecureSum aggregate prior (r17: in-process); round 20: tau and the uniform target (lau)
            self.R.la_shift = self.R.log_pi - (-math.log(16) if self.R.la == 'lau' else self.R.log_pg)
            tau = {'la05': .5, 'la15': 1.5}.get(self.R.la)
            if tau: self.R.la_shift = tau * self.R.la_shift
        if self.bits:
            self.exact = {k: v.detach().clone() for k, v in self.global_state.items()}
            self.qpacket = QPacket(self.proxy[0], f'q{self.bits}:{self.arm}:{self.R.a.seed}', self.bits, lossless=self.zl)
            self.qpacket_up = self.qpacket if self.ubits == self.bits else \
                QPacket(self.proxy[0], f'q{self.ubits}up:{self.arm}:{self.R.a.seed}', self.ubits, lossless=self.zl)

    def start_round(self):
        super().start_round()
        for key in ('codec_up_encode_seconds', 'codec_up_decode_seconds', 'codec_down_encode_seconds', 'codec_down_decode_seconds'):
            self.diag[key] = 0.
        if self.bits:
            self.qsum, self.qseen = {}, set()

    def proxy_epoch(self, k, r):
        """QProxy (antimem): up = per-row int-b delta vs the replica, stochastic; BN buffers fp16; integers raw."""
        if not self.bits:
            started = time.perf_counter(); result = super().proxy_epoch(k, r)
            self.diag['codec_up_encode_seconds'] += time.perf_counter() - started
            return result
        self.accumulate_quantized(k, r, self.quantized_wire(k, r))
        return None, self.logs[k]

    def quantized_wire(self, k, r):
        """Client side of QProxy: the packet client k puts on the wire."""
        g = self.R.gen(r, k, 77); entries = {}
        for name, value in self.proxy[k].state_dict().items():
            if not value.is_floating_point(): entries[name] = value.clone()
            elif is_buffer(name): entries[name] = value.detach()
            elif self.ef:   # round 21: v = delta + e_k; send Q(v); e_k <- v - Q(v)
                v = value.detach().float() - self.global_state[name].float() + self.ef_state.get((k, name), 0.)
                sent, q, scale, _ = quantize(v, self.ubits, g, self.rs); self.ef_state[(k, name)] = v - sent
                entries[name] = (q, scale)
            else:
                _, q, scale, _ = quantize(value.detach().float() - self.global_state[name].float(), self.ubits, g, self.rs)
                entries[name] = (q, scale)
        started = time.perf_counter(); packet = self.qpacket_up.encode(entries, r - 1, r)
        self.diag['codec_up_encode_seconds'] += time.perf_counter() - started
        return packet

    def accumulate_quantized(self, k, r, packet):
        """Server side of QProxy: decode client k's packet, running sum over clients in client order."""
        started = time.perf_counter(); wires = self.qpacket_up.decode(packet, self.R.dev, r - 1)
        self.diag['codec_up_decode_seconds'] += time.perf_counter() - started
        for name, wire in wires.items():
            if not wire.is_floating_point():
                self.qsum[name] = wire if name not in self.qsum else torch.maximum(self.qsum[name], wire)
            else:
                self.qsum[name] = wire.double() if name not in self.qsum else self.qsum[name] + wire.double()
        assert k not in self.qseen; self.qseen.add(k)
        if self.zl: self.qraw_up += self.qpacket_up.raw_len   # diagnostics only: body size before the lossless stage
        self.R.bytes.append((r, k, 'up', 'proxy_quantized', len(packet)))

    def aggregate_quantized(self, r):
        """Server keeps the exact model (error feedback); down = int-b nearest of exact − replica."""
        assert self.qseen == set(range(self.R.K))
        entries = {}
        for name, total in self.qsum.items():
            if not total.is_floating_point():
                self.exact[name] = total.clone(); entries[name] = total
            elif is_buffer(name):
                self.exact[name] = (total / self.R.K).float(); entries[name] = self.exact[name]
            else:
                self.exact[name] = self.exact[name] + (total / self.R.K).float()
                _, q, scale, _ = quantize(self.exact[name] - self.global_state[name], self.bits, rms=self.rs); entries[name] = (q, scale)
        started = time.perf_counter(); packet = self.qpacket.encode(entries, r - 1, r)
        self.diag['codec_down_encode_seconds'] += time.perf_counter() - started
        started = time.perf_counter(); decoded = self.qpacket.decode(packet, self.R.dev, r - 1)
        self.diag['codec_down_decode_seconds'] += time.perf_counter() - started
        num = den = 0.
        for name, total in self.qsum.items():   # what every client rebuilds from the downlink bytes
            if not total.is_floating_point() or is_buffer(name): new = decoded[name]
            else:
                new = self.global_state[name] + decoded[name]
                num += float((self.exact[name] - new).double().pow(2).sum()); den += float(self.exact[name].double().pow(2).sum())
            assert torch.isfinite(new.float()).all(), ('quantized proxy nonfinite', name)
            self.global_state[name] = new.to(self.global_state[name].dtype)
        for k in range(self.R.K):
            self.R.bytes.append((r, k, 'down', 'proxy_quantized', len(packet)))
            self.proxy[k].load_state_dict(self.global_state, strict=True)
        self.global_vector = self.layout.flatten(self.proxy[0]).clone(); self.latest_packet = packet
        self.diag['q_residual_rel'] = (num / max(den, 1e-30)) ** 0.5; self.diag['q_bytes_down'] = len(packet)
        if self.zl:   # round 21: wire bytes / bytes before the lossless stage (up: mean over clients)
            self.diag['q_zl_ratio_down'] = len(packet) / (QHEADER.size + self.qpacket.raw_len)
            self.diag['q_raw_up_mean'] = self.qraw_up / self.R.K; self.qraw_up = 0
        self.version = r

    def make_copula(self, chunk=262144):
        """Round 18 cop, one-shot bootstrap in two SecureSum phases (every message additive, no record leaves a client):
        A) each client: per class x visible feature, counts in COP_BINS fixed public bins of asinh(z) -> pooled marginal CDFs;
        B) each client maps its rows through the pooled CDFs to normal scores s = Phi^-1(F(x)) and sends per class sum s, sum s s^T
           -> pooled normal-score correlation (Gaussian copula). Masked (m4b) features are never counted, drawn as 0."""
        R = self.R; C = P.C; keep = torch.nonzero(~R.zero_cols).squeeze(1); d = len(keep); B = COP_BINS
        edges = torch.linspace(-COP_LIM, COP_LIM, B + 1, device=R.dev); width = float(edges[1] - edges[0])
        feature = torch.arange(d, device=R.dev)
        def bins(x): return torch.bucketize(torch.asinh(x).clamp(-COP_LIM, COP_LIM - 1e-6), edges[1:-1])
        def rows(k):
            X, Y = R.X[k], R.Y[k]
            for i in range(0, len(Y), chunk): yield X[i:i + chunk][:, keep].float(), Y[i:i + chunk]
        hist = SecureSum('counts', 0, C * d * B, R.dev)
        for k in range(R.K):   # phase A
            h = torch.zeros(C * d * B, dtype=torch.long, device=R.dev)
            for x, y in rows(k): h += torch.bincount(((y[:, None] * d + feature) * B + bins(x)).flatten(), minlength=C * d * B)
            packet = pack_vector('counts', h.double(), count=R.n[k]); self.record(packet, 0, k, 'up', 'copula_hist'); hist.add(k, packet)
        total, _ = hist.finish(); packet = pack_vector('counts', total)
        for k in range(R.K): self.record(packet, 0, k, 'down', 'copula_hist')
        counts = unpack(packet, R.dev)['values'].double().view(C, d, B); n = counts[:, 0].sum(1)
        assert (n > 0).all() and torch.equal(counts.sum(2), n[:, None].expand(C, d))
        cum = torch.cat([torch.zeros(C, d, 1, device=R.dev, dtype=torch.float64), counts.cumsum(2)], 2) / n[:, None, None]
        flat = cum.float().flatten()
        tri = torch.triu_indices(d, d, device=R.dev); size = C * (d + tri.shape[1])
        scores = SecureSum('moments', 0, size, R.dev)
        for k in range(R.K):   # phase B
            t1 = torch.zeros(C, d, dtype=torch.float64, device=R.dev); t2 = torch.zeros(C, d, d, dtype=torch.float64, device=R.dev)
            for x, y in rows(k):
                b = bins(x); at = ((y[:, None] * d + feature) * (B + 1) + b)
                lo, hi = flat[at], flat[at + 1]
                frac = (torch.asinh(x).clamp(-COP_LIM, COP_LIM - 1e-6) - edges[b]) / width
                s = torch.special.ndtri((lo + frac * (hi - lo)).clamp(1e-6, 1 - 1e-6)).double()
                for c in torch.unique(y).tolist():
                    sc = s[y == c]; t1[c] += sc.sum(0); t2[c] += sc.T @ sc
            packet = pack_vector('moments', torch.cat([t1, t2[:, tri[0], tri[1]]], 1).flatten(), count=R.n[k])
            self.record(packet, 0, k, 'up', 'copula_scores'); scores.add(k, packet)
        total, _ = scores.finish(); total = total.double().view(C, -1)
        mean = total[:, :d] / n[:, None]; second = torch.zeros(C, d, d, dtype=torch.float64, device=R.dev)
        second[:, tri[0], tri[1]] = total[:, d:]; second = second + second.triu(1).transpose(1, 2)
        cov = (second / n[:, None, None] - mean[:, :, None] * mean[:, None, :]).cpu()   # CPU float64: deterministic eigh
        var = cov.diagonal(dim1=1, dim2=2); live = var > 1e-8   # a feature constant within a class (atom 100 %) has no dependence
        sd = var.clamp_min(1e-8).sqrt(); corr = cov / (sd[:, :, None] * sd[:, None, :])
        corr = torch.where(live[:, :, None] & live[:, None, :], corr, torch.zeros_like(corr))
        corr.diagonal(dim1=1, dim2=2).fill_(1.)
        value, vector = torch.linalg.eigh(corr)   # nearest-PD by eigenvalue floor, then back to unit diagonal
        corr = vector @ torch.diag_embed(value.clamp_min(1e-4)) @ vector.transpose(1, 2)
        unit = corr.diagonal(dim1=1, dim2=2).sqrt(); corr = corr / (unit[:, :, None] * unit[:, None, :])
        factor = torch.linalg.cholesky(corr).float().to(R.dev)
        packet = pack_vector('moments', factor[:, tri[1], tri[0]].flatten())   # lower triangle of each class Cholesky factor
        for k in range(R.K): self.record(packet, 0, k, 'down', 'copula_corr')
        low = torch.zeros(C, d, d, device=R.dev); low[:, tri[1], tri[0]] = unpack(packet, R.dev)['values'].view(C, -1)
        self.copula = dict(cum=cum.float(), factor=low, keep=keep, edges=edges, width=width)

    def draw_copula(self, labels, g):
        t = self.copula; d = len(t['keep'])
        z = torch.randn(len(labels), d, device=self.R.dev, generator=g)
        u = torch.special.ndtr((t['factor'][labels] @ z[:, :, None]).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
        seq = t['cum'][labels]
        b = (torch.searchsorted(seq, u[:, :, None]).squeeze(-1) - 1).clamp(0, COP_BINS - 1)
        lo, hi = seq.gather(2, b[:, :, None]).squeeze(-1), seq.gather(2, (b + 1)[:, :, None]).squeeze(-1)
        frac = ((u - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
        x = torch.zeros(len(labels), P.D, device=self.R.dev)
        x[:, t['keep']] = torch.sinh(t['edges'][b] + frac * t['width'])
        self.draw_bad |= ~(torch.isfinite(x).all() & torch.isfinite(x.half()).all())
        return x

    def draw(self, labels, g, k):
        """v7_nystrom.Method.draw ('full' form: same ops, same generator stream) for aux (Gaussian) and paux (gpool/gown);
        its finite/fp16-overflow assert is deferred to the end of the client epoch, so drawing needs no host sync."""
        if self.cop: return self.draw_copula(labels, g)
        assert self.gaussian.form == 'full'
        tables = self.gaussian_state['tables']; t = tables[k] if (self.gmode == 'own' if self.paux else self.own) else tables[0]
        z = torch.randn(len(labels), P.D, device=self.R.dev, generator=g)
        x = t['mu'][labels]
        x = x + (t['factor'][labels] @ z[:, :, None]).squeeze(-1)
        self.draw_bad |= ~(torch.isfinite(x).all() & torch.isfinite(x.half()).all())
        return x

    def aggregate(self, r):
        if self.scaf:   # SCAFFOLD server control: c <- c + (1/N) sum_i delta c_i (full participation, cohort sum only)
            self.scaf_c = [c + d / self.R.K for c, d in zip(self.scaf_c, self.scaf_dsum)]; self.scaf_dsum = None
            nbytes = 4 * sum(c.numel() for c in self.scaf_c)
            for k in range(self.R.K): self.R.bytes.append((r, k, 'down', 'control', nbytes))
            self.diag['scaf_c_norm'] = sum(float(c.square().sum()) for c in self.scaf_c) ** 0.5
            self.diag['scaf_ci_norm'] = float(np.mean([sum(float(v.square().sum()) for v in c) ** 0.5 for c in self.scaf_ci.values()]))
        if self.bits:
            return self.aggregate_quantized(r)
        if self.dyn:
            return self.aggregate_dyn(r)
        if not self.momentum:
            return super().aggregate(r)
        # FedAvgM: server velocity on the averaged parameter delta; BN buffers/counters are averaged as usual.
        state = self.aggregate_sum.finish(self.R.K); norm = 0.
        for name, value in state.items():
            if not value.is_floating_point() or 'running_' in name: continue
            delta = value.float() - self.global_state[name].float()
            self.velocity[name] = SERVER_BETA * self.velocity.get(name, torch.zeros_like(delta)) + delta
            state[name] = (self.global_state[name].float() + self.velocity[name]).to(value.dtype)
            norm += float(self.velocity[name].square().sum())
        self.diag['server_velocity_norm'] = norm ** 0.5
        packet = self.codec.encode(state, r - 1, r)
        self.global_state, _, _ = self.codec.decode(packet, self.R.dev, r - 1)
        for k in range(self.R.K):
            self.record(packet, r, k, 'down', 'proxy_state')
            self.proxy[k].load_state_dict(self.global_state, strict=True)
        self.global_vector = self.layout.flatten(self.proxy[0]).clone(); self.latest_packet = packet
        self.version = r

    def aggregate_dyn(self, r):
        """FedDyn server (Acar et al. 2021, Alg. 1, full participation): h <- h - alpha * mean_i(w_i - w), w <- mean_i(w_i) - h/alpha.
        Trainable proxy parameters only, from the cohort sum (no per-client view); BN buffers/counters averaged as usual."""
        state = self.aggregate_sum.finish(self.R.K); h_norm = 0.
        for name, _ in self.proxy[0].named_parameters():
            mean = state[name].float(); delta = mean - self.global_state[name].float()
            self.dyn_h[name] = self.dyn_h.get(name, torch.zeros_like(delta)) - DYN_ALPHA * delta
            state[name] = (mean - self.dyn_h[name] / DYN_ALPHA).to(state[name].dtype); h_norm += float(self.dyn_h[name].square().sum())
        self.diag['dyn_h_norm'] = h_norm ** 0.5
        self.diag['dyn_a_norm'] = float(np.mean([sum(float(v.square().sum()) for v in a) ** 0.5 for a in self.dyn_a.values()]))
        packet = self.codec.encode(state, r - 1, r)
        self.global_state, _, _ = self.codec.decode(packet, self.R.dev, r - 1)
        for k in range(self.R.K):
            self.record(packet, r, k, 'down', 'proxy_state')
            self.proxy[k].load_state_dict(self.global_state, strict=True)
        self.global_vector = self.layout.flatten(self.proxy[0]).clone(); self.latest_packet = packet
        self.version = r

    def update_ema(self):
        if self.fedavg: return
        def blend(old, new):
            if old is None:
                return {k: v.detach().clone() for k, v in new.items()}
            return {k: (EMA_BETA * old[k] + (1 - EMA_BETA) * v.detach()) if v.is_floating_point() else v.detach().clone()
                    for k, v in new.items()}
        olds = self.ema_private or [None] * self.R.K
        self.ema_private = [blend(o, m.state_dict()) for o, m in zip(olds, self.R.priv)]


class Run(FullRun):
    def __init__(self, args):
        self.a, self.dev, self.K, self.bytes = args, torch.device('cuda'), getattr(args, 'clients', 20), []
        tokens = args.arm.split('_')
        self.aux_w = GW2 if 'gw2' in tokens else args.aux_weight
        self.kdup, self.kdpx = 'kd2' in tokens, 'kdpx' in tokens   # round 11 KD weights (private x2 / proxy side capped)
        self.mu = MU if {'prox', 'fedprox'} & set(tokens) else 0.
        self.swap = 'swappos' in tokens
        self.fedavg = any(name in args.arm for name in ('fedavg', 'fedprox', 'fedtgp', 'canperfl', 'centralized'))
        self.tgp, self.canperfl, self.centralized = 'fedtgp' in args.arm, 'canperfl' in args.arm, 'centralized' in args.arm
        self.ntd = 'ntd' in tokens
        # round 8: capped KD weight; private KD off until the first aggregated snapshot (wu) or from a frozen snapshot (frz);
        # FedDyn proxy; two receiver views + JS (cons, lambda 1) or the same two views without JS (cons0)
        self.kdcap = 'kdcap' in tokens; self.frz = 'frz' in tokens; self.warm = self.frz or 'wu' in tokens
        self.dyn = 'dyn' in tokens; self.cons = 'cons' in tokens or 'cons0' in tokens
        self.lam = CONS_LAMBDA if 'cons' in tokens else 0.
        # round 9: plain-SGD proxy (sgd*/scaf*), SCAFFOLD correction (scaf*), private-only extra rows (gpool/gown/replay)
        self.sgd_lr = sgd_lr(tokens); self.scaf = any(t.startswith('scaf') for t in tokens); self.replay = 'replay' in tokens
        self.paux = 'gpool' in tokens or 'gown' in tokens or self.replay
        # round 13: water-filled Gaussian labels (fill); a snapshot teacher scores the Gaussian rows (rej / p1 / p1self / p1cm)
        self.fill, self.rej = 'fill' in tokens, 'rej' in tokens
        self.p1 = next((t for t in tokens if t in ('p1', 'p1self', 'p1cm')), None)
        self.gkd = 'gkd' in tokens; assert not self.gkd or self.paux   # round 15: KL to the received snapshot on the Gaussian rows
        self.la = next((t for t in tokens if t in ('la', 'la05', 'la15', 'lau')), None)   # round 17: logit-adjusted local CE
        assert not self.la or self.frz   # (teacher = global snapshot); round 20: tau 0.5/1.5 (la05/la15), uniform target prior (lau)
        self.lap = 'lap' in tokens; assert not self.lap or self.la   # round 20: the same shift on the proxy CE
        # round 18: proxy CE only (pce), private KD at temperature KD_T (t2), Gaussian KL weight x GK2 (gk2); g512 is set in main()
        self.pce, self.t2, self.gk2 = 'pce' in tokens, 't2' in tokens, 'gk2' in tokens
        assert not (self.t2 or self.gk2) or self.gkd
        self.xo = next((t for t in tokens if t in ('xo', 'xou')), None); assert not self.xo or (self.gkd and not self.t2)   # round 24
        self.cv = next((t for t in tokens if t in ('cvf', 'cvl', 'cvp')), None); assert not self.cv or (self.gkd and self.frz)   # round 25/26
        assert ('g512' in tokens) == (args.aux_batch == (128 if args.smoke else 512)), 'g512 <=> doubled aux_batch (set in main)'
        self.drop53 = 'drop53' in tokens
        hidden = BLOCK + POSITION if 'm4b' in tokens else POSITION if self.swap else []
        # --final (owner 19/10): the final-protocol fixture (every train80% row, no group/old); smoke reuses the pilot fixture
        self.final = getattr(args, 'final', False)
        fixture = fixture_path(args)
        manifest = json.loads((fixture / 'manifest.json').read_text())
        names = manifest['features']
        self.block = torch.tensor([names.index(n) for n in BLOCK], device=self.dev)
        self.zero = torch.tensor([names.index(n) for n in hidden], device=self.dev, dtype=torch.long)
        self.zero_cols = torch.zeros(len(names), dtype=torch.bool, device=self.dev); self.zero_cols[self.zero] = True
        dropped = set(BLOCK + POSITION)
        self.keep = torch.tensor([i for i, name in enumerate(names) if name not in dropped], device=self.dev, dtype=torch.long)
        assert len(self.keep) == 53 and len(dropped) == 13
        def rows(k, name, dtype=None):
            v = np.load(fixture / f'train_{k}_{name}.npy', mmap_mode='r')
            if args.smoke: v = v[np.random.default_rng(k).choice(len(v), 1024, replace=False)]
            return torch.from_numpy(np.array(v, dtype=dtype, copy=True)).to(self.dev)
        sizes = [len(np.load(fixture / f'train_{k}_y.npy', mmap_mode='r')) for k in range(self.K)]
        self.own = assign(sizes, getattr(args, 'workers', 1))[getattr(args, 'rank', 0)]   # clients this process trains
        self.fused = getattr(args, 'fast', False) or None   # --fast: fused AdamW; None keeps the default (foreach) bit-exact
        self.X = Clients([rows(k, 'X') if k in self.own else None for k in range(self.K)], lambda k: rows(k, 'X'))
        self.Y = [rows(k, 'y', 'int64') for k in range(self.K)]
        self.G = [rows(k, 'g', 'int64') if k in self.own else None for k in range(self.K)] if self.swap else []
        self.H = {}
        holdouts = ('test',) if self.final else ('group', 'old') if getattr(args, 'holdout_only', False) else ('group', 'old', 'test')
        for hold in holdouts:
            name = 'old' if args.smoke and hold == 'test' else hold
            x = np.load(fixture / f'{name}_X.npy', mmap_mode='r'); y = np.load(fixture / f'{name}_y.npy', mmap_mode='r')
            if args.smoke: x, y = x[:256], y[:256]
            self.H[hold] = (torch.from_numpy(np.array(x, copy=True)).to(self.dev),
                            torch.from_numpy(np.array(y, dtype='int64', copy=True)).to(self.dev))
        self.n = [len(y) for y in self.Y]
        test_rows = len(self.H['test'][1]) if 'test' in self.H else int(manifest['test_rows'])
        if not args.smoke: (check_final_manifest if self.final else check_full_manifest)(manifest, self.n, test_rows)
        self.counts = torch.stack([torch.bincount(y, minlength=16) for y in self.Y]).double()
        self.log_pi = ((self.counts + 1) / (self.counts + 1).sum(1, keepdim=True)).log().float()
        self.log_pg = ((self.counts.sum(0) + 1) / (self.counts.sum() + 16)).log().float()
        self.la_shift = None   # la: CE(z + log pi_k - log pi_g), set from the SecureSum prior in Method.prior() (round 18 fix)
        # private-only Gaussian labels per client: the class cycle, or (fill) the integer water-fill of the local prior
        cycle = torch.arange(args.aux_batch, device=self.dev) % 16
        self.aux_labels = [fill_labels(self.log_pi[k].exp(), args.aux_batch).to(self.dev) if self.fill else cycle
                           for k in range(self.K)]
        # gpe rule (round 13, C): the prior the private's two CE terms train toward, (pi_k + w q_k) / (1 + w); w = aux weight
        self.log_pe = None
        if self.paux and not self.replay:
            q = torch.stack([torch.bincount(l, minlength=16).float() / len(l) for l in self.aux_labels])
            self.log_pe = ((self.log_pi.exp() + self.aux_w * q) / (1 + self.aux_w)).log().float()
        self.rules = ('head', 'gp', 'gpe') if self.log_pe is not None else ('head', 'gp')
        self.architecture = dict(P.CFG, n_features=53, num_classes=P.C, input_pad=1) if self.drop53 else \
                            dict(P.CFG, n_features=P.D, num_classes=P.C)
        torch.manual_seed(args.seed)
        self.init = P.build_model(self.architecture, latent=True).to(self.dev) if self.drop53 else P.make('dagsnet').to(self.dev)
        self.priv = [copy.deepcopy(self.init) for _ in range(self.K)]
        self.work_private = copy.deepcopy(self.init); self.work_proxy = copy.deepcopy(self.init)
        self.eval_template = copy.deepcopy(self.init).eval()
        self.work_teacher = copy.deepcopy(self.init).eval() if self.frz else None
        self.syn_teacher = copy.deepcopy(self.init).eval() if self.rej or self.p1 or self.gkd else None
        self.cv_model = copy.deepcopy(self.init).eval() if self.cv else None   # round 25: eager snapshot copy for the class moments
        self.cv_seconds = 0.
        self.compiled = {}; self.compile_gates = []; self.compiled_more = {}   # round 24: a second compiled size per model (xo)
        if args.backend == 'compiled' and getattr(args, 'evaluate_only', False):
            self.compile_extra(self.eval_template, 'teacher', 16384)
        elif args.backend == 'compiled':
            arm = args.arm; args.arm = arm + '_dml' if self.fedavg else arm   # frozen preflight compiles the proxy only for 'dml'
            batch = args.batch; args.batch = 2 * batch if self.cons else batch   # cons: both views in one forward
            self.compile_preflight(); args.arm = arm; args.batch = batch
            if self.frz: self.compile_extra(self.work_teacher, 'teacher', args.batch)
            if self.paux: self.compile_extra(self.work_private, 'private', args.batch + args.aux_batch)   # proxy stays at batch
            if self.xo: self.compile_extra(self.work_private, 'private', args.batch + 2 * args.aux_batch)   # + crossover rows
            if self.syn_teacher is not None: self.compile_extra(self.syn_teacher, 'teacher', args.aux_batch * (2 if self.xo else 1))
        self.swap_stats = [0, 0]
        # hkd (FedKD hidden-state loss): client-local projector private h -> proxy h, never uploaded (R1), any private width (R2)
        self.hkd = 'hkd' in tokens; self.proj = []
        for _ in range(self.K if self.hkd else 0):
            width = P.latent_dim(self.init)   # private width -> proxy width (both DAGSNet here)
            self.proj.append(torch.nn.Linear(width, width, bias=False).to(self.dev)); torch.nn.init.eye_(self.proj[-1].weight)
        self.by_class = []
        for y in (self.Y if self.replay else []):
            count = torch.bincount(y, minlength=16); order = torch.argsort(y, stable=True)
            self.by_class.append((order, torch.cumsum(count, 0) - count, count, torch.nonzero(count).squeeze(1)))
        self.xo_index = []   # round 24: own rows sorted by class (order, start, count) for the crossover draws
        for y in (self.Y if self.xo else []):
            count = torch.bincount(y, minlength=16); self.xo_index.append((torch.argsort(y, stable=True), torch.cumsum(count, 0) - count, count))
        cal = np.random.default_rng(4242)
        self.cal = [torch.from_numpy(np.sort(cal.choice(n, min(n, CAL_ROWS), replace=False))).to(self.dev) for n in self.n]

    def compile_preflight(self):
        """full_run.compile_preflight on this process's own client rows (round 18; it read client 0): state and RNG restored."""
        rows = self.X; self.X = [rows[self.own[0]]]
        if self.drop53 or len(self.zero): self.X = [_Masked(self.X[0], self)]
        try: super().compile_preflight()
        finally: self.X = rows

    def compile_extra(self, model, role, batch):
        """One more compiled forward (frozen teacher in eval mode, or the private at batch + private-only rows), same gate as
        full_run.compile_preflight: dropout off on both copies, BN state and CUDA RNG restored."""
        training = role != 'teacher'; model.train(training)
        x = self.X[self.own[0]][:batch].float()   # round 18: this process's own client rows (was client 0)
        if len(x) < batch: x = x.repeat((batch + len(x) - 1) // len(x), 1)[:batch]
        if self.drop53 or len(self.zero): x = self.mask(x)
        saved = {k: v.clone() for k, v in model.state_dict().items()}; rng = torch.cuda.get_rng_state()
        dropout = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        for m, _ in dropout: m.p = 0.
        reference = copy.deepcopy(model); compiled = torch.compile(model, mode='reduce-overhead', dynamic=False)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            torch.compiler.cudagraph_mark_step_begin(); expected = reference(x)[0].float()
            actual = compiled(x)[0].float().clone()
        error = float((actual - expected).abs().max()); decisive = expected.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1) > .02
        assert error <= .005 and torch.equal(actual.argmax(1)[decisive], expected.argmax(1)[decisive]), ('compile gate', role, error)
        model.load_state_dict(saved, strict=True); torch.cuda.set_rng_state(rng)
        for m, p in dropout: m.p = p
        if id(model) in self.compiled: self.compiled_more[(id(model), batch)] = compiled
        else: self.compiled[id(model)] = (compiled, batch)
        self.compile_gates.append(dict(role=role, rows=batch, max_error=error, decisive=int(decisive.sum())))
        print('COMPILE EXTRA PASS', self.compile_gates[-1], flush=True)

    def forward(self, model, x):
        more = self.compiled_more.get((id(model), len(x)))   # round 24 xo: the private at batch + Gaussian + crossover rows
        return more(x) if more is not None else super().forward(model, x)

    def xo_weights(self, k, private, teacher):
        """Round 24: class-pair weights (16 x 16 flattened, cumulative) for the crossover draws, over the pairs a != b the client
        holds both of. xo: (1 - XO_FLOOR) x the (teacher argmax a, private argmax b) histogram on the calibration rows (epoch-start
        private vs the received snapshot) + XO_FLOOR uniform; xou: uniform. None when the client holds fewer than two classes."""
        held = (self.xo_index[k][2] > 0).float(); both = (held[:, None] * held[None, :] * (1 - torch.eye(16, device=self.dev))).flatten()
        if float(both.sum()) == 0: return None, {}
        w = both / both.sum()
        if self.xo == 'xo':
            rows = self.X[k][self.cal[k]]; a, b = self.logits(teacher, rows).argmax(1), self.logits(private, rows).argmax(1)
            hist = torch.zeros(256, device=self.dev).index_add_(0, a * 16 + b, torch.ones(len(rows), device=self.dev)) * both
            w = (1 - XO_FLOOR) * hist / hist.sum().clamp_min(1) + XO_FLOOR * w
        w = w / w.sum(); top = w.topk(3)   # no own-row disagreement: the floor alone, i.e. xou
        # sequential CPU scan: zero-weight pairs keep an exactly flat cdf (the GPU scan leaves ~1e-7 there)
        return w.cpu().cumsum(0).to(self.dev), dict(xo_pairs=[[int(i) // 16, int(i) % 16, round(float(v), 4)] for v, i in zip(*top)],
                                 xo_disagree_own=float((a != b).float().mean()) if self.xo == 'xo' else None)

    @torch.no_grad()
    def cv_diagnostics(self, method, k, private):
        """Round 25, own calibration rows and own labels only: confusions of the snapshot with its trained / calibrated head, and
        the private's (epoch start) wrong rows each head gets right."""
        h, y = self.cv_features(method.global_state, k)
        head = lambda w, b: (h @ w.float().T + b.float()).argmax(1)
        t0 = head(method.global_state['head.5.weight'], method.global_state['head.5.bias'])
        t1 = head(method.cv_head['head.5.weight'], method.cv_head['head.5.bias'])
        p = self.logits(private, self.X[k][self.cal[k]]).argmax(1); wrong = p != y
        cm = lambda pred: torch.bincount(y * 16 + pred, minlength=256).view(16, 16).tolist()
        return dict(cv_cm_orig=cm(t0), cv_cm_cal=cm(t1), cv_teach=[int(wrong.sum()), int((wrong & (t0 == y)).sum()), int((wrong & (t1 == y)).sum())])

    def crossover(self, k, cdf, gen):
        """Round 24: aux_batch rows, each an own row of class a whose features come w.p. XO_RHO from an own row of class b,
        (a, b) drawn from the cumulative pair weights (no host sync); masked like the real rows."""
        order, start, count = self.xo_index[k]; n = self.a.aux_batch
        v = torch.searchsorted(cdf, torch.rand(n, device=self.dev, generator=gen) * cdf[-1], right=True).clamp_max(255)
        def rows(c): return self.X[k][order[start[c] + (torch.rand(n, device=self.dev, generator=gen) * count[c]).long().clamp_max(count[c] - 1)]].float()
        xa, xb = rows(v // 16), rows(v % 16)
        return self.mask(torch.where(torch.rand(xa.shape, device=self.dev, generator=gen) < XO_RHO, xb, xa))

    @torch.no_grad()
    def cv_features(self, state, k):
        """Round 25: h256 (input of the last Linear) of the snapshot `state` on client k's calibration rows, with their labels."""
        model = self.cv_model; model.load_state_dict(state, strict=True); model.eval(); x = self.X[k][self.cal[k]]; out = []
        for s in range(0, len(x), 16384):
            with torch.autocast('cuda', dtype=torch.float16):
                out.append(model.head[:4](model.combined(self.mask(x[s:s + 16384].float()))).float())
        return torch.cat(out), self.Y[k][self.cal[k]]

    def cv_basis(self, state):
        """Round 26 cvp: orthonormal CV_PROJ-column basis of h256 whose span holds every row of the snapshot head.5 weight, completed
        by a fixed random complement (CPU float64 QR, seed 23): every client and the server derive it from the snapshot, no bytes."""
        g = torch.Generator(device='cpu').manual_seed(23); extra = torch.randn(256, 256, generator=g, dtype=torch.float64)
        w = state['head.5.weight'].double().cpu()
        return torch.linalg.qr(torch.cat([w.T, extra], 1), mode='reduced').Q[:, :CV_PROJ].to(self.dev)

    def cv_stats(self, state, k):
        """Round 25 client message (SecureSum-additive sums): class counts, class sums of h, sum of h h^T per class (cvf) or over
        all rows (cvl: the server derives one shared within-class covariance); cvp (round 26): the same per class in the projection."""
        h, y = self.cv_features(state, k); h = h.double(); onehot = F.one_hot(y, 16).double()
        if self.cv == 'cvp': h = h @ self.cv_basis(state)
        ss = torch.stack([h[y == c].T @ h[y == c] for c in range(16)]) if self.cv in ('cvf', 'cvp') else h.T @ h
        return dict(n=onehot.sum(0).cpu(), s=(onehot.T @ h).cpu(), ss=ss.cpu())

    def cv_calibrate(self, stats, state, r):
        """Round 25 server: pooled moments -> virtual h ~ N(mu_c, Sigma) at the pooled class shares -> the snapshot's last Linear
        re-fitted from its own weights (Adam, CV_STEPS x CV_BATCH). Returns the calibrated head.5 tensors (sent to every client)."""
        n = sum(v['n'] for v in stats).to(self.dev); s = sum(v['s'] for v in stats).to(self.dev); ss = sum(v['ss'] for v in stats).to(self.dev)
        mu = s / n[:, None].clamp_min(1); outer = n[:, None, None] * mu[:, :, None] * mu[:, None, :]
        eye = CV_JITTER * torch.eye(mu.shape[1], dtype=torch.float64, device=self.dev)
        cov = (ss - outer) / (n[:, None, None] - 1).clamp_min(1) + eye if self.cv in ('cvf', 'cvp') else (ss - outer.sum(0)) / (n.sum() - 16) + eye
        L, mu = torch.linalg.cholesky(cov).float(), mu.float()
        fc = torch.nn.Linear(mu.shape[1], 16).to(self.dev)
        basis = self.cv_basis(state).float() if self.cv == 'cvp' else None   # cvp: fit the head restricted to the projection
        with torch.no_grad():
            fc.weight.copy_(state['head.5.weight'].float() @ basis if basis is not None else state['head.5.weight'].float())
            fc.bias.copy_(state['head.5.bias'].float())
        opt = torch.optim.Adam(fc.parameters(), lr=CV_LR); g = self.gen(r, 0, 25); share = (n / n.sum()).float()
        for _ in range(CV_STEPS):
            c = torch.multinomial(share, CV_BATCH, replacement=True, generator=g)
            z = torch.randn(CV_BATCH, mu.shape[1], device=self.dev, generator=g)
            x = mu[c] + (z @ L.T if L.dim() == 2 else (L[c] @ z[:, :, None]).squeeze(-1))
            loss = F.cross_entropy(fc(x), c); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        assert torch.isfinite(fc.weight).all() and torch.isfinite(fc.bias).all(), ('cv head', r)
        weight = fc.weight.detach() @ basis.T if basis is not None else fc.weight.detach()
        return {'head.5.weight': weight.to(state['head.5.weight'].dtype),
                'head.5.bias': fc.bias.detach().to(state['head.5.bias'].dtype)}

    def cv_round(self, method, r, pool=None):
        """Round 25, before the local epochs: every client's class moments of the received snapshot, summed in client order,
        then the server fit; no calibration before the first aggregated snapshot. Bytes at 4 per value (fixed-point SecureSum),
        upper triangle of the symmetric sums; the calibrated head goes down to every client."""
        method.cv_head = None
        if method.version == 0: return
        started = time.perf_counter()
        stats = pool.cv_stats(method, self) if pool else {k: self.cv_stats(method.global_state, k) for k in range(self.K)}
        method.cv_head = self.cv_calibrate([stats[k] for k in range(self.K)], method.global_state, r)
        self.cv_seconds = time.perf_counter() - started
        width = CV_PROJ if self.cv == 'cvp' else 256
        up = 4 * (16 + 16 * width + (1 if self.cv == 'cvl' else 16) * width * (width + 1) // 2); down = 4 * sum(v.numel() for v in method.cv_head.values())
        for k in range(self.K): self.bytes += [(r, k, 'up', 'cv_stats', up), (r, k, 'down', 'cv_head', down)]

    @staticmethod
    def cv_install(model, head):
        with torch.no_grad(): model.head[5].weight.copy_(head['head.5.weight']); model.head[5].bias.copy_(head['head.5.bias'])

    def replay_rows(self, k, n, gen):
        """Class-balanced local replay: labels cycle over the classes this client holds, rows drawn uniformly within class."""
        order, start, count, present = self.by_class[k]
        labels = present[torch.arange(n, device=self.dev) % len(present)]
        pick = (torch.rand(n, device=self.dev, generator=gen) * count[labels]).long().clamp_max(count[labels] - 1)
        return self.mask(self.X[k][order[start[labels] + pick]].float()), labels

    def two_views(self, x, index, k, gen):
        """cons: view a keeps the receiver block, view b takes the whole 9-feature block of another receiver of the same client;
        both zero the 4 positions (swappos mask). Rows without a different-receiver donor keep their block (logged)."""
        n, G = len(index), self.G[k]
        donor = torch.randint(0, len(G), (n,), device=self.dev, generator=gen)
        for _ in range(3):
            same = G[donor] == G[index]
            donor = torch.where(same, torch.randint(0, len(G), (n,), device=self.dev, generator=gen), donor)
        ok = G[donor] != G[index]; rows = torch.nonzero(ok).squeeze(1)
        b = x.clone(); b[rows[:, None], self.block[None, :]] = self.X[k][donor[rows]][:, self.block].float()
        self.swap_stats[0] += int(ok.sum()); self.swap_stats[1] += n
        return self.mask(x), self.mask(b)

    def kd_probe(self, k, private, proxy, teacher, cap, scale=1.):
        """Fixed train probe batch (first PROBE_ROWS calibration rows), eval mode, fp32 eager, after the epoch: gradient norm
        of the private CE vs the private KD term toward the teacher it learned from; no state, RNG or optimizer is touched."""
        x, y = self.mask(self.X[k][self.cal[k][:PROBE_ROWS]].float()), self.Y[k][self.cal[k][:PROBE_ROWS]]
        modes = [(m, m.training) for m in (private, proxy, teacher)]
        for m, _ in modes: m.eval()
        params = list(private.parameters()); zs = private(x)[0].float()
        with torch.no_grad(): zr, zt = proxy(x)[0].float(), teacher(x)[0].float()
        ce = F.cross_entropy(zs, y); kl = F.kl_div(F.log_softmax(zs, 1), F.log_softmax(zt, 1), log_target=True, reduction='batchmean')
        norm = lambda gs: float(torch.sqrt(sum(g.double().square().sum() for g in gs)))
        g_ce = norm(torch.autograd.grad(ce, params, retain_graph=True)); g_kl = norm(torch.autograd.grad(kl, params))
        raw = 1 / max(float(ce) + float(F.cross_entropy(zr, y)), 1e-6); used = scale * min(raw, cap)
        for m, mode in modes: m.train(mode)
        return dict(probe_ce=float(ce), probe_kl=float(kl), probe_weight_raw=raw, probe_weight=used,
                    probe_grad_ce=g_ce, probe_grad_kl=g_kl, probe_ratio=used * g_kl / max(g_ce, 1e-30))

    def mask(self, x):
        if self.drop53: return x.index_select(1, self.keep)
        return x.masked_fill(self.zero_cols, 0) if len(self.zero) else x   # == clone + x[:, zero] = 0, without a host sync

    def transform(self, x, index, k, gen):
        n, G = len(index), self.G[k]
        apply = torch.rand(n, device=self.dev, generator=gen) < SWAP_P
        donor = torch.randint(0, len(G), (n,), device=self.dev, generator=gen)
        for _ in range(3):
            same = G[donor] == G[index]
            donor = torch.where(same, torch.randint(0, len(G), (n,), device=self.dev, generator=gen), donor)
        ok = apply & (G[donor] != G[index])
        rows = torch.nonzero(ok).squeeze(1)
        x = x.clone()
        x[rows[:, None], self.block[None, :]] = self.X[k][donor[rows]][:, self.block].float()
        self.swap_stats[0] += int(ok.sum()); self.swap_stats[1] += n
        return self.mask(x)

    @torch.no_grad()
    def logits(self, model, x):
        return super().logits(model, x) if not (self.drop53 or len(self.zero)) else super().logits(model, _Masked(x, self))

    def private_epoch(self, method, k, r):
        """full_run.Run.private_epoch + hooks: FedProx on the proxy (prox), receiver swap + position mask (swappos), receiver+position features zeroed (m4b),
        private learns the proxy only on not-true classes (ntd, FedNTD), private hidden layer matches the proxy's through a
        client-local projector (hkd, FedKD); round 8-9 hooks: kdcap/wu/frz, dyn, cons, sgd/scaf, gpool/gown/replay.
        Identity for fd_dml/fd_gaussian."""
        started = time.time()
        original = self.priv[k]; self.work_private.load_state_dict(original.state_dict(), strict=True)
        private = self.work_private; private.train()
        models = [] if self.fedavg else [private]; proxy = None
        if method.kd:
            proxy = self.work_proxy; proxy.load_state_dict(method.proxy[k].state_dict(), strict=True); proxy.train(); models.append(proxy)
        local = list(self.proj[k].parameters()) if self.hkd else []
        if self.sgd_lr:   # proxy on plain SGD (SCAFFOLD Option II assumes SGD), private keeps AdamW
            eta = self.sgd_lr * self.lr(r) / self.a.lr
            optimizer = Mixed(torch.optim.AdamW(list(private.parameters()) + local, lr=self.lr(r), weight_decay=0., fused=self.fused),
                              torch.optim.SGD(proxy.parameters(), lr=eta, momentum=0., weight_decay=0.))
        else:
            optimizer = torch.optim.AdamW([p for m in models for p in m.parameters()] + local, lr=self.lr(r), weight_decay=0.,
                                          fused=self.fused)
        scaler = torch.amp.GradScaler(); g = self.gen(r, k, 0)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed, r, k, 7]).generate_state(1)[0]))
        order = torch.randperm(self.n[k], device=self.dev, generator=g)
        gd = g   # private-only rows (gpool/gown/replay) share the epoch generator
        batches = (self.n[k] + self.a.batch - 1) // self.a.batch
        swap_gen = self.gen(r, k, 55, 0) if self.swap else None   # trailing 0 keeps the audited swap streams bit-exact
        anchor = None
        if self.mu or self.dyn or self.scaf:
            anchor = [method.global_state[name].detach().float() for name, _ in proxy.named_parameters()]
        a_dyn = method.dyn_a.get(k) or [torch.zeros_like(w) for w in anchor] if self.dyn else None
        corr = None
        if self.scaf:   # SCAFFOLD correction c - c_i on the proxy gradient (both start at 0)
            c_i = method.scaf_ci.get(k) or [torch.zeros_like(w) for w in anchor]
            c = method.scaf_c or [torch.zeros_like(w) for w in anchor]; method.scaf_c = c
            corr = torch._foreach_sub(c, c_i)
        teach = not (self.warm and method.version == 0)   # wu/frz: no private KD before the first aggregated snapshot
        teacher = None
        if self.frz:
            teacher = self.work_teacher; teacher.load_state_dict(method.global_state, strict=True); teacher.eval()
            if method.cv_head is not None: self.cv_install(teacher, method.cv_head)   # round 25: the calibrated head teaches
        wbuf = torch.zeros(batches, device=self.dev); kdstat = torch.zeros(4, device=self.dev)
        total = torch.zeros(5, device=self.dev); skips = torch.zeros((), dtype=torch.long, device=self.dev)
        nonfinite = torch.zeros((), dtype=torch.bool, device=self.dev); aux_labels = self.aux_labels[k]
        syn = None   # round 13: snapshot teacher on the Gaussian rows (eval mode, no grad), off before the first aggregated snapshot
        if self.syn_teacher is not None and method.version > 0:
            syn = self.syn_teacher; syn.load_state_dict((original if self.p1 == 'p1self' else method.proxy[k]).state_dict(), strict=True)
            syn.eval()
            if method.cv_head is not None: self.cv_install(syn, method.cv_head)
        cv_log = self.cv_diagnostics(method, k, original) if method.cv_head is not None else {}
        xo_cdf, xo_log = (self.xo_weights(k, original, syn) if self.xo and syn is not None else (None, {}))
        xo_gen = self.gen(r, k, 77) if xo_cdf is not None else None; xostat = torch.zeros(2, device=self.dev)   # disagree, rows
        drawn = torch.bincount(aux_labels, minlength=16).float(); p1_classes = torch.zeros(16, device=self.dev)
        p1_classes[list(P1_CLASSES)] = 1
        synstat = torch.zeros(3, 16, device=self.dev); synkl = torch.zeros(2, device=self.dev)   # accepted/drawn/gated; KL sum, gated
        proto_sum = torch.zeros(P.C, TGP_DIM, device=self.dev) if self.tgp else None
        proto_count = torch.zeros(P.C, device=self.dev) if self.tgp else None
        for start in range(0, len(order), self.a.batch):
            torch.compiler.cudagraph_mark_step_begin()
            index = order[start:start + self.a.batch]; x, y = self.X[k][index].float(), self.Y[k][index]
            if self.cons: x = torch.cat(self.two_views(x, index, k, swap_gen)); y = torch.cat([y, y])
            elif self.swap: x = self.transform(x, index, k, swap_gen)
            elif self.drop53 or len(self.zero): x = self.mask(x)
            real = len(y)
            labels = aux_labels
            if method.aux: x = torch.cat([x, self.mask(method.draw(labels, g, k))])
            xp = x   # private input: + private-only Gaussian draws (gpool/gown) or class-balanced local replay (replay)
            if method.paux: xp = torch.cat([x, self.mask(method.draw(labels, gd, k))])
            if xo_cdf is not None: xp = torch.cat([xp, self.crossover(k, xo_cdf, xo_gen)])   # round 24: after the Gaussian rows
            elif self.replay:
                rows, labels = self.replay_rows(k, self.a.aux_batch, gd); xp = torch.cat([x, rows])
            with torch.autocast('cuda', dtype=torch.float16):
                if self.fedavg:
                    if self.tgp:
                        encoded = proxy.combined(x).float(); logits = proxy.head(encoded).float()
                        proxy_ce = F.cross_entropy(logits, y); guidance = proxy_ce.detach() * 0
                        for label in torch.unique(y):
                            selected = y == label; mean = encoded[selected].mean(0)
                            guidance = guidance + (mean - method.tgp_proto[label]).square().mean()
                            proto_sum[label] += encoded[selected].detach().sum(0); proto_count[label] += selected.sum()
                        proxy_ce = proxy_ce + TGP_LAMBDA * guidance / max(int(torch.unique(y).numel()), 1)
                    else:
                        proxy_ce = F.cross_entropy(self.forward(proxy, x)[0].float(), y)
                    ce = extra = proxy_extra = proxy_ce.detach() * 0; loss = proxy_ce
                else:
                    if self.hkd: zs, hs = (v.float() for v in self.forward(private, x))
                    else: zs = self.forward(private, xp)[0].float()
                    aux_ce = F.cross_entropy(zs[real:real + len(labels)], labels) if self.paux else None
                    syn_kd = None
                    if syn is not None:
                        with torch.no_grad(): zt = self.forward(syn, xp[real:])[0].float()
                        if xo_cdf is not None: zt, zx = zt[:len(labels)], zt[len(labels):]   # Gaussian rows, crossover rows
                        accept = (zt.argmax(1) == labels).float()
                        kept = torch.zeros(16, device=self.dev).index_add_(0, labels, accept)
                        if self.rej:   # drop rejected rows; accepted rows of a class carry that class's Gaussian mass
                            w = accept * drawn[labels] / (len(labels) * kept[labels].clamp_min(1))
                            aux_ce = (w * F.cross_entropy(zs[real:], labels, reduction='none')).sum()
                        gate = accept * p1_classes[labels]
                        if self.p1:   # KL(teacher || private) on gated rows, mean over all m Gaussian rows; hard CE kept
                            lt, ls_syn = F.log_softmax(zt / P1_T, 1), F.log_softmax(zs[real:] / P1_T, 1)
                            if self.p1 == 'p1cm':   # x-independent control: per-class mean of the gated teacher probabilities
                                mass = torch.zeros(16, device=self.dev).index_add_(0, labels, gate).clamp_min(1)
                                mean = torch.zeros(16, 16, device=self.dev).index_add_(0, labels, lt.exp() * gate[:, None]) / mass[:, None]
                                kl = F.kl_div(ls_syn, mean[labels], reduction='none').sum(1)
                            else:
                                kl = F.kl_div(ls_syn, lt, log_target=True, reduction='none').sum(1)
                            syn_kd = P1_LAMBDA * P1_T ** 2 * (gate * kl).sum() / len(labels)
                            synkl += torch.stack([(gate * kl).sum().detach(), gate.sum()])
                        if self.gkd:   # KL(snapshot || private) on every Gaussian row replaces the hard CE (weighted like the KD below)
                            aux_ce = None
                            T = KD_T if self.t2 else 1.
                            syn_kd = F.kl_div(F.log_softmax(zs[real:real + len(labels)] / T, 1), F.log_softmax(zt / T, 1), log_target=True, reduction='batchmean') * T * T
                            if xo_cdf is not None:   # round 24: crossover rows learn the snapshot by KL, same weight as the Gaussian rows
                                zxs = zs[real + len(labels):]
                                syn_kd = syn_kd + F.kl_div(F.log_softmax(zxs, 1), F.log_softmax(zx, 1), log_target=True, reduction='batchmean')
                                xostat += torch.stack([(zx.argmax(1) != zxs.argmax(1)).float().sum().detach(), zxs.new_tensor(float(len(zxs)))])
                        synstat += torch.stack([kept, drawn, torch.zeros(16, device=self.dev).index_add_(0, labels, gate)])
                    if self.paux: zs = zs[:real]   # KD and JS on real rows only
                    ce = F.cross_entropy(zs[:real] + self.la_shift[k] if self.la else zs[:real], y); extra = ce.detach() * 0
                    proxy_ce = proxy_extra = ce.detach() * 0
                    if method.aux: extra = F.cross_entropy(zs[real:], labels)
                    if proxy is not None:
                        if self.hkd: zr, hr = (v.float() for v in self.forward(proxy, x))
                        else: zr = self.forward(proxy, x)[0].float()
                        proxy_ce = F.cross_entropy(zr + self.la_shift[k] if self.lap else zr, y)
                        ls, lr = F.log_softmax(zs, 1), F.log_softmax(zr, 1)
                        raw = 1 / (ce + proxy_ce).detach().clamp_min(1e-6)
                        weight = raw.clamp_max(KD_CAP) if self.kdcap else raw
                        wpriv = KD_UP * weight if self.kdup else weight   # private <- proxy
                        wproxy = raw.clamp_max(KD_CAP) if self.kdpx else weight   # proxy <- private
                        if self.pce: wproxy = raw * 0   # round 18: the proxy learns CE only (FedAvg proxy)
                        if self.ntd:
                            kd = F.kl_div(not_true(zs, y), not_true(zr, y).detach(), log_target=True, reduction='batchmean')
                        else:
                            target = lr.detach()
                            if self.frz and teach:
                                with torch.no_grad(): zt_real = self.forward(teacher, x)[0].float()
                                target = F.log_softmax(zt_real / (KD_T if self.t2 else 1.), 1)
                            if self.t2 and self.frz and teach:   # round 18: KL(teacher || private) at temperature KD_T, x T^2
                                kd = F.kl_div(F.log_softmax(zs / KD_T, 1), target, log_target=True, reduction='batchmean') * KD_T * KD_T
                            else:
                                kd = F.kl_div(ls, target, log_target=True, reduction='batchmean')
                        extra = wpriv * kd if teach else ce.detach() * 0
                        if self.hkd:   # scale-free MSE to the proxy's 256-d hidden layer, same adaptive weight as the KL
                            target = hr.detach()
                            extra = extra + weight * (self.proj[k](hs).float() - target).square().mean() / target.square().mean().clamp_min(1e-6)
                        kl_proxy = F.kl_div(lr, ls.detach(), log_target=True, reduction='batchmean')
                        proxy_extra = wproxy * kl_proxy
                        if self.lam: extra = extra + self.lam * js(zs[:len(index)], zs[len(index):real])
                        if aux_ce is not None: extra = extra + self.aux_w * aux_ce
                        if syn_kd is not None: extra = extra + ((GK2 if self.gk2 else 1.) * wpriv * syn_kd if self.gkd else syn_kd)
                        wbuf[start // self.a.batch] = raw
                        step_stat = torch.stack([raw, (raw > KD_CAP).float(), kd.detach(), kl_proxy.detach()])
                    loss = ce + extra + proxy_ce + proxy_extra
            prox = ce.detach() * 0
            if anchor is not None and self.mu:
                prox = 0.5 * MU * sum((p.float() - a).square().sum() for p, a in zip(proxy.parameters(), anchor))
                loss = loss + prox
            nonfinite |= ~torch.isfinite(loss)   # asserted after the epoch: no per-step host sync
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if self.dyn:   # FedDyn local objective F_i(w) - <a_i, w> + alpha/2 ||w - w_t||^2: its gradient added before clipping
                params = list(proxy.parameters())
                reg = torch._foreach_sub([p.detach() for p in params], anchor); torch._foreach_mul_(reg, DYN_ALPHA)
                torch._foreach_sub_(reg, a_dyn); torch._foreach_add_([p.grad for p in params], reg)
            for model in models:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.a.clip)
                assert scaler.is_enabled() or torch.isfinite(norm)
            if local: torch.nn.utils.clip_grad_norm_(local, self.a.clip)
            if corr is not None: torch._foreach_add_([p.grad for p in proxy.parameters()], corr)   # after clipping g_i
            before = scaler._get_scale_async().clone(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            skip = scaler._get_scale_async() < before; skips += skip   # the scale backs off only on a skipped (inf/NaN) step
            total += torch.where(skip, 0., torch.stack([ce.detach(), extra.detach(), proxy_ce.detach(), proxy_extra.detach(), prox.detach()]))
            if proxy is not None and not self.fedavg: kdstat += torch.where(skip, 0., step_stat)
            if self.a.wandb_online and (start // self.a.batch + 1) % 1024 == 0:
                import wandb
                skipped = int(skips); applied = start // self.a.batch + 1 - skipped
                wandb.run.log({'progress/round_in_flight': r, 'progress/client': k,
                               'progress/client_steps': start // self.a.batch + 1, 'progress/applied': applied,
                               'progress/skipped': skipped, 'progress/seconds': time.time() - started,
                               'progress/loss': float(loss.detach()), 'progress/grad_norm': float(norm)})
        assert not nonfinite, ('full loss', k, r)
        assert not method.draw_bad, 'V7 synthetic fp16 input overflow'
        skipped = int(skips); applied = batches - skipped
        assert applied > 0 and skipped <= 16, (k, r, applied, skipped)
        for model in models: finite(model)
        kd = {}
        if self.tgp: kd['tgp_payload'] = dict(sum=proto_sum.cpu().numpy(), count=proto_count.cpu().numpy())
        if proxy is not None and not self.fedavg:   # KD weight/coefficient diagnostics (plan §4.2), no effect on training
            w = wbuf.cpu(); s = (kdstat / applied).tolist()
            kd = dict(w_median=float(w.median()), w_p95=float(torch.quantile(w, .95)), w_max=float(w.max()), cap_hit=s[1],
                      kl_private=s[2], kl_proxy=s[3])
            kd.update(self.kd_probe(k, private, proxy, teacher if self.frz else proxy,
                                    (KD_CAP if self.kdcap else float('inf')) if teach else 0., KD_UP if self.kdup else 1.))
        if syn is not None:   # teacher acceptance of the Gaussian rows (overall, per class) and the P1 gate/KL
            kept, total_drawn, gated = synstat
            kd.update(syn_accept=float(kept.sum() / total_drawn.sum()), synacc=(kept / total_drawn.clamp_min(1)).tolist(),
                      syn_gate=float(gated.sum() / total_drawn.sum()))
            if self.p1: kd['syn_kl'] = float(synkl[0] / synkl[1].clamp_min(1))
        if xo_cdf is not None: kd.update(xo_log, xo_disagree=float(xostat[0] / xostat[1].clamp_min(1)))
        kd.update(cv_log)
        if self.dyn:
            method.dyn_a[k] = [a - DYN_ALPHA * (p.detach().float() - w) for a, p, w in zip(a_dyn, proxy.parameters(), anchor)]
        if self.scaf:   # Option II: c_i+ = c_i - c + (w_base - w_i) / (K eta), K = applied SGD steps; up = delta c_i
            new = [ci - cc + (w - p.detach().float()) / (applied * eta) for ci, cc, w, p in zip(c_i, c, anchor, proxy.parameters())]
            delta = torch._foreach_sub(new, c_i); method.scaf_ci[k] = new
            method.scaf_dsum = delta if method.scaf_dsum is None else torch._foreach_add(method.scaf_dsum, delta)
            self.bytes.append((r, k, 'up', 'control', 4 * sum(d.numel() for d in delta)))
            kd['scaf_corr_norm'] = sum(float(v.square().sum()) for v in corr) ** 0.5
        if not self.fedavg: original.load_state_dict(private.state_dict(), strict=True)
        drift = None
        if proxy is not None:
            method.proxy[k].load_state_dict(proxy.state_dict(), strict=True)
            num = sum(float((p.detach().float() - method.global_state[n].float()).square().sum()) for n, p in proxy.named_parameters())
            den = sum(float(method.global_state[n].float().square().sum()) for n, _ in proxy.named_parameters())
            drift = (num / max(den, 1e-30)) ** 0.5
        ce, extra, proxy_ce, proxy_extra, prox = (total / applied).tolist()
        common = dict(client=k, rows=self.n[k], applied=applied, skipped=skipped)
        method.logs[k] = dict(common, loss=proxy_ce + proxy_extra, positive_ce=proxy_ce, negative_ce=proxy_extra,
                              prox=prox, drift=drift)
        return dict(common, loss=ce + extra, real_ce=ce, auxiliary=extra, seconds=time.time() - started, **kd)

    def client_round(self, method, k, r):
        """Client k's local epoch and upload, detached from the server so any process can run it; install() applies it."""
        if method.kd: method.prepare(k, r)
        start, swap = len(self.bytes), list(self.swap_stats)
        encoding_before = method.diag['codec_up_encode_seconds']
        log = self.private_epoch(method, k, r); up = None
        if method.kd:
            if method.bits: up = ('q', method.quantized_wire(k, r))
            else:
                up = ('p', method.proxy_epoch(k, r)[0])
        rows = self.bytes[start:]; del self.bytes[start:]
        delta = [a - b for a, b in zip(self.swap_stats, swap)]; self.swap_stats = swap
        cpu = lambda model: {n: v.detach().cpu() for n, v in model.state_dict().items()}
        ef = {n: v.cpu() for (c, n), v in method.ef_state.items() if c == k} if getattr(method, 'ef', False) else None
        return dict(k=k, log=log, logs=method.logs[k], rows=rows, swap=delta, up=up,
                    codec_up_encode_seconds=method.diag['codec_up_encode_seconds'] - encoding_before,
                    private=None if self.fedavg else cpu(self.priv[k]),
                    proxy=cpu(method.proxy[k]) if method.kd and method.own else None, ef=ef)

    def centralized_epoch(self, method, r):
        """One pooled epoch: one optimizer/model and a global row permutation, assembled in bounded GPU chunks."""
        started = time.time(); model = self.work_proxy; model.load_state_dict(method.global_state, strict=True); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr(r), weight_decay=0., fused=self.fused)
        scaler = torch.amp.GradScaler(); g = self.gen(r, 0, 901)
        bounds = torch.tensor([0, *np.cumsum(self.n).tolist()], device=self.dev, dtype=torch.long)
        order = torch.randperm(int(bounds[-1]), device=self.dev, generator=g)
        total = torch.zeros(2, device=self.dev); skips = torch.zeros((), dtype=torch.long, device=self.dev)
        chunk = 1 << 20; batches = 0
        for offset in range(0, len(order), chunk):
            ids = order[offset:offset + chunk]; owner = torch.bucketize(ids, bounds[1:-1], right=True)
            xb = torch.empty((len(ids), P.D), device=self.dev, dtype=self.X[0].dtype)
            yb = torch.empty(len(ids), device=self.dev, dtype=self.Y[0].dtype)
            for k in range(self.K):
                pos = torch.nonzero(owner == k).squeeze(1)
                if len(pos):
                    local = ids[pos] - bounds[k]; xb[pos] = self.X[k][local]; yb[pos] = self.Y[k][local]
            for start in range(0, len(ids), self.a.batch):
                batches += 1
                x, y = self.mask(xb[start:start + self.a.batch].float()), yb[start:start + self.a.batch]
                with torch.autocast('cuda', dtype=torch.float16): loss = F.cross_entropy(self.forward(model, x)[0].float(), y)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.a.clip)
                before = scaler._get_scale_async().clone(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                skip = scaler._get_scale_async() < before; skips += skip
                total += torch.where(skip, 0., torch.stack([loss.detach(), norm.detach()]))
        skipped = int(skips); applied = batches - skipped; skip_budget = 16 * self.K
        print('CENTRALIZED AMP', dict(batches=batches, applied=applied, skipped=skipped, skip_budget=skip_budget), flush=True)
        assert applied > 0 and skipped <= skip_budget; finite(model)
        method.global_state = {n: v.detach().clone() for n, v in model.state_dict().items()}; method.version = r
        for replica in method.proxy: replica.load_state_dict(method.global_state, strict=True)
        common = dict(client=-1, rows=sum(self.n), applied=applied, skipped=skipped)
        method.logs[0] = dict(common, loss=float(total[0] / applied), positive_ce=float(total[0] / applied),
                              negative_ce=0., prox=0., drift=0.)
        return dict(common, loss=float(total[0] / applied), real_ce=float(total[0] / applied), auxiliary=0.,
                    grad_norm=float(total[1] / applied), seconds=time.time() - started)

    def personalize_canperfl(self, method, out, holds):
        """CanPer-FL Fed-FN stage: freeze shared representation, fine-tune each client's head on at most 30k own rows."""
        started = time.time(); models, clients = [], []
        for k in range(self.K):
            model = P.build_model(self.architecture, latent=True).to(self.dev)
            model.load_state_dict(method.global_state, strict=True)
            for p in model.parameters(): p.requires_grad_(False)
            for p in model.head.parameters(): p.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0., fused=self.fused)
            scaler = torch.amp.GradScaler(); g = self.gen(self.a.stop_after + 1, k, 710)
            X, Y = self.X[k], self.Y[k]
            chosen = torch.randperm(self.n[k], device=self.dev, generator=g)[:min(self.n[k], PERSONAL_MAX_ROWS)]
            applied = skipped = 0; loss_sum = torch.zeros((), device=self.dev)
            model.train()
            for _ in range(PERSONAL_EPOCHS):
                order = chosen[torch.randperm(len(chosen), device=self.dev, generator=g)]
                for start in range(0, len(order), self.a.batch):
                    idx = order[start:start + self.a.batch]; x, y = self.mask(X[idx].float()), Y[idx]
                    with torch.autocast('cuda', dtype=torch.float16): loss = F.cross_entropy(model(x)[0].float(), y)
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), self.a.clip)
                    before = scaler._get_scale_async().clone(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if bool(scaler._get_scale_async() < before): skipped += 1
                    else: applied += 1; loss_sum += loss.detach()
            assert applied > 0 and skipped <= 16; finite(model); model.eval()
            clients.append(dict(client=k, rows=len(chosen), epochs=PERSONAL_EPOCHS, applied=applied, skipped=skipped,
                                loss=float(loss_sum / applied)))
            models.append(model)
            del X
        arrays, evaluated = {}, {}
        for hold in holds:
            y = self.H[hold][1]; arrays[hold + '_y'] = y.cpu().numpy().astype('uint8')
            cms, preds = [], []
            for k, model in enumerate(models):
                cm, pred = self.score(model, k, hold, ('head',))['head']; cms.append(cm); preds.append(pred)
            mm = [metrics_from_confusion(cm) for cm in cms]
            pc = np.mean([[x['f1'] for x in per_class_from_confusion(cm, P.CLASSES)] for cm in cms], axis=0)
            evaluated[hold] = {'head': dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                             f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                             f1_per_class=pc.tolist(), confusion_clients=np.stack(cms).tolist())}
            arrays[f'personalized_{hold}_head'] = np.stack(preds)
        folder = out / 'personalized'; folder.mkdir(exist_ok=True)
        with (folder / 'predictions.tmp').open('wb') as f: np.savez_compressed(f, **arrays)
        os.replace(folder / 'predictions.tmp', folder / 'predictions.npz')
        save(folder / 'weights.pt', dict(architecture=self.architecture,
             clients=[{n: v.detach().cpu() for n, v in model.state_dict().items()} for model in models]))
        result = dict(method='Fed-FN', epochs=PERSONAL_EPOCHS, max_rows=PERSONAL_MAX_ROWS, clients=clients,
                      eval=evaluated, seconds=time.time() - started,
                      predictions_sha=digest(folder / 'predictions.npz'), weights_sha=digest(folder / 'weights.pt'))
        save(folder / 'metrics.json', result)
        return result

    def install(self, method, res, r, private_logs, proxy_logs):
        """Server side of client_round(), called in client order: the same state and byte log as the serial loop in main()."""
        k = res['k']
        if k not in self.own: method.diag['codec_up_encode_seconds'] += res['codec_up_encode_seconds']
        if res['private'] is not None: self.priv[k].load_state_dict(res['private'], strict=True)
        method.logs[k] = res['logs']; private_logs.append(res['log'])
        self.bytes.extend(res['rows']); self.swap_stats = [a + b for a, b in zip(self.swap_stats, res['swap'])]
        if method.kd:
            kind, up = res['up']
            if kind == 'q': method.accumulate_quantized(k, r, up)
            elif up is not None:
                started = time.perf_counter(); method.aggregate_sum.add(k, up)
                method.diag['codec_up_decode_seconds'] += time.perf_counter() - started
            if res['proxy'] is not None: method.proxy[k].load_state_dict(res['proxy'], strict=True)
            proxy_logs.append(method.logs[k])
        if res.get('ef'):   # copy of the client's residual for rank 0's resume bundle (the owning process keeps computing with its own)
            method.ef_state.update({(k, n): v.to(self.dev) for n, v in res['ef'].items()})

    @torch.no_grad()
    def score(self, model, k, hold, rules):
        """One model on one holdout: per rule, the 16x16 confusion and the uint8 predictions."""
        X, y = self.H[hold]; z = self.logits(model, X); assert torch.isfinite(z).all(); out = {}
        for rule in rules:
            bias = self.log_pg - self.log_pi[k] if rule == 'gp' else self.log_pg - self.log_pe[k] if rule == 'gpe' else 0.
            pred = (z + bias).argmax(1)
            out[rule] = (torch.bincount(y * P.C + pred, minlength=P.C ** 2).view(P.C, P.C).cpu().numpy(), pred.cpu().numpy().astype('uint8'))
        return out

    def score_clients(self, holds):
        """Private models of this process's clients (head and gp) on each holdout, for rank 0's evaluate()."""
        return {(k, hold): self.score(self.priv[k], k, hold, self.rules) for k in self.own for hold in holds}

    @torch.no_grad()
    def evaluate(self, m, out, r, holds, scores=None):
        """Ten metrics per client; the shared DML proxy is one model after aggregation, so it is scored once. scores: private
        (k, hold) results already computed by the worker processes (--workers)."""
        arrays, evaluated = {}, {}
        roles = {} if self.fedavg else {'private': (self.priv, self.rules)}
        if m.ema_private is not None:
            models = []
            for k in range(self.K):
                model = copy.deepcopy(self.priv[k]); model.load_state_dict(m.ema_private[k], strict=True)
                calibrate(model, self.mask(self.X[k][self.cal[k]].float())); models.append(model)
            roles['private_ema'] = (models, ('head', 'gp'))
        if m.kd and not m.own:   # own: 20 local proxies, never a shared model; the client target is the private model
            for k in range(self.K): m.prepare(k, r)
            for k in range(1, self.K):
                assert all(torch.equal(a, b) for a, b in zip(m.proxy[0].state_dict().values(), m.proxy[k].state_dict().values()))
            roles['proxy'] = ([m.proxy[0]], ('head',))
        for role, (models, rules) in roles.items():
            results = {}
            for hold in holds:
                y = self.H[hold][1]
                arrays[hold + '_y'] = y.cpu().numpy().astype('uint8'); result = {}
                cms = {rule: [] for rule in rules}; preds = {rule: [] for rule in rules}
                for k, model in enumerate(models):
                    scored = scores[(k, hold)] if role == 'private' and scores is not None else self.score(model, k, hold, rules)
                    for rule in rules:
                        cms[rule].append(scored[rule][0]); preds[rule].append(scored[rule][1])
                for rule, cc in cms.items():
                    cc = cc * self.K if len(cc) == 1 else cc     # shared proxy: identical confusion for every client
                    mm = [metrics_from_confusion(c) for c in cc]
                    pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cc], axis=0)
                    result[rule] = dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                        f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                        f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
                    arrays[f'{role}_{hold}_{rule}'] = np.stack(preds[rule])
                results[hold] = result
            evaluated[role] = results
        if self.tgp:
            results = {}
            model = m.proxy[0].eval()
            for hold in holds:
                X, y = self.H[hold]; preds = []
                for start in range(0, len(y), 16384):
                    h = model.combined(self.mask(X[start:start + 16384].float())).float()
                    preds.append(torch.cdist(h, m.tgp_proto.float()).argmin(1))
                pred = torch.cat(preds); cm = torch.bincount(y * P.C + pred, minlength=P.C ** 2).view(P.C, P.C).cpu().numpy()
                mm = metrics_from_confusion(cm); pc = [x['f1'] for x in per_class_from_confusion(cm, P.CLASSES)]
                results[hold] = {'nearest': dict(mean_metrics={key: float(mm[key]) for key in METRIC_KEYS},
                                                  f1_macro=float(mm['f1_macro']), benign_f1=float(pc[0]),
                                                  f1_per_class=list(map(float, pc)), confusion_clients=np.stack([cm] * self.K).tolist())}
                arrays[f'prototype_{hold}_nearest'] = np.stack([pred.cpu().numpy().astype('uint8')] * self.K)
            evaluated['prototype'] = results
        pf = out / 'predictions' / f'round_{r:03d}.npz'; pf.parent.mkdir(exist_ok=True)
        tmp = pf.with_suffix('.tmp')
        with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
        os.replace(tmp, pf)
        return evaluated


class _Masked:
    """Lazy masked view so the frozen chunked/compiled evaluation path stays unchanged."""
    def __init__(self, x, run): self.x, self.run = x, run
    def __len__(self): return len(self.x)
    def __getitem__(self, s): return self.run.mask(self.x[s].float())


def deterministic(fast=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False   # speed only: no code reads uninitialized memory
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if fast:   # --fast (owner 2026-10-18, tolerance class): cuDNN autotune + nondeterministic kernels; not bit-reproducible
        torch.use_deterministic_algorithms(False); torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False


def put(conn, value):   # plain pickle: torch's ForkingPickler would share tensors back to the sender (wrong authkey)
    conn.send_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def get(conn):
    return pickle.loads(conn.recv_bytes())


class Pool:
    """Rank 0 side of --workers N: N-1 worker processes (same command line + --rank/--address), GPUs round-robin by rank.
    Each round every process trains its own clients from the broadcast server state; rank 0 installs all clients in index
    order, so state, bytes and logs equal the serial loop (client epochs do not depend on process or order)."""
    def __init__(self, a, out):
        key = secrets.token_bytes(16); listener = Listener(('127.0.0.1', 0), authkey=key)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES') or ','.join(map(str, range(torch.cuda.device_count())))
        gpus = [g for g in visible.split(',') if g]; self.procs, self.conns, self.logs = [], {}, []
        for rank in range(1, a.workers):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[rank % len(gpus)], FD_AUTHKEY=key.hex())
            self.logs.append(open(out / f'worker{rank}.log', 'w'))
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--rank', str(rank), '--address', f'127.0.0.1:{listener.address[1]}']
            self.procs.append(subprocess.Popen(command, env=env, stdout=self.logs[-1], stderr=subprocess.STDOUT))
        for _ in range(a.workers - 1):
            conn = listener.accept(); self.conns[get(conn)] = conn
        listener.close(); print('workers', a.workers, 'gpus', [gpus[w % len(gpus)] for w in range(a.workers)], flush=True)

    def round(self, m, R, r):
        state = {n: v.detach().cpu() for n, v in m.global_state.items()} if m.kd else None
        head = None if m.cv_head is None else {n: v.detach().cpu() for n, v in m.cv_head.items()}   # round 25
        proto = m.tgp_proto.detach().cpu() if getattr(m, 'tgp', False) else None
        for conn in self.conns.values(): put(conn, ('round', r, state, m.version, head, proto))
        results = {res['k']: res for res in (R.client_round(m, k, r) for k in R.own)}
        for conn in self.conns.values(): results.update({res['k']: res for res in get(conn)})
        private_logs, proxy_logs = [], []
        for k in range(R.K): R.install(m, results[k], r, private_logs, proxy_logs)
        return private_logs, proxy_logs

    def cv_stats(self, m, R):   # round 25: class moments of the snapshot from every process's own clients
        state = {n: v.detach().cpu() for n, v in m.global_state.items()}
        for conn in self.conns.values(): put(conn, ('cv', state))
        merged = {k: R.cv_stats(m.global_state, k) for k in R.own}
        for conn in self.conns.values(): merged.update(get(conn))
        return merged

    def scores(self, R, holds):
        for conn in self.conns.values(): put(conn, ('eval', holds))
        merged = R.score_clients(holds)
        for conn in self.conns.values(): merged.update(get(conn))
        return merged

    def close(self):
        for conn in self.conns.values():
            try: put(conn, ('stop',))
            except OSError: pass
        for proc in self.procs:
            try: proc.wait(timeout=120)
            except subprocess.TimeoutExpired: proc.kill()
        for log in self.logs: log.close()


def serve(a):
    """Worker rank: the same deterministic Run/Method construction as rank 0, then train its clients per command."""
    host, port = a.address.split(':'); conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ['FD_AUTHKEY']))
    put(conn, a.rank); a.wandb_online = False
    deterministic(a.fast); R = Run(a); m = Method(R); m.prior()
    if a.parent: load_parent(a, R, m)
    print('worker', a.rank, 'clients', R.own, 'rows', sum(R.n[k] for k in R.own), flush=True)
    while True:
        command = get(conn)
        if command[0] == 'round':
            _, r, state, version, head, proto = command
            if state is not None: m.global_state = {n: v.to(R.dev) for n, v in state.items()}
            m.cv_head = None if head is None else {n: v.to(R.dev) for n, v in head.items()}
            if proto is not None: m.tgp_proto = proto.to(R.dev)
            m.version = version; m.start_round()
            put(conn, [R.client_round(m, k, r) for k in R.own])
        elif command[0] == 'eval':
            put(conn, R.score_clients(command[1]))
        elif command[0] == 'cv':
            put(conn, {k: R.cv_stats({n: v.to(R.dev) for n, v in command[1].items()}, k) for k in R.own})
        else:
            return


def parent_marker(a):
    """The parent run's config and its committed marker of --parent_round, whose weights file must match the marker SHA."""
    parent = Path(a.parent); cfg = json.loads((parent / 'config.json').read_text())
    row = json.loads((parent / 'complete' / f'round_{a.parent_round:03d}.json').read_text())
    assert row['round'] == a.parent_round and row['fingerprint'] == cfg['fingerprint'], 'parent marker'
    assert digest(parent / 'weights' / f'round_{a.parent_round:03d}.pt') == row['weights_sha'], 'parent weights sha'
    if 'resume_sha' in row: assert digest(parent / 'resume' / f'round_{a.parent_round:03d}.pt') == row['resume_sha'], 'parent resume sha'
    return cfg, row


def load_parent(a, R, m):
    """State after the parent's committed epoch: privates, their EMA (evaluated as private_ema) and the aggregated proxy. Exact
    only for arms whose cross-epoch state is just these weights (dense wire, no momentum/FedDyn/SCAFFOLD); generators and the
    optimizer restart every epoch. Quantized wires (q4/q2, ef) also restore the server's exact model and the client residuals from the
    resume bundle (round 21)."""
    assert m.kd and not m.own and not (m.momentum or m.dyn or m.scaf), 'state beyond weights'
    cfg, row = parent_marker(a)
    w = torch.load(Path(a.parent) / 'weights' / f'round_{a.parent_round:03d}.pt', map_location=R.dev, weights_only=True)
    assert w['round'] == a.parent_round and w['fingerprint'] == cfg['fingerprint'] and len(w['proxy']) == 1
    for model, state in zip(R.priv, w['private'], strict=True): model.load_state_dict(state, strict=True)
    m.ema_private = [{n: v.clone() for n, v in s.items()} for s in w['ema_private']] or None
    m.global_state = {n: v.clone() for n, v in w['proxy'][0].items()}
    for proxy in m.proxy: proxy.load_state_dict(m.global_state, strict=True)
    m.global_vector = m.layout.flatten(m.proxy[0]).clone(); m.version = a.parent_round
    if m.bits:
        z = torch.load(Path(a.parent) / 'resume' / f'round_{a.parent_round:03d}.pt', map_location=R.dev, weights_only=True)
        assert z['round'] == a.parent_round and z['fingerprint'] == cfg['fingerprint'] and set(z['exact']) == set(m.exact)
        m.exact = {n: v.clone() for n, v in z['exact'].items()}
        m.ef_state = {(int(key.split('/', 1)[0]), key.split('/', 1)[1]): v.clone() for key, v in z['ef'].items()}
        assert not m.ef or {k for k, _ in m.ef_state} == set(range(R.K)), 'ef residuals of every client'
    R.bytes.clear()   # the bootstrap was paid (and logged) by the parent run
    return row['weights_sha']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--seed', type=int, choices=(42, 43), required=True)
    p.add_argument('--rounds', type=int, default=50); p.add_argument('--stop_after', type=int, default=6)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-5); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--aux_batch', type=int, default=256); p.add_argument('--aux_weight', type=float, default=1.)
    p.add_argument('--backend', choices=['eager', 'compiled'], default='compiled')
    p.add_argument('--out', default=str(P.ROOT / 'papers/full-drift/runs')); p.add_argument('--smoke', action='store_true')
    p.add_argument('--wandb_project', default='mobihet-fulldrift'); p.add_argument('--wandb_online', action='store_true')
    p.add_argument('--max_hours', type=float, default=7.)
    # speed only (results bit-exact with --workers 1): processes that train disjoint clients, GPUs round-robin by rank
    p.add_argument('--workers', type=int, default=1); p.add_argument('--rank', type=int, default=0); p.add_argument('--address')
    # round 17: continue from a committed epoch of an audited run of the same arm/seed/config (exact resume across sessions)
    p.add_argument('--parent'); p.add_argument('--parent_round', type=int)
    # owner 2026-10-18: tolerance-class speed (fused AdamW, nondeterministic cuDNN) for runs built from round 19 on; such arms are
    # compared only with controls also run with --fast (config key 'fast' -> different fingerprint, refuses a non-fast parent)
    p.add_argument('--fast', action='store_true')
    # owner 19/10 (protocol §3): final 50-epoch run -- train every train80% row, official test20% after every round, headline r50
    p.add_argument('--final', action='store_true')
    p.add_argument('--clients', type=int, choices=(20, 50, 100), default=20)
    p.add_argument('--defer_eval', action='store_true', help='commit training; evaluate every round in separate notebooks')
    p.add_argument('--holdout_only', action='store_true', help='score internal group/old holdouts; never score official test20%')
    p.add_argument('--session_rounds', type=int, help='operational cap; total cosine schedule remains --rounds 50')
    a = p.parse_args()
    assert a.final or a.clients == 20
    assert not a.defer_eval or a.final
    assert not (a.final and a.holdout_only)
    assert a.session_rounds is None or a.session_rounds > 0
    assert not a.final or a.smoke or a.stop_after == 50, 'final runs end at round 50 (sessions chain with --parent)'
    if 'g512' in a.arm.split('_'): a.aux_batch *= 2   # round 18: 512 synthetic rows per batch (smoke: 128)
    if a.smoke: a.stop_after = 2; a.aux_batch = 128 if 'g512' in a.arm.split('_') else 64
    else: assert a.rounds == 50 and a.stop_after in (6, 15, 50)
    assert (a.parent is None) == (a.parent_round is None) and (a.parent is None or 0 < a.parent_round < a.stop_after)
    tokens = a.arm.split('_')
    assert a.workers == 1 or not ({'dyn', 'cons', 'cons0'} & set(tokens) or any(t.startswith('scaf') for t in tokens)), \
        'FedDyn/SCAFFOLD keep cross-client sums inside private_epoch: serial only'
    assert not ('centralized' in tokens and a.workers != 1), 'centralized uses one model/optimizer on one GPU'
    if a.rank: return serve(a)
    out = Path(a.out) / (f'{a.arm}_s{a.seed}' + ('' if a.smoke or a.stop_after == 6 else f'_e{a.stop_after}')); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'config.json').exists(), 'single-session probe: no resume'
    config = {k: v for k, v in vars(a).items() if k not in ('out', 'wandb_online', 'wandb_project', 'max_hours', 'workers', 'rank', 'address',
                                                            'parent', 'parent_round', 'fast', 'final', 'clients', 'defer_eval', 'holdout_only', 'session_rounds')}
    if a.fast: config['fast'] = True   # absent when off: configs (and parent checks) of earlier runs unchanged
    if a.final: config['final'] = True
    if a.clients != 20: config['clients'] = a.clients
    if a.defer_eval: config['defer_eval'] = True
    if a.holdout_only: config['holdout_only'] = True
    config.update(mu=MU, swap_p=SWAP_P, ema_beta=EMA_BETA, cal_rows=CAL_ROWS,
                  kd_cap=KD_CAP, dyn_alpha=DYN_ALPHA, cons_lambda=CONS_LAMBDA, probe_rows=PROBE_ROWS, kd_up=KD_UP, gw2=GW2,
                  p1_lambda=P1_LAMBDA, p1_t=P1_T, p1_classes=list(P1_CLASSES))
    if 'fedtgp' in a.arm: config.update(tgp_dim=TGP_DIM, tgp_lambda=TGP_LAMBDA, tgp_steps=TGP_STEPS,
                                        tgp_lr=TGP_LR, tgp_tau=TGP_TAU)
    if 'canperfl' in a.arm: config.update(personal_epochs=PERSONAL_EPOCHS, personal_max_rows=PERSONAL_MAX_ROWS,
                                          personal_lr=1e-3)
    if {'xo', 'xou'} & set(a.arm.split('_')): config.update(xo_rho=XO_RHO, xo_floor=XO_FLOOR)   # round 24; absent otherwise
    if {'cvf', 'cvl', 'cvp'} & set(a.arm.split('_')): config.update(cv_steps=CV_STEPS, cv_batch=CV_BATCH, cv_lr=CV_LR, cv_jitter=CV_JITTER)   # round 25
    if 'cvp' in a.arm.split('_'): config.update(cv_proj=CV_PROJ)   # round 26 (cvf/cvl configs unchanged)
    if a.parent:   # same scientific config as the parent run (only the stop epoch differs); its committed weights SHA-checked
        cfg, row = parent_marker(a)
        same = lambda c: {k: v for k, v in c.items() if k not in ('stop_after', 'parent')}
        assert same(cfg['config']) == same(config), ('parent config differs', cfg['config'], config)
        config['parent'] = dict(round=a.parent_round, weights_sha=row['weights_sha'], fingerprint=cfg['fingerprint'],
                                fd=cfg['source']['proj/fd.py'], parent=cfg['config'].get('parent'))
    source = {f'mh/{f.name}': digest(f) for f in sorted((P.ROOT / 'mh').glob('*.py'))}
    source.update({f'proj/{f.name}': digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    fx = fixture_path(a)
    manifest = json.loads((fx / 'manifest.json').read_text())
    for name, sha in manifest['files'].items(): assert digest(fx / name) == sha, ('fixture mismatch', name)
    identity = dict(config=config, source=source, fixture=digest(fx / 'manifest.json'))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    save(out / 'config.json', dict(identity, fingerprint=fingerprint))
    wb = None
    if a.wandb_online:
        import wandb
        key = os.environ.get('WANDB_API_KEY'); assert key, 'W&B key required'
        assert wandb.login(verify=True)
        wb = wandb.init(project=a.wandb_project, id=out.name + '-' + fingerprint[:8], resume='allow', name=out.name + ('-fast' if a.fast else ''),
                        config=dict(config, fingerprint=fingerprint), mode='online', dir=str(out),
                        settings=wandb.Settings(save_code=False, disable_code=True))
        assert not wb.settings._offline
        wb.log({'preflight/online': 1}); print('WANDB', wb.url, flush=True)
    deterministic(a.fast)
    t0 = time.time(); pool = Pool(a, out) if a.workers > 1 else None
    try:
        train(a, out, fingerprint, wb, t0, pool)
    finally:
        if pool: pool.close()


def train(a, out, fingerprint, wb, t0, pool):
    R = Run(a); m = Method(R); m.prior()
    save(out / 'runtime.json', dict(torch=torch.__version__, workers=a.workers,
                                    devices=[dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)))
                                             for i in range(torch.cuda.device_count())],
                                    setup_seconds=time.time() - t0, compile_gates=R.compile_gates,
                                    flops_scope='model conv/linear operator estimate; excludes BN, KL, optimizer, calibration and probes'))
    start, previous = 0, None
    if a.parent: start, previous = a.parent_round, load_parent(a, R, m)
    print('setup', a.arm, 'rows', sum(R.n), 'mu', R.mu, 'swap', R.swap, 'kdcap', R.kdcap, 'warm', R.warm,
          'frz', R.frz, 'dyn', R.dyn, 'cons', R.cons, R.lam, 'own', m.own, 'kd2', R.kdup, 'kdpx', R.kdpx, 'fill', R.fill, 'rej', R.rej, 'p1', R.p1, 'gkd', R.gkd, 'bits', m.bits,
          'rules', R.rules, 'la', R.la, 'parent', a.parent_round, 'sec', round(time.time() - t0, 1), flush=True)
    hist = []
    session_end = min(a.stop_after, start + a.session_rounds) if a.session_rounds else a.stop_after
    for r in range(start + 1, session_end + 1):
        if time.time() - t0 + 1.15 * max([v['sec'] for v in hist] or [0]) + 60 > a.max_hours * 3600:
            print('budget stop before epoch', r, flush=True); break
        t1 = time.time(); private_logs, proxy_logs = [], []
        m.start_round()
        if R.cv: R.cv_round(m, r, pool)   # round 25: calibrated snapshot head before the local epochs
        if R.centralized:
            private_logs = [R.centralized_epoch(m, r)]; proxy_logs = [m.logs[0]]
        elif pool: private_logs, proxy_logs = pool.round(m, R, r)
        else:
            for k in range(R.K):
                if m.kd: m.prepare(k, r)
                private_logs.append(R.private_epoch(m, k, r))
                if m.kd:
                    packet, log = m.proxy_epoch(k, r); proxy_logs.append(log)
                    if packet is not None: m.aggregate_sum.add(k, packet)
        aggregation_started = time.perf_counter()
        if not R.centralized: m.aggregate(r)
        aggregation_seconds = time.perf_counter() - aggregation_started
        if R.tgp:
            payloads = [row.pop('tgp_payload') for row in private_logs]
            m.tgp_proto, tgp_loss, tgp_margin = m.tgp_server.update(payloads)
            m.diag.update(tgp_loss=tgp_loss, tgp_margin=tgp_margin)
        m.update_ema()
        algorithm_seconds = time.time() - t1
        diagnostics = dict(swap_fraction=R.swap_stats[0] / max(R.swap_stats[1], 1),
                           **{k: v for k, v in m.diag.items() if isinstance(v, (int, float))})
        kd_keys = [key for key in private_logs[0] if key.startswith(('w_', 'cap_hit', 'kl_', 'probe_', 'scaf_', 'syn_'))] if private_logs else []
        diagnostics.update({'kd_' + key: float(np.mean([v[key] for v in private_logs])) for key in kd_keys})
        if private_logs and 'cv_teach' in private_logs[0]:   # round 25: pooled own-row teacher F1 (trained / calibrated head), teachable share
            for tag in ('orig', 'cal'):
                c = np.sum([v[f'cv_cm_{tag}'] for v in private_logs], 0); tp = np.diag(c)
                diagnostics[f'cv_teacher_f1_{tag}'] = float(np.mean(2 * tp / np.maximum(c.sum(0) + c.sum(1), 1)))
            t = np.sum([v['cv_teach'] for v in private_logs], 0)
            diagnostics.update(cv_teachable_orig=float(t[1] / max(t[0], 1)), cv_teachable_cal=float(t[2] / max(t[0], 1)), cv_server_seconds=R.cv_seconds)
        if proxy_logs:
            diagnostics.update(drift_mean=float(np.mean([v['drift'] for v in proxy_logs])),
                               prox_mean=float(np.mean([v['prox'] for v in proxy_logs])))
        row = dict(algorithm_seconds=algorithm_seconds, round=r, lr=R.lr(r), clients=private_logs, proxy_clients=proxy_logs,
                   version=m.version, fingerprint=fingerprint, prev_sha=previous, diagnostics=diagnostics,
                   loss=float(np.mean([v['loss'] for v in (proxy_logs if m.fedavg else private_logs)])))
        traffic = [v for v in R.bytes if v[0] == r]
        up_bytes = sum(v[4] for v in traffic if v[2] == 'up')
        down_bytes = sum(v[4] for v in traffic if v[2] == 'down')
        steps = sum(v['applied'] + v['skipped'] for v in private_logs)
        row['cost'] = dict(up_bytes=up_bytes, down_bytes=down_bytes, total_bytes=up_bytes + down_bytes,
                           payload_bytes_per_algorithm_second=(up_bytes + down_bytes) / algorithm_seconds,
                           local_steps=steps, applied_steps=sum(v['applied'] for v in private_logs),
                           skipped_steps=sum(v['skipped'] for v in private_logs),
                           gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           server_aggregation_seconds=aggregation_seconds,
                           sum_client_epoch_wall_seconds=sum(v['seconds'] for v in private_logs),
                           **{key: value for key, value in m.diag.items() if key.startswith('codec_')})
        if R.gkd:   # operator-count estimate, measured by scripts/count_dagsnet_flops.py; excludes KL/BN/optimizer/probes/codec
            real, synthetic = sum(R.n), a.aux_batch * steps
            extra = synthetic if R.xo and r > 1 else 0   # round 24: crossover rows (private forward/backward + snapshot forward)
            row['cost']['model_flops_estimate'] = (2 * real + synthetic + extra) * 13280256 + ((real + synthetic + extra) * 4443648 if r > 1 else 0)
            row['cost']['gaussian_matvec_flops_estimate'] = synthetic * 2 * P.D * P.D
            if R.cv and r > 1: row['cost']['cv_forward_rows'] = 3 * sum(len(c) for c in R.cal)   # round 25: moments + 2 diagnostic passes
        last = r == a.stop_after
        holds = ('test',) if a.final else ('group', 'old') if getattr(a, 'holdout_only', False) else \
                ('group', 'old', 'test') if last else ('group', 'old')   # --final: test20% every round
        row['eval'] = {} if a.defer_eval else R.evaluate(m, out, r, holds, scores=pool.scores(R, holds) if pool and not m.fedavg else None)
        if a.defer_eval: row['evaluation_pending'] = True
        else: row['predictions_sha'] = digest(out / 'predictions' / f'round_{r:03d}.npz')
        weights = dict(private=[] if m.fedavg else [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in R.priv],
                       proxy=[{k: v.detach().cpu() for k, v in s.items()} for s in
                              [p.state_dict() for p in (m.proxy if m.own else m.proxy[:1])]] if m.kd else [],
                       ema_private=[{k: v.detach().cpu() for k, v in s.items()} for s in (m.ema_private or [])],
                       architecture=R.architecture, round=r, fingerprint=fingerprint)
        if R.tgp:
            weights['tgp_proto'] = m.tgp_proto.detach().cpu()
            weights['tgp_server'] = {k: v.detach().cpu() for k, v in m.tgp_server.state_dict().items()}
        wp = out / 'weights' / f'round_{r:03d}.pt'; save(wp, weights); row['weights_sha'] = digest(wp)
        if m.bits:   # codec state beyond the weights (server exact model, client uplink residuals): resume bundle before the marker
            rp = out / 'resume' / f'round_{r:03d}.pt'
            save(rp, dict(round=r, fingerprint=fingerprint, exact={n: v.detach().cpu() for n, v in m.exact.items()},
                          ef={f'{k}/{n}': v.detach().cpu() for (k, n), v in m.ef_state.items()}))
            row['resume_sha'] = digest(rp)
        row['sec'] = time.time() - t1; previous = row['weights_sha']
        save(out / 'complete' / f'round_{r:03d}.json', row); hist.append(row)   # atomic commit LAST
        role, rule = ('prototype', 'nearest') if R.tgp else ('proxy', 'head') if m.kd and not m.own else ('private', 'gp')
        shown = 'test' if a.final else 'group'
        display = 'pending' if a.defer_eval else f'{row["eval"][role][shown][rule]["f1_macro"]:.4f}'
        print(f'r{r} {a.arm} s{a.seed} {role} {shown} {rule} {display} '
              f'loss {row["loss"]:.4f} {row["sec"]:.0f}s', flush=True)
        if wb:
            metrics = dict(round=r, loss=row['loss'], lr=row['lr'], seconds=row['sec'], algorithm_seconds=algorithm_seconds,
                           **{'algorithm/' + k: v for k, v in diagnostics.items()})
            for model_role, holds in row['eval'].items():
                for hold, rules in holds.items():
                    for rule_name, ev in rules.items():
                        metrics.update({f'{model_role}/{hold}/{rule_name}/{key}': v for key, v in ev['mean_metrics'].items()})
            wb.log(metrics)
    personalized = R.personalize_canperfl(m, out, ('group', 'old')) if R.canperfl and hist and hist[-1]['round'] == a.stop_after else None
    result = dict(config=vars(a), fingerprint=fingerprint, history=hist, bytes=R.bytes, train_rows=R.n,
                  complete=len(hist) == a.stop_after - start, session_complete=len(hist) == session_end - start,
                  evaluation_pending=a.defer_eval, personalized=personalized, minutes=(time.time() - t0) / 60)
    save(out / 'result.json', result)
    with (out / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 'role', 'holdout', 'rule', *METRIC_KEYS]); w.writeheader()
        for row in hist:
            for role, holds in row['eval'].items():
                for hold, rules in holds.items():
                    for rule, ev in rules.items():
                        w.writerow(dict(round=row['round'], role=role, holdout=hold, rule=rule, **ev['mean_metrics']))
    if wb:
        wb.summary['complete'] = result['complete']; wb.summary['rounds_completed'] = len(hist)
        if personalized:
            for hold, rules in personalized['eval'].items():
                for rule, ev in rules.items():
                    for key, value in ev['mean_metrics'].items(): wb.summary[f'personalized/{hold}/{rule}/{key}'] = value
        wb.finish()
    print('done', a.arm, 'complete', result['complete'], flush=True)


if __name__ == '__main__':
    main()

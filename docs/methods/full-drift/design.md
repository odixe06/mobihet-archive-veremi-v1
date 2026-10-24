# FullDrift — kiểm drift và kỹ thuật FL trên full data (holdout nội bộ)

Owner duyệt 15/10 tối: kiểm kỹ thuật mới **thẳng trên full** vì pilot không tái hiện động học full;
**một seed chung (42)** để so sánh phương pháp, thử seed khác để sau. Cổng dưới đây khóa **trước** khi push.
Owner 17/10 ~10:10Z: mọi arm một lần gộp đồng bộ sau mỗi epoch local (như FL_NIDS_10client).

## Vấn đề đã đo

- Full mỗi round = 1 epoch ≈4.200 bước/client. DML proxy bão hòa ~0,78–0,79 từ r3–r5 trong khi LR còn ~1e-3;
  Gaussian/local đạt đỉnh r1 rồi giảm ([chẩn đoán](../../archive/research/plateau-diagnosis-2026-10-15.md)).
- Trần centralized DAGSNet 0,8532; proxy DML r23 0,787 (test, mô tả).
- Pilot: FedAvg > DML proxy +0,027, nhưng full chưa từng đo FedAvg thuần.
- AntiMem pilot (W&B, chưa audit): chống học thuộc giúp Gaussian private (EMA, swappos) nhưng không giúp proxy.

## Dữ liệu và đánh giá

`prepare(holdout=True, train_cap=None, eval_cap=65536)` trên full 43.045.415 dòng: loại 2% cuối theo thời gian
(mỗi lớp×scenario) và receiver-group hash mod20==0 (~5%) khỏi train; `group` (receiver chưa thấy) và `old`
(temporal) mỗi tập 65.536 dòng. Official test20% (10.761.343) chỉ chạy **một lần sau epoch 6**, chỉ báo cáo.
Epoch 1–6 của lịch cosine T50 gốc (LR giống headline). Mean20client, đủ 10 metrics + per-class.

## Arm và cơ chế (seed 42)

| arm | thay đổi so với recipe full | vì sao có thể giúp | dấu hiệu sai |
|---|---|---|---|
| `fd_dml` | recipe full DML (bit-exact epoch `full_run`) | đối chứng | — |
| `fd_dml_prox` | FedProx μ=0,01 trên proxy, neo model chung đầu round | kéo proxy về model chung, giảm drift, không thêm byte | drift không giảm, hoặc F1 < +0,010 |
| `fd_dml_avgm` | FedAvgM: server velocity β=0,9 trên delta trung bình (BN trung bình thường) | giữ hướng chung qua round, tắt nhiễu client; không thêm byte | F1 < +0,010 hoặc mất ổn định |
| `fd_fedavg` | một model chung, chỉ CE, không private/KD | kiểm xem KD với private học thuộc có kéo proxy xuống | (so mô tả với `fd_dml`) |
| `fd_gaussian` | recipe full Gaussian | đối chứng Gaussian | — |
| `fd_gaussian_swappos` | swap 50% khối 9 feature receiver (dòng khác receiver cùng client) + đặt0 4 vị trí | cắt shortcut receiver; pilot +0,016–0,018 | private gp group < +0,010 |
| role `private_ema` | EMA trọng số private β=0,5^0,5/epoch, BN tính lại trên 65.536 dòng local cố định | trung bình quỹ đạo, giảm học thuộc; pilot Gaussian +0,036–0,039 | < +0,010 so private |

Mọi arm ghi `drift_mean` (‖θ_k−θ_chung‖/‖θ_chung‖ proxy sau train local), `prox_mean`, `server_velocity_norm`,
`swap_fraction`. Byte: prox/avgm/FedAvg không đổi byte.

## Cổng khóa (group holdout, epoch 6, seed 42 — sàng lọc)

Primary: DML/FedAvg = proxy head; Gaussian = private gp; EMA = private_ema gp so private gp cùng arm.
- Kỹ thuật **có triển vọng** nếu technique − đối chứng ≥ +0,010 và benign ≥ −0,005 ở epoch 6.
- `fd_fedavg` vs `fd_dml` proxy: so sánh mô tả, quyết định headline FL cho bước sau (owner chốt).
- Một seed là sàng lọc, không là kết luận cuối; kỹ thuật có triển vọng mới chạy seed khác/r50.
- Không chọn epoch/HP theo holdout hay test; không quét μ/β sau khi thấy kết quả. Old báo kèm.

## Giới hạn

Epoch 6/50 không chứng minh r50.
FedProx/FedAvgM dẫn xuất cho SGD; ở đây optimizer AdamW của recipe (chỉ thêm số hạng/velocity).
Không resume qua session: probe một session/arm.

## Vòng 2 — owner duyệt 15/10 ~22:00Z (cổng khóa trước push; PUSHED 16/10 02:19Z)

Kết quả vòng 1: [results](results.md). Seed 42; đối chứng 6 epoch lấy từ vòng 1 (cùng fixture/seed/code path, chạy tất định).

| arm | thay đổi | cơ chế/kỳ vọng | cổng |
|---|---|---|---|
| `fd_dml`, `fd_fedavg` @ep15 | cùng arm, 15 epoch (thư mục `_e15`) | đối chứng horizon cho các vòng sau | ep1–6 tái lập vòng 1 |

Old/test báo kèm, test chỉ epoch cuối.

### Cổng phụ private gp — owner 16/10 ~03:00Z (khóa ~04:00Z, trước mọi kết quả ep6 vòng 2)

Owner đổi mục tiêu sang **dự đoán của client**: primary = private gp (A, giữ R2), báo kèm góc nhìn theo prior nhãn
từng client và client tệ nhất ([hướng client](../../archive/research/client-model-directions-2026-10-16.md)).
Arm DML dùng **cùng rule** với cổng proxy, đo trên private gp group. FedAvg không có private (R2) ⇒ chỉ mô tả.
Vòng 1 private gp (đã thấy trước khi khóa, chỉ mô tả): prox −0,013, avgm +0,002.
Mã: `PRIVATE_GATES` trong [report_fd.py](../../../methods/full-drift/builds/fd-probe/scripts/report_fd.py).

## Vòng 3 — mục tiêu client (owner 16/10; cổng khóa ~04:05Z trước push)

Owner 16/10: primary = **private gp group ep6** (A), giữ **R1** (trao đổi ngắn, byte hạn chế) và **R2**
(private có thể khác kiến trúc; hiện DAGSNet để so). Mọi arm vòng 3 **không thêm byte** (1 lần gộp/epoch như
`fd_dml`) và chỉ đổi phía client, nên dùng được với private bất kỳ kiến trúc.
Đối chứng `fd_dml` vòng 1 (tất định; vòng 2 kiểm lại ep1–6 qua `fd_dml@e15`). Seed 42.

| arm | thay đổi | bằng chứng → cơ chế | cổng private gp group ep6 |
|---|---|---|---|
| `fd_dml_m4b` | đặt 0 khối 9 feature receiver + 4 vị trí (train và eval, cả hai model; như AntiMem) | pilot DML private gp +0,021/+0,028 hai seed → bỏ shortcut receiver mà private học thuộc | Δ ≥ +0,010, benign ≥ −0,005 |
| `fd_dml_swappos` | swap 50% khối receiver giữa receiver khác cùng client + đặt 0 4 vị trí | pilot +0,019/+0,025; Gaussian full +0,052 → phi tương quan receiver–nhãn | như trên |
| `fd_dml_ntd` | KL proxy→private chỉ trên lớp không đúng (FedNTD, Lee et al. NeurIPS 2022), trọng số giữ nguyên | head→gp +0,12 cho thấy private lệch prior local mạnh → bỏ xung đột giữa CE và lớp đúng của thầy, giữ tri thức lớp khác | như trên |
| `fd_dml_q4` | QProxy q4 (up int4 stochastic, server exact/EF, down int4) | pilot q4 ±0,001 byte 7,65× nhỏ → R1: payload nhỏ hơn mà client không mất | private gp và proxy head Δ ≥ −0,010, benign ≥ −0,010 |

Proxy head của m4b/swappos/ntd báo mô tả. Kèm cột local (prior nhãn từng client) và client tệ nhất, mô tả.
Không quét tỉ lệ swap/feature/nhiệt độ NTD sau khi thấy kết quả. Session `nguyenminhtran060113` g: m4b ‖ swappos,
h: ntd ‖ q4, 4,5h mỗi session (arm ≈3,62h + setup 0,1h).

## Vòng 4 — client, quota ngắn (cổng khóa 16/10 ~06:30Z trước push)

Owner 16/10: dùng account còn quota ngắn cho pilot; giữ R1/R2. `odixe0502` (4,0h, reset 17/10 00:00Z), session 3,95h:
GPU0 `fd_dml_hkd`, GPU1 `fd_gaussian_m4b` (arm ≈3,62h/2,5h + setup 0,1h). Đối chứng vòng 1 cùng fixture/seed.

| arm | thay đổi | bằng chứng → cơ chế | cổng private gp group ep6 |
|---|---|---|---|
| `fd_dml_hkd` | private khớp tầng ẩn 256-d của proxy qua projector tuyến tính **local** (khởi tạo đơn vị, không upload); MSE chuẩn hóa theo ‖h_proxy‖², cùng trọng số thích nghi với KL; proxy không đổi (FedKD hidden loss, Wu et al. Nat. Commun. 2022) | ProxyFuse: private đúng mà proxy sai chỉ ≈2% ⇒ private là bản chép kém của proxy; logit KL chỉ chở 16 số/dòng → thêm kênh đặc trưng | vs `fd_dml`: Δ ≥ +0,010, benign ≥ −0,005 |
| `fd_gaussian_m4b` | Gaussian + đặt 0 13 feature receiver/vị trí (train, draw, eval) | AntiMem pilot Gaussian m4b +0,025 hai seed (lớn nhất); full swappos +0,052 → bỏ shortcut receiver | vs `fd_gaussian`: Δ ≥ +0,010, benign ≥ −0,005 |

0 byte thêm (R1); projector/feature mask chỉ phía client (R2). Không quét trọng số/scale sau kết quả.

## Vòng 5 — m4b: horizon và ghép (owner 16/10 ~08:20Z; cổng khóa ~08:25Z trước push)

Vòng 3: m4b private gp +0,036, proxy +0,052 (ep6). Owner: kiểm horizon + ghép trước; headline full chỉ đổi khi
horizon + seed 43 xác nhận. Session `nguyenminhtran060113` j (8,95h): GPU0 `fd_dml_m4b:15`; GPU1 `fd_dml_m4b_q4`.

| arm | so với | câu hỏi | cổng (group) |
|---|---|---|---|
| `fd_dml_m4b@e15` | `fd_dml@e15` (vòng 2) | lợi m4b còn ở ep15? | private gp ep15 Δ ≥ +0,010 (proxy báo kèm, cùng rule); ep1–6 phải tái lập vòng 3 |
| `fd_dml_m4b_q4` | `fd_dml_m4b` | R1: nén 7,65× có giữ lợi? | private gp và proxy Δ ≥ −0,010, benign ≥ −0,010 |

## Vòng 6 — seed 43 xác nhận m4b (owner 16/10 ~16:45Z; cổng khóa ~16:50Z trước push)

Owner: headline full chỉ đổi sang m4b khi horizon (vòng 5) + seed 43 xác nhận; chọn 2 session trên `anhtruong1909`
(9,36h live, reset 17/10 00:00Z), mỗi session 4,2h (DML ep6 đo 205–217 phút + setup). Cùng source/fixture vòng 5
(fd.py SHA không đổi), chỉ `--seed 43`; smoke local vốn chạy seed 43. Đối chứng cùng seed trong vòng (không tái dùng s42).
Session k: GPU0 `fd_dml`, GPU1 `fd_dml_m4b`.
Lúc khóa đã thấy W&B s42 (chưa audit) `m4b_q4` ep6 private gp 0,701/proxy 0,816, `m4b@e15` ep14 0,728/0,849.

| so sánh (seed 43, group ep6) | câu hỏi | cổng private gp (primary) | proxy head (cùng rule, báo kèm) |
|---|---|---|---|
| `fd_dml_m4b` vs `fd_dml` | lợi m4b lặp lại ở seed khác? | Δ ≥ +0,010, benign ≥ −0,005 | như trên |

Quy tắc headline (owner): đổi khi `fd_dml_m4b@e15` PASS (vòng 5) **và** `fd_dml_m4b` vs `fd_dml` seed 43 PASS cả private
gp lẫn proxy. Một seed thêm vẫn là sàng lọc, không paired CI.

## Vòng 7 — seed 43 swappos + Gaussian m4b (owner 16/10 ~17:25Z; cổng khóa ~17:29Z trước push)

Owner: dùng nốt `annhannh331` (5,29h, reset 00:00Z) lặp kỹ thuật đã PASS ở seed 43. Session m (4,2h): GPU0 `fd_dml_swappos`,
GPU1 `fd_gaussian_m4b`, `--seed 43`, cùng source vòng 5/6. `fd_gaussian` s43 không vừa quota (cần 4,85h trên một GPU).

| so sánh (seed 43, group ep6) | cổng |
|---|---|
| `fd_dml_swappos` vs `fd_dml` (s43, vòng 6) | private gp Δ ≥ +0,010, benign ≥ −0,005; proxy head cùng rule báo kèm |
| `fd_gaussian_m4b` s43 | **mô tả**: so `fd_gaussian` s42 (0,5882) và `fd_gaussian_m4b` s42 (0,6553); không có đối chứng cùng seed |

## Vòng 8 — cập nhật client/trao đổi tri thức (owner 17/10 ~20:55Z; cổng khóa ~21:00Z trước push)

Nguồn: [algorithm-improvement-plan](../../archive/research/algorithm-improvement-plan-2026-10-17.md) §4.1–4.5, §5.1, §7.1. Owner chốt:
chạy mọi nhánh, dùng mọi slot quota; frozen teacher theo plan (tắt KD private trước snapshot gộp đầu tiên, thêm control
warm-up); cổng §7.1, **dùng lại control `fd_dml_m4b` s42 ep6 đã audit** (vòng 3; bit-exact source mới kiểm local 2 epoch).
Source mới tag `20261015-r8`; mọi arm cũ tính toán y hệt vòng 5 (đã kiểm).

| arm (s42, ep6, m4b) | cơ chế (một yếu tố) |
|---|---|
| `fd_dml_m4b_kdcap` | A: hệ số KD `w=min(1/(CE_priv+CE_proxy), 1)` cho cả hai chiều KL |
| `fd_dml_m4b_kdcap_wu` | A + KD private tắt khi chưa có snapshot gộp (ep1) — control warm-up |
| `fd_dml_m4b_kdcap_frz` | A + B: private học teacher = proxy gộp đầu round (eval, đóng băng); ep1 như `wu` |
| `fd_dml_m4b_own` | DML-m4b, proxy không gộp (no-peer, cùng 2 model/compute) |
| `fd_gaussian_m4b_own` | Gaussian-m4b moment riêng từng client |
| `fd_dml_m4b_dyn` | FedDyn α=0,01 trên tham số proxy (a_i local, h server từ tổng cohort) |
| `fd_dml_swappos_cons` / `_cons0` | hai view (giữ / tráo khối 9 feature receiver), 4 vị trí = 0; JS λ=1 / λ=0 cùng 2 forward |
| `fd_gaussian`, `fd_gaussian_swappos` **s43** | đối chứng cùng seed cho `fd_gaussian_m4b` s43 (vòng 7) |

Cổng gain (private gp, group, ep6): **Δ ≥ +0,010, benign Δ ≥ −0,005, client tệ nhất (global macro-F1 private gp)
không giảm quá 0,010**. Proxy head báo kèm.

| so sánh | cổng |
|---|---|
| `kdcap` vs `fd_dml_m4b` | gain (A) |
| `kdcap_frz` vs `kdcap_wu` | gain (B cô lập nguồn teacher); vs `kdcap`, `fd_dml_m4b` báo kèm |
| `kdcap_wu` vs `kdcap` | mô tả (tác động warm-up) |
| `fd_dml_m4b` vs `fd_dml_m4b_own` | gain (lợi FL của DML-m4b) |
| `fd_gaussian_m4b` vs `fd_gaussian_m4b_own` | gain (lợi FL của Gaussian-m4b) |
| `dyn` vs `fd_dml_m4b` | gain |
| `cons` vs `cons0` | gain (cô lập JS); `cons` vs `fd_dml_m4b` mô tả (thua m4b ⇒ không ưu tiên) |
| `fd_gaussian_m4b`@s43 vs `fd_gaussian`@s43; `fd_gaussian_swappos`@s43 vs `fd_gaussian`@s43 | gain (seed 43) |

Chẩn đoán khóa trước: `kdcap` có `cap_hit` (tỷ lệ bước w_raw>1) median client < 0,5 ⇒ giả thuyết trọng số KD yếu.
Log mỗi client/epoch: median/p95/max w_raw, cap_hit, KL hai chiều, probe cố định 512 dòng train (‖∇CE‖, ‖∇KL‖, tỷ số).
FedDyn log ‖h‖, mean ‖a_i‖. Đã biết từ vòng 4 (không cần chạy): Gaussian-m4b + EMA 0,6452 < non-EMA 0,6553 (−0,010)
⇒ nhánh EMA §4.3 trượt cổng dừng (+0,010) ở s42.
Giới hạn: FedDyn với AdamW/DML là variant dự án, round 1 server ngoại suy w0+2·Δ̄; `own` chỉ chấm private.

## Vòng 9 — SCAFFOLD-proxy, Hybrid private-only, horizon seed 43 (owner 17/10 ~20:55Z "chạy mọi nhánh"; cổng khóa ~21:10Z trước push)

Source tag `20261015-r9` (arm vòng 5/8 bit-exact, kiểm local 2 epoch). Account nhỏ chỉ push khi
(giờ tới reset × số session) < GPU left, để quota cũ không dừng giữa chừng (owner chọn).

| arm | cơ chế |
|---|---|
| `fd_dml_m4b_sgd{1,3,10}` | proxy chuyển AdamW → SGD thuần lr 0,01/0,03/0,1 × cosine; private giữ AdamW (control c=0) |
| `fd_dml_m4b_scaf{1,3,10}` | + SCAFFOLD Option II trên proxy: g+c−c_i sau clip, c_i⁺=c_i−c+(w−y_i)/(Kη), K = bước áp thật; c←c+mean Δc_i |
| `fd_dml_m4b_gpool` / `_gown` | private thêm CE trên 256 mẫu Gaussian full-cov (moment gộp / riêng), proxy chỉ real, KD chỉ real |
| `fd_dml_m4b_replay` | private thêm CE trên 256 dòng local cân bằng lớp (lớp client có), cùng số forward |
| `fd_dml_m4b` (rerun, source r9) | log trọng số/gradient KD của baseline chưa cap (plan §3.2); nhân bản cross-source của control vòng 3 |
| `fd_dml_m4b:15`, `fd_dml:15` **s43** | horizon ep15 seed 43 cho headline m4b |

Quy tắc LR (khóa trước): chọn LR = argmax **old-holdout proxy head ep6** của ba control `sgd*`; không nhìn group/test.

| so sánh (group, ep6 trừ khi ghi) | cổng |
|---|---|
| `scaf{LR*}` vs `sgd{LR*}` | gain (cô lập correction); hai LR còn lại mô tả |
| `scaf{LR*}` vs `fd_dml_m4b` | gain (giá trị thực dụng so baseline AdamW) |
| `sgd{LR*}` vs `fd_dml_m4b` | mô tả (tác động đổi optimizer) |
| `gpool` vs `fd_dml_m4b`; `gpool` vs `gown`; `gpool` vs `replay` | gain cả ba (Hybrid phải hơn nền, hơn moment riêng, hơn replay local) |
| `fd_dml_m4b` rerun vs vòng 3 | mô tả: bit-exact mong đợi (cùng tính toán) |
| `fd_dml_m4b@e15@s43` vs `fd_dml@e15@s43` | horizon (private gp Δ ≥ +0,010) |

Gain = private gp Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất không giảm quá 0,010 (như vòng 8). Bytes thêm của SCAFFOLD
(c xuống, Δc_i lên, FP32 ≈ 1,58 MB/chiều) ghi trong `bytes`; giai đoạn A không dùng byte để loại arm.

## Vòng 10 — dùng nốt quota trước reset (owner 16/10 ~22:10Z; cổng khóa ~22:20Z trước push)

Owner: push mọi account còn vừa trước 00:00Z, theo dõi và dừng nếu bất ổn. Source = bytes vòng 9 (tag `20261015-r10`).
`y` (anhtruong1909, s42) thay `x` của trietbackup (không push). Seed 43 lặp các arm vòng 8–9, so đối chứng s43 đã audit
(`fd_dml_m4b`@s43, `fd_gaussian_m4b`@s43 vòng 6–7) hoặc control cùng vòng: cùng cổng gain+worst như vòng 8.

| arm s43 | so với |
|---|---|
| `fd_dml_m4b_own`, `fd_gaussian_m4b_own` | `fd_dml_m4b`@s43 / `fd_gaussian_m4b`@s43 phải hơn own (lợi FL) |
| `fd_dml_m4b_kdcap`; `kdcap_frz` vs `kdcap_wu` | `fd_dml_m4b`@s43; cô lập teacher |
| `fd_dml_m4b_q4` | `fd_dml_m4b`@s43 noninferior (Δ ≥ −0,010, benign ≥ −0,010) |
| `fd_dml_m4b_gpool` vs `fd_dml_m4b`@s43, vs `gown`, vs `replay`; `fd_dml_m4b_dyn` | như vòng 8–9 |

Quan sát trước khóa (W&B r2, không đổi cổng): `kdcap_frz` s42 private head 0,584 (> wu 0,549) nhưng **gp 0,244**:
teacher gộp mang prior toàn cục, luật gp (đã khóa) cộng thêm `log π_g − log π_k` ⇒ sửa quá tay. Báo head kèm, không đổi rule.

## Vòng 11 — cải tiến headline m4b + gpool (owner 17/10 ~05:10Z; cổng khóa ~06:00Z trước push)

Owner: headline client = DML-m4b + `gpool`; trước khi chạy đủ 50 round, thử cải tiến kỹ thuật tốt nhất (ghép/bổ sung);
phương pháp đã FAIL chỉ chạy lại khi có chỉnh sửa. Quota tuần mới: query mọi account. Seed 42 sàng lọc; arm PASS lặp
seed 43 vòng sau. Source tag `20261015-r11`; kernel la (kd2 ‖ kdpx), lc (gpool:15 ‖ gw2).

**Cải tiến** (đối chứng `fd_dml_m4b_gpool` s42 đã audit, private gp 0,7174; cổng gain+worst như vòng 8):

| arm | bằng chứng → cơ chế → dấu hiệu sai |
|---|---|
| `gpool_kd2` | Proxy head > private gp ở 15/16 lớp (benign 0,759 vs 0,502; macro 0,79 vs 0,70) và giảm KD (`kdcap`) làm private −0,028/−0,025 ⇒ private chưa hấp thụ hết teacher. KD phía private ×2 (proxy giữ nguyên). Sai nếu Δ < +0,010 hoặc client tệ nhất < −0,010. |
| `gpool_kdpx` | `kdcap` cho proxy +0,024/+0,033 nhưng hại private vì cap cả chiều private←proxy. Chỉ cap trọng số KL proxy←private ở 1 ⇒ proxy bớt bị kéo về dữ liệu local, private giữ KD đầy đủ, teacher tốt hơn. Sửa đổi của `kdcap` (FAIL). Sai như trên; proxy head báo kèm. |
| `gpool_gw2` | gpool hơn gown/replay +0,03 (tri thức gộp có ích), liều chưa biết. Trọng số CE Gaussian 1 → 2. Sai nếu Δ < +0,010. |
| `gpool:15` | Gaussian riêng đạt đỉnh sớm rồi giảm theo epoch; trước 50 round phải biết lợi gpool còn ở ep15. Horizon vs `fd_dml_m4b@e15` (s42 audited): private gp Δ ≥ +0,010 ở ep15. |

## Vòng 12 — headline qua kênh int4 thật (owner 17/10 ~14:05Z; cổng khóa ~14:15Z trước push)

Owner 17/10 ~13–14Z: ưu tiên communication budget hạn chế + metrics của private (computational cost không phải mục tiêu;
bỏ hướng giảm độ phức tạp model). Owner chọn chỉ `gpool + q4`. Source tag `20261015-r12`: so r11 thêm vòng lặp không đồng
bộ host, codec `QPacket` (byte = `len(packet)`), `--workers` — đều bit-exact (test `--part r12`, `par`, `checks`).
Session `odixe0502/mobihet-fd-s42qa-20261015-r12-v1`: một arm trên cả hai GPU (`--workers 2`).

| arm | bằng chứng → cơ chế → dấu hiệu sai |
|---|---|
| `fd_dml_m4b_gpool_q4` | q4 noninferior trên m4b hai seed (Δ +0,000 / −0,003), sai số downlink 2–5 % chuẩn model ⇒ proxy dense thừa độ chính xác; gpool chỉ thêm CE private nên kênh proxy giống m4b. Proxy lên/xuống bằng QPacket int4 (208.268 B/chiều, 7,65× nhỏ hơn fp32). Sai nếu private gp Δ < −0,010 hoặc benign < −0,010 so `fd_dml_m4b_gpool` s42 (client tệ nhất báo kèm). |

Đối chứng: `fd_dml_m4b_gpool` s42 vòng 9 đã audit (private gp 0,7174). Round-1 private của `gpool_q4` = `gpool` (q4 chỉ tác
động sau train local) — kiểm trong test `r12`.

## Vòng 13 — A/B/C + đề xuất của owner (owner 17/10 ~16Z; cổng khóa ~16:40Z trước push)

Owner: chạy thử A, B, C và các đề xuất trong file ưu tiên 17/10 (đã tách thành từng file phương pháp, xem
[research/methods](../../research/methods/README.md)). Owner chọn: arm cần train chạy **từ đầu ep1→6** (không fork checkpoint);
P4 chỉ kiểm **teacher-only**; q2 từ đầu. Source tag `20261015-r13`: arm cũ bit-exact so r12 (test `--part r13`). Seed 42,
group holdout, ep6, `--workers 2`, mỗi session một arm. Đối chứng chung: `fd_dml_m4b_gpool` s42 vòng 9 (private gp 0,7174).

| arm | bằng chứng → cơ chế → dấu hiệu sai |
|---|---|
| `gpool_fill` (A) | 56 % khoảng private–proxy ở lớp < 1 % dữ liệu local (recall private 0,57–0,60 vs proxy 0,73–0,76). Cùng 256 mẫu Gaussian/bước, nhãn chia theo water-filling: q = max(0, L − π_k), Σq = 1 (lớp lớn nhất local thường 0 mẫu, lớp hiếm 16 → 22–26). Sai nếu Δ < +0,010 hoặc recall nhóm lớp < 1 % không tăng. Bằng chứng ngược: gw2 (×2 liều mọi lớp, W&B chưa audit) −0,005. |
| `gpool_rej` (B1) | Proxy gộp ep6 chỉ nhận 53 % mẫu Gaussian (benign 25 %, positionMirroring 3 %, timeDelay 0 %). Từ round 2: proxy snapshot nhận được đầu epoch (eval, no-grad) chấm mẫu; mẫu có argmax ≠ nhãn bị bỏ; mẫu được nhận của một lớp mang trọng lượng của lớp đó (lớp không còn mẫu nào thì mất phần đó trong bước). Round 1 = gpool. Sai nếu Δ < +0,010. |
| `gpool_p1` (P1) | Như đề xuất P1: thêm λ_s T² /m Σ a_c a_j KL(q_teacher ‖ p_private) trên 256 mẫu Gaussian; teacher = proxy snapshot nhận được; a_j = argmax teacher = nhãn; a_c = {benign, dataReplay}; λ_s 0,25, T 1, chia m = 256; CE cứng giữ nguyên; tắt ở round 1. |
| `gpool_p1self` | Control own: cùng công thức, teacher = private của chính client đầu epoch. |
| `gpool_p1cm` | Control class-mean: cùng gate, target = trung bình xác suất teacher của các mẫu được gate cùng lớp trong batch (không phụ thuộc x). |
| `gpool_q2` (C2) | q4 noninferior hai seed. Codec ternary 2-bit (q ∈ {−1,0,1}, scale fp16 theo hàng, 4 mã/byte, EF phía server như q4), byte = `len(packet)`. Sai nếu private gp Δ < −0,010 hoặc benign < −0,010 so gpool (so kèm `gpool_q4` của qa). |

Cổng: quality arm (fill, rej, p1, p1self, p1cm) = gain + worst như vòng 8 (Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất
≥ −0,010) so gpool. P1 cơ chế (chỉ đọc khi p1 qua cổng): p1 − p1self ≥ +0,005 và p1 − p1cm ≥ +0,005. q2 noninferior.
Luật chính vẫn private gp. Mọi arm paux giờ chấm thêm luật `gpe` (C): bias = log π_g − log π_e,k, π_e,k = (π_k + w·q_k)/(1 + w),
q_k = phân bố nhãn Gaussian của client, w = trọng số CE Gaussian (= 1) — prior mà hai số hạng CE huấn luyện tới, bỏ qua KD
(xấp xỉ, khóa trước, không tham số). Cổng C: gpe vs gp cùng weights, gain + worst; PASS chỉ khi qua trên **cả** gpool s42 và s43.
Chẩn đoán log mỗi client/epoch: tỷ lệ teacher nhận mẫu (tổng, từng lớp), tỷ lệ gate P1, KL trung bình trên mẫu được gate.

**Kernel probe (không train DAGSNet), khóa trước:**
- C: chấm lại weights ep6 đã audit (gpool s42/s43, kd2, kdpx s42) bằng head/gp/gpe trên group + old; gp phải tái hiện số audit.
- P3: `calibrate()` (BN tính lại trên 65.536 dòng thật của chính client, che m4b) cho cùng weights. Mở arm train P3 nếu gp
  Δ ≥ +0,005 trên cả hai seed gpool; không thì backlog.
- P1 coverage: proxy gpool s42 ep6 trên 16 × 4.000 mẫu từ bảng moment gộp — tỷ lệ nhận từng lớp, độ phủ gate {benign, dataReplay}.
- P4 teacher-only: RFF `sqrt(2/M) cos(Ωx₅₃ + b)` + intercept, σ = trung vị khoảng cách cặp trên 4.096 dòng train gộp; G, H trên
  mọi dòng train; λ ∈ {1e-4, 1e-3, 1e-2}·tr(G)/M chọn trên **old**; chấm group (argmax). Pooled M128/M256 (M1024 mô tả),
  own (mỗi client, cùng hệ số λ), linear ridge. Mở train private P4 chỉ khi pooled M256 ≥ 0,7174 **và** pooled − own ≥ +0,010
  **và** pooled M256 − linear ≥ +0,010. Tiền lệ âm: Build13 S13 (ridge/RFF tại RSU ≈ làm mượt theo lớp).
- P2 chẩn đoán mode: moment theo (lớp, urban/highway) cho {benign, dataReplay, positionMirroring, timeDelay} (đọc scenario từ
  parquet, gồm cả dòng holdout — chỉ chẩn đoán); proxy nhận mẫu hỗn hợp 2 mode vs 1 Gaussian. Mở arm P2a nếu hỗn hợp nhận
  nhiều hơn ≥ +0,05 tuyệt đối ở ≥ 2/4 lớp.
- C1 (moment53): không train — với m4b, mẫu 66 chiều đặt 0 có cùng phân phối N(μ_S, Σ_SS) với mẫu từ moment 53 chiều; chỉ kiểm
  covariance + byte bằng unit test.
P5/P6 chờ cổng teacher (tiền lệ âm: BoundaryCells, V9, Build7).

## Vòng 14 — kiểm lặp arm PASS vòng 13 (khóa ~19:40Z 17/10 trước push)

Theo quy ước "arm PASS lặp seed 43 vòng sau" và bài học lc (cổng ep6 đo tốc độ học, gpool horizon FAIL). Source = bytes r13
(tag `20261015-r14`, cùng fd.py). `--workers 2`.

| arm | so với | cổng |
|---|---|---|
| `gpool_rej:15` s42 | `fd_dml_m4b@e15` (0,7323) và `gpool@e15` (0,7342), ep15 | horizon: private gp Δ ≥ +0,010 so **m4b@e15** (cần để thay headline); vs gpool@e15 báo kèm |
| `gpool_rej` s43 | `fd_dml_m4b_gpool@s43` (0,7175), ep6 | gain + worst |
| `gpool_q2:15` s42 | `gpool@e15` | noninferior ở ep15 (Δ ≥ −0,010, benign ≥ −0,010); proxy head báo kèm |

rej s42 ep1–6 phải trùng run vòng 13 (cùng source, cùng seed) — kiểm `replay_ep1-6` như gpool:15.

## Probe 14 — vì sao private gp chững ~0,73–0,75 (owner 17/10 ~20Z; khóa ~20:30Z trước push, không train)

Owner: phân tích thật kỹ nguyên nhân private không cải thiện, bằng phép thử thực tế; giữ nguyên dataset đã xử lý (chỉ đọc),
mọi client DAGSNet, không đề xuất đổi mô hình; đề xuất sau phải giữ R1 (byte ít, ngắn hạn) và R2 (hỗ trợ client khác kiến trúc,
không chạy thử). Weights ep15 đã audit: `fd_dml_m4b` s42 / s43, `fd_dml_m4b_gpool` s42 (gp phải tái hiện audit ±0,002).
Mẫu: S = 16.384 dòng train ngẫu nhiên mỗi client (327.680 dòng gộp), Q = 16.384 dòng group. `fd_probe14.py`, `gen_probe14.py`.

| phép đo | giả thuyết | đọc kết quả |
|---|---|---|
| T1 ma trận chéo: private k chấm dòng train của client j; đồng ý/KL với proxy trên dòng mình vs người khác | private khớp dữ liệu mình, hỏng khi sang phân phối client khác | acc/đồng ý own ≫ other ⇒ lỗi do lệch phân phối giữa client (coverage), không do thiếu học |
| T2/T7 group: khoảng proxy − private theo lớp × số mẫu local; dòng private mất (proxy đúng, private sai) đi về lớp nào | khoảng nằm ở lớp hiếm local (vòng 11: 56 % ở lớp < 1 %) | phần lớn khoảng ở bin hiếm ⇒ nút thắt là tri thức lớp vắng mặt; ở lớp đủ mẫu ⇒ ranh giới/receiver |
| T3 đóng băng backbone, fit lại head (Linear h256; MLP c544) trên mẫu gộp mọi client vs chỉ dòng own (cùng số bước) | backbone private đã đủ đặc trưng, head/prior là nút thắt | refit gộp ≈ proxy ⇒ nút thắt ở tầng cuối (kênh tri thức cỡ head đủ); ≈ gốc ⇒ nút thắt ở biểu diễn (cần kênh tác động backbone). Refit own ≈ gốc là kiểm thủ tục |
| T4 bias 16 lớp oracle (fit trên `old`, chấm `group`) | luật quyết định/prior | bias oracle − gp nhỏ (< 0,01) ⇒ loại hướng sửa luật/prior |
| T5 dòng trùng hệt (fp16 sau mask) khác nhãn trên toàn train; kNN own vs gộp cùng cỡ vs gộp ×10 | trần thông tin của 53 feature; giá trị của đa dạng dữ liệu | tỷ lệ dòng không thể đúng theo lớp = trần cứng; kNN gộp − own (cùng số dòng) = lợi của đa dạng client |

Không cổng PASS/FAIL (chẩn đoán). Refit/oracle dùng dữ liệu nhiều client chỉ để đo, không là phương pháp. Kết luận chỉ phạm vi
weights ep15 s42/s43; đề xuất tiếp theo phải dẫn từ bảng này (bằng chứng → cơ chế → dấu hiệu sai).

## Probe 15 — quên do mục tiêu local hay thiếu phủ input (khóa ~20:50Z 17/10 trước push; 1 epoch có kiểm soát)

Từ weights ep15 `fd_dml_m4b` s42 đã audit, mỗi client chạy thêm **một epoch local** (epoch 16, LR lịch T50 khóa = 7,88e-4,
batch 512, AdamW mới, clip 1, fp16) dưới 8 mục tiêu private; đo private gp/head trên `group`, đồng ý với proxy gộp ep15, acc
trên dòng own tại bước 0/64/256/1024/2048/cuối. Đủ 20 client (metric = trung bình client). `fd_probe15.py`, `gen_probe15.py`.
Hàng oracle (dòng thật client khác) và khởi tạo private bằng proxy chỉ để đo, không là phương pháp.

| biến thể | khởi tạo | mục tiêu | trả lời câu hỏi |
|---|---|---|---|
| `cont` | private ep15 | DML đúng như train | đường nền epoch 16 |
| `kdfix` | private ep15 | CE + KD proxy gộp **cố định** trên dòng local | đối chứng của cov_* |
| `cov_real` | private ep15 | kdfix + KD proxy trên 256 dòng thật client khác/bước (không nhãn) | thêm **input** vùng khác có kéo private lên proxy? |
| `cov_gauss` | private ep15 | kdfix + KD proxy trên 256 mẫu Gaussian gộp/bước | input tổng hợp hiện có thay được input thật tới đâu |
| `pinit` | proxy ep15 | DML đúng như train | private bắt đầu ở mức proxy có giữ được không |
| `pinit_ce` | proxy ep15 | chỉ CE local | tốc độ quên thuần do ERM local |
| `pinit_kdfix` | proxy ep15 | CE + KD proxy cố định trên dòng local | KD trên dòng local có giữ tri thức không |
| `pinit_kdonly` | proxy ep15 | chỉ KD proxy cố định trên dòng local | vùng ngoài dữ liệu local có trôi khi không ai neo |

Đọc: (1) `pinit*` rơi từ mức proxy về ~private ⇒ chững do **mục tiêu local kéo đi** (tri thức có sẵn vẫn bị mất), không do
thiếu thông tin; `pinit_kdonly` cũng rơi ⇒ KD chỉ neo trên giá đỡ dữ liệu local. (2) `cov_real − kdfix` ≥ +0,02 ⇒ thiếu
**phủ input** là nút thắt chính; `cov_gauss` đạt bao nhiêu phần khoảng đó đo giá trị của mẫu tổng hợp hiện tại. (3) `cov_real`
≈ `kdfix` ⇒ nút thắt không ở input mà ở dung lượng/mục tiêu. Không cổng PASS/FAIL; mọi đề xuất sau dẫn từ đây và phải giữ R1/R2.

## Probe 16 — nút thắt ở tầng quyết định: dữ liệu gửi được nào fit được nó (khóa ~20:50Z 17/10 trước push, không train)

Probe 14 (AUDITED ~20:40Z): đóng băng backbone private ep15, fit lại **Linear cuối** trên mẫu gộp → 0,843 (MLP head 0,875) so private gp
0,732 và proxy 0,851; mọi client +0,05…+0,19; bias oracle +0,045. ⇒ backbone đủ, head/quyết định là nút thắt. Probe 16 hỏi:

| phần | đo | đọc |
|---|---|---|
| A | refit Linear/MLP gộp vs own trên backbone ep6 của DML-m4b, DML-m4b không gộp (own), Gaussian-m4b (không proxy), Gaussian-m4b own (không FL) | backbone không FL cũng refit lên ~0,84 ⇒ đặc trưng học được tại chỗ, lợi FL cần đến tầng quyết định; thấp hơn rõ ⇒ proxy/FL góp vào biểu diễn |
| B | fit Linear cuối của private ep15 (m4b, gpool) bằng: own thật; + Gaussian gộp (nhãn cứng); + Gaussian proxy nhận; + Gaussian nhãn proxy; + dòng thật client khác nhãn proxy (oracle input); gộp thật (oracle). Bias-only fit trên Gaussian | nguồn gửi được (moment, R1 bootstrap 142,8 kB, R2 vì ở input-space) đạt bao nhiêu phần khoảng own → gộp thật; nhãn proxy trên input thật đạt gần gộp thật ⇒ proxy là người gán nhãn đủ tốt, chỉ thiếu input |
| C | head proxy fit từ moment lớp gộp của đặc trưng 256-d proxy (CCVR; full / diag) | hiệu chỉnh head proxy chỉ bằng thống kê gộp; byte full 2,1 MB/diag 32 kB/client chỉ là dự toán |

Chẩn đoán; không cổng. Đề xuất sau phải dẫn từ đây, giữ R1/R2, không đổi mô hình.

## Probe 17 — mẫu tổng hợp tinh chỉnh bằng proxy cho tầng quyết định (khóa ~21:50Z 17/10 trước push, không train)

Từ probe 16: input thực tế + nhãn proxy → 0,831; Gaussian gộp + nhãn proxy → 0,767. Hỏi: tinh chỉnh mẫu Gaussian ngay trên xe chỉ bằng
thứ đã nhận (moment gộp + proxy gộp, gồm BN running stats đã trung bình) có thu hẹp khoảng không — **0 byte thêm**, R2 (mẫu ở input-space,
mỗi private tự fit head của nó). Tối ưu x (Adam 200 bước, lr 0,05, fp32, cột m4b giữ 0; nhãn lô ∝ π_g, 16 lô × 4.096):
`gauss` (không tinh chỉnh) · `ce` CE(proxy(x), c) + 0,1·Mahalanobis biên 53 chiều · `di_lo`/`di_hi` thêm 0,1/1,0 × khớp mean/var đầu vào
mọi BN với running stats của proxy. Fit Linear cuối (own + mẫu, nhãn proxy argmax hoặc mềm) trên private m4b ep15 s42 / s43 và m4b ep6.
Đo thêm độ thực: tỷ lệ proxy nhận, khoảng cách 1-NN tới dòng thật (so dòng thật giữ lại), nhãn 1-NN khớp. CCVR proxy sửa thứ tự nhãn.

Đọc: biến thể tinh chỉnh ≥ gauss +0,02 hai seed và gần 0,83 ⇒ kênh "mẫu tinh chỉnh + fit tầng quyết định" đáng train; ≈ gauss ⇒ tinh chỉnh
theo proxy không làm mẫu thực hơn, cần kênh input khác. Chẩn đoán; không cổng.

## Vòng 15 — teacher snapshot cố định cho private (khóa ~00:10Z 18/10 trước push)

**Bằng chứng** (probe 14–16, 15): backbone private đủ (refit Linear gộp 0,84); tầng quyết định trôi về dữ liệu local trong mỗi epoch
dài (2–5k bước) vì teacher DML (bản proxy) cũng học local: `cont` gp 0,732 → 0,758 (bước 64) → 0,727 (cuối epoch). **Cơ chế:** KD
private nhận teacher = proxy gộp **đóng băng suốt epoch** (proxy vẫn train DML để gộp như cũ) ⇒ private giữ ranh giới toàn cục; mẫu Gaussian
gộp học KL từ snapshot đó (`gkd`) thay nhãn cứng ⇒ thêm input ngoài giá đỡ local có nhãn mềm nhất quán. **Kỳ vọng** (1 epoch từ ep15):
kdfix head 0,770 (+0,043 so `cont`), cov_gauss head 0,792 (+0,065), client tệ nhất 0,676 / 0,755 vs 0,646. **Dấu hiệu sai:** từ đầu tới
ep15 mà head ≤ control gp + 0,010 ⇒ lợi probe chỉ là trạng thái tạm từ lịch sử DML.

**R1:** `frz` 0 byte thêm (snapshot đã nhận); `gkd` dùng moment gộp đã có (bootstrap C1 142.752 B/client một lần). **R2:** chỉ KD
logit + mẫu ở input-space; private tùy kiến trúc. Không đổi mô hình. Token `frz` không cap (vòng 8 chỉ chạy kèm `kdcap`, gp sập vì cộng
prior hai lần); `warm`: không KD private trước snapshot gộp đầu tiên; `gkd` ep1 = nhãn cứng (chưa có snapshot).

| session | arm (s42, ep1→15 từ đầu, `--workers 2`) | vai trò |
|---|---|---|
| oa | `fd_dml_m4b_frz:15` | M1: teacher cố định, không Gaussian |
| ob | `fd_dml_m4b_gpool_frz:15` | control của gkd (frz + gpool nhãn cứng) |
| oc | `fd_dml_m4b_gpool_frz_gkd:15` | M2: frz + Gaussian KL từ snapshot |

**Cổng (ep15, gain + worst: Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất ≥ −0,010):** arm frz đọc **luật head** (teacher mang prior toàn
cục — LESSONS: luật theo nguồn teacher); control giữ luật gp đã khóa: `m4b_frz` head vs `fd_dml_m4b@e15` gp 0,7323;
`gpool_frz` head vs `gpool@e15` gp 0,7342; `gkd` head vs `gpool_frz` head (cô lập gkd) và vs `gpool@e15` gp. ep6 cùng cặp chỉ mô tả.
Proxy head báo kèm. **Đổi luật triển khai sang head cho arm frz cần owner xác nhận trước khi nhận làm headline.**
Test local: `--part r15` (7 arm vòng 13 bit-exact 2 epoch; gkd ep1 == gpool_frz, ep2 khác, hữu hạn; m4b_frz khác m4b) PASS;
`--part par` gkd (compiled) + m4b_frz (eager): workers 3 == 1 PASS.

## Vòng 16 — lặp seed 43 + gkd qua kênh q2 (khóa ~04:55Z 18/10 trước push)

Vòng 15 PASS cả ba arm ở ep15 s42 (luật head; [kết quả](results.md)). Theo lệ vòng 6–7/14: arm PASS phải lặp seed 43 cùng horizon ep15.
R1: private không phụ thuộc chất lượng proxy (q2:15 noninferior; probe 15) ⇒ thử headline mới qua kênh q2 (219.024 B/round vs q4 416.536 B).
Source tag `20261015-r16` = r15 + một dòng ARMS (`fd_dml_m4b_gpool_q2_frz_gkd`); test `--part r15` PASS (gồm arm mới 2 epoch hữu hạn, khác gkd dense).

| session | arm (ep1→15, `--workers 2`) | cổng (ep15) |
|---|---|---|
| pa | `fd_dml_m4b_gpool_frz_gkd:15` s43 | head vs `gpool@e15@s43` gp (pb) và vs `m4b@e15@s43` gp 0,7263: gain + worst |
| pb | `fd_dml_m4b_gpool:15` s43 | control seed 43 cho pa (gpool s43 mới có ep6) |
| pc | `fd_dml_m4b_gpool_q2_frz_gkd:15` s42 | head vs `gpool_frz_gkd@e15` head 0,8030: noninferior (Δ ≥ −0,010, benign ≥ −0,010) |
| pd | `fd_dml_m4b_frz:15` s43 | head vs `m4b@e15@s43` gp: gain + worst |

Luật head cho arm frz vẫn chờ owner xác nhận (như vòng 15).

## Probe 18 — đo probe 14 trên weights vòng 15 (khóa ~05:05Z 18/10, không train)

Cùng `fd_probe14.py` (T1 chéo client, T3 refit Linear/MLP gộp vs own, T4 bias oracle, T5) trên ep15 `m4b_frz`, `gpool_frz`, `gpool_frz_gkd` s42.
Hỏi: sau frz/gkd, tầng quyết định còn cách refit gộp bao xa (probe 14: m4b 0,732 → 0,843) và backbone có đổi không (refit gộp ≈ 0,84 ⇒ lợi
nằm hết ở tầng quyết định). Đọc `head` của private (teacher cố định); gp chỉ để tái hiện audit. Chẩn đoán; không cổng.

## Probe 19 — bias 16 lớp ước federated (khóa ~05:40Z 18/10, không train)

Probe 18: khoảng còn lại của gkd (0,803 → refit gộp 0,855) phần lớn là bias theo lớp (oracle +0,045); proxy cũng vậy. Đo trên ep15 m4b / frz /
gkd s42: bias proxy fit trên `old` gộp (= tổng thống kê holdout temporal của mọi client qua secure aggregation) theo macro-F1 (oracle),
CE, CE cân bằng lớp (lồi; federated chính xác: mỗi client gửi gradient/Hessian 16 + 136 số mỗi bước); chuyển nguyên bias proxy sang private
(R2: chỉ cộng vào logit); bias private trên `old` (oracle) và trên dòng own (local, 0 byte). Đọc: bias proxy chuyển sang private đạt ≥ +0,02
so head ⇒ đề xuất hiệu chỉnh bias federated (R1 vài kB, R2 OK, không đổi mô hình — là đổi luật quyết định, cần owner duyệt).

## Luật triển khai head vs gp (owner 18/10 ~06Z: luật nào chính xác hơn thì làm luật chính)

So cùng weights đã audit, group holdout, mean 20 client (min = client tệ nhất); local = confusion đánh lại theo prior nhãn
train của từng client ([report §Góc nhìn client](results/report.md)):

| arm (ep cuối) | luật | global F1 (min) | local F1 (min) | local acc (min) |
|---|---|---|---|---|
| m4b@e15 | head / **gp** | 0,609 (0,432) / **0,732** (0,629) | 0,696 / **0,708** | **0,891** / 0,867 |
| gpool@e15 | head / **gp** | 0,614 (0,446) / **0,734** (0,633) | 0,697 / **0,709** | **0,892** / 0,867 |
| m4b_frz@e15 | **head** / gp | **0,769** (0,673) / 0,640 (0,483) | **0,760** / 0,517 | **0,906** / 0,468 |
| gpool_frz@e15 | **head** / gp | **0,772** (0,725) / 0,637 (0,473) | **0,758** / 0,514 | **0,903** / 0,466 |
| **gpool_frz_gkd@e15** | **head** / gp | **0,803** (0,761) / 0,677 (0,557) | **0,775** (0,710) / 0,541 | **0,907** (0,842) / 0,501 |

**Quyết định:** luật chính theo nguồn teacher — họ `frz` (teacher = snapshot gộp cố định, đã mang prior toàn cục) đọc **head**: hơn gp
ở mọi góc (global +0,126, local F1 +0,234, local acc +0,406, client tệ nhất +0,20). Arm DML cũ (teacher = bản proxy học local) giữ **gp**
(global +0,12; local F1 ≈ nhau; head chỉ hơn local acc +0,025). So sánh giữa phương pháp: mỗi bên đọc luật chính của nó (gkd head 0,803
vs gpool gp 0,734 = +0,069; góc local F1 +0,066, local acc +0,015 so gpool head). Rò prior local còn lại trong head gkd (độ dốc
over-prediction theo log π_k − log π_g, `prior_leak`): m4b head +0,74, frz +0,18, gkd **+0,13**; gp trên frz/gkd −0,41…−0,47 (sửa quá tay).

## Vòng 17 — cải tiến nhỏ của gkd + gkd tới ep50 (owner 18/10 ~06Z; khóa ~06:25Z trước push)

Owner: gkd hoạt động ổn thì thử cải tiến nhỏ trong phương án; chắc chắn là cách tốt nhất thì chạy 50 round. Source tag `20261015-r17`
(r16 + token `la`, 3 arm, `--parent`); test `--part r17` (8 arm r16 bit-exact 2 epoch; la khác gkd từ ep1; kdpx privates ep1 == gkd,
proxy khác; 3 arm mới hữu hạn), `--part parent` (r16 chạy liền ep1→2 == ep1 của r16 + nối tiếp ep2 bằng source mới, serial và
workers 3: weights, predictions, eval) PASS.

| session | arm (s42) | bằng chứng → cơ chế | kỳ vọng / dấu hiệu sai |
|---|---|---|---|
| ra | `..._gkd_la:15` (ep1→15) | head gkd còn rò prior local (độ dốc +0,13) và khoảng tới bias oracle +0,045 là bias **riêng từng private** (probe 18–19): CE local kéo head về π_k, KD teacher kéo về π_g. `la`: CE(z + log π_k − log π_g, y) trên dòng local ⇒ cả ba số hạng cùng huấn luyện posterior prior toàn cục (FedLC); lớp vắng local không bị CE đè xuống | head ≥ gkd + 0,010; sai nếu Δ < +0,010 hoặc độ dốc rò không giảm (< 0,07) |
| rb | `..._gkd_kdpx:15` | private bám snapshot (khoảng proxy−private 0,03 suốt ep4→15); proxy frz thấp hơn m4b 0,018; `kdpx` (trần 1 cho KL proxy←private) từng nâng proxy +0,031 (vòng 11) nhưng hại private khi teacher là bản proxy. Với frz, teacher = snapshot gộp ⇒ proxy tốt hơn = teacher tốt hơn | private ≥ gkd + 0,010; sai nếu proxy không tăng ≥ +0,010 (cơ chế) hoặc proxy tăng mà private không theo |
| rc | `..._gkd_la_kdpx:15` | kiểm tổ hợp (LESSONS: không cộng lợi) | mô tả vs đơn lẻ; cổng vs gkd |
| rd | `fd_dml_m4b_gpool_frz_gkd:50`, nối tiếp từ ep15 audited của oc (`--parent`, SHA weights khớp marker) | headline tới 50 round (protocol full §1, cosine T50 đã khóa từ đầu) | mô tả horizon ep50; test một lần ở ep50 |

**Cổng (ep15, gain + worst: Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất ≥ −0,010):** ra/rb/rc head vs `gpool_frz_gkd@e15` head 0,8030
(cùng luật). Proxy head và độ dốc rò prior báo kèm (dấu hiệu cơ chế). Arm PASS: nối tiếp ep16→50 từ ep15 của chính nó (`--parent`) và lặp
seed 43; so ep50 với rd. R1: `la` 0 byte (π_g = đếm lớp gộp đã có cho gp/gpool); `kdpx` chỉ đổi trọng số loss proxy, 0 byte. R2: `la` chỉ
dịch logit private, `kdpx` chỉ ở proxy ⇒ private tùy kiến trúc.

## Vòng 18 — mọi cải tiến có bằng chứng trên gkd (owner 18/10 ~07Z: quota thoải mái, thử hết; khóa ~07:30Z trước push)

Source tag `20261015-r18` = r17 + token `cop`/`pce`/`t2`/`g512`/`gk2`, arm q4 + hai sửa FL ([kiểm toán](../../research/federated-information-flow.md)):
`la` lấy log π_g từ SecureSum (r17 tính trong tiến trình, lệch ≤ 2,4e-7), compile probe dùng dòng client của chính tiến trình (r17: client 0;
trạng thái/RNG khôi phục, không đổi kết quả). Mỗi arm s42 ep1→15 từ đầu, `--workers 2`.

| session | arm | bằng chứng → cơ chế | dấu hiệu sai |
|---|---|---|---|
| sa | `gkd_cop` | Biên Gaussian lệch thật: fixture 1M dòng, 53 feature m4b, KS trung vị 0,19 (p90 0,49), 33 % cặp lớp×feature có atom > 20 %. **Gaussian copula**: biên = histogram gộp theo lớp (256 bin asinh z, SecureSum cộng tính), phụ thuộc = tương quan normal-score gộp (phase 2 SecureSum). Kiểm local với snapshot gkd ep15: teacher đồng ý 0,673 vs Gaussian 0,473 (thật 0,805), benign 0,27 → 0,51, 1-NN 5,53 → 4,61 | head < gkd + 0,010, hoặc syn_accept trên Kaggle không tăng |
| sb | `gkd_pce` | private bám snapshot (−0,03 suốt ep4→15); proxy frz thấp hơn gpool 0,02–0,03; FedAvg proxy ≥ DML +0,008 (ep15). Proxy chỉ CE (bỏ KL←private) ⇒ teacher = FedAvg-m4b | proxy không tăng hoặc private không theo |
| sc | `gkd_t2` | lỗi còn lại trong khối 4 lớp nhầm; KD T=2 (×T²) cả dòng local và Gaussian chở xác suất tương đối giữa lớp không thắng | head < gkd + 0,010 |
| sd | `gkd_g512` | phủ input là phần còn lại (probe 15 cov_gauss +0,022/epoch); 512 dòng tổng hợp mỗi bước (KL batchmean ⇒ cùng trọng số, nhiều phủ hơn) | head < gkd + 0,010 |
| se | `gkd_gk2` | cùng lý do; trọng số KL trên dòng tổng hợp ×2 (so với CE local) | head < +0,010 hoặc client tệ nhất giảm |
| sf | `q4_frz_gkd` | R1: q2_frz_gkd W&B ep5 0,638 vs gkd 0,768 (frz phụ thuộc chất lượng snapshot); int4 từng ±0,001 | noninferior Δ ≥ −0,010 |

**Cổng (ep15):** sa–se head vs `gpool_frz_gkd@e15` head 0,8030 gain + worst; sf noninferior. Byte `cop` (một lần, đo ở test r18): lên histogram 868.392 B
+ normal-score 95.016 B/client, xuống histogram gộp 868.384 B + Cholesky tương quan 91.616 B (thay moment Gaussian 145.968 B lên / 72.912 B xuống); proxy mỗi epoch không đổi. R2: mẫu ở
input space, mọi kiến trúc private. Privacy: histogram theo lớp cùng giả định SecAgg như moment (ô 4–10 dòng). Arm PASS → tổ hợp (vòng 19)
rồi 50 round.

## Code nhanh `--fast` — lớp tolerance (owner 18/10 ~09:15Z)

Owner duyệt tối ưu lớp tolerance (`fused` AdamW + cuDNN autotune, bỏ thuật toán deterministic; local −25 %/bước,
[computational cost §6](../../research/computational-cost.md)) với điều kiện **so sánh luôn cùng điều kiện code**:

- Kernel đã chạy/đã build tới vòng 18 (gồm `rd` = gkd ep16→50 nối tiếp source r17) giữ code bit-exact cũ và so với đối chứng cũ.
- Từ vòng 19 mọi kernel build mới chạy `--fast`; arm chỉ so với đối chứng **cũng chạy `--fast`** (chạy lại gkd s42 ep15, và mọi đối chứng
  khác cổng vòng đó cần, kể cả seed 43 / ep50 nếu cổng dùng).
- Cưỡng chế bằng code: `--fast` ghi `config.fast = true` (fingerprint khác; nối tiếp `--parent` từ run không fast bị từ chối vì config khác);
  không cờ thì config và kết quả bit-exact source r18 (`test_fd_local --part fast`). `report_fd.py` gắn khóa `@fast`, nên cổng phải ghi rõ
  `<arm>@e15@fast` vs `<đối chứng>@e15@fast`. W&B tên run thêm `-fast`. `gen_fd.py`: plan dạng dict thêm `'fast': True`.
- Run `--fast` không bảo đảm lặp lại bit-for-bit (local cùng GPU thì lặp lại được), kể cả `--workers 1` vs `W`; mức chênh do đối chứng
  chạy lại xác định (phải nằm trong dao động seed). Test local PASS: [computational cost §6](../../research/computational-cost.md).

## Vòng 19 — code `--fast`: đối chứng chạy lại + tổ hợp PASS (khóa ~14:35Z 18/10 trước push)

Source tag `20261015-r19` = r18 + cờ `--fast` + arm `fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx`. Mọi session `--fast`, `--workers 2`, ep1→15 từ đầu.
Kết quả vòng 16–18 dùng: la_kdpx PASS +0,0195 (s42), la PASS +0,016, q4_frz_gkd PASS noninferior (byte proxy ÷7,6); kdpx, cop, t2, gk2, pce FAIL;
g512 đang chạy (W&B ep12 +0,002 — nếu PASS thì vào vòng sau).

| session | seed | arm | bằng chứng → cơ chế | dấu hiệu sai / cổng |
|---|---|---|---|---|
| ta | 42 | `gkd` | đối chứng code mới (owner: arm `--fast` chỉ so `--fast`) | `equiv`: \|head − gkd@e15 cũ 0,8030\| ≤ 0,010 (≈ chênh seed 0,008); lệch hơn ⇒ `--fast` đổi kết quả, dừng dùng |
| td | 43 | `gkd` | như ta, seed 43 | `equiv` vs gkd@e15@s43 cũ 0,7953 |
| tb | 42 | `gkd_la_kdpx` | rc +0,0195: `la` bỏ rò prior local khỏi head (độ dốc +0,133 → +0,032), `kdpx` thêm phần proxy | gain+worst vs ta (≥ +0,010) |
| te | 43 | `gkd_la_kdpx` | lặp seed trước khi chạy 50 round (luật lặp seed) | gain+worst vs td |
| tc | 42 | `q4_frz_gkd_la_kdpx` | R1: int4 giữ snapshot frz (sf Δ +0,001, q2 thì hỏng) ⇒ ghép với la_kdpx | noninferior vs tb (Δ ≥ −0,010, tệ nhất ≥ −0,010) |

Cổng trong `report_fd.py` (khóa `@fast`). Sau vòng: arm thắng nối tiếp ep16→50 bằng `--parent` từ ep15 `--fast` của chính nó, kèm gkd `--fast` ep50
làm đối chứng 50 round. Giới hạn: `load_parent` chặn arm q4 (trạng thái codec chưa lưu) ⇒ tc chỉ đến ep15 cho tới khi thêm resume codec.

## Vòng 20 — vét hướng la/kdpx (owner 19/10 ~07:40Z; khóa ~08:30Z trước push)

Owner: 50 epoch chỉ cho phương pháp đã tối ưu xong và thử hết khả năng của hướng đó, chạy theo protocol cuối (train toàn bộ 80%, test20%
mỗi round). Vòng 19 PASS: la_kdpx +0,021 / +0,014 (s42/s43), `--fast` tương đương. Source tag `20261015-r20` = r19 + token `la15`/`la05`/`lau`/`lap`
(`kd2` có từ vòng 11). Mọi session `--fast`, s42 ep1→15 từ đầu, `--workers 2`; đối chứng = tb (`la_kdpx` `--fast` s42, head 0,8269).

| session | arm | bằng chứng → cơ chế | dấu hiệu sai |
|---|---|---|---|
| va | `la15_kdpx` | sau la còn rò prior (độ dốc +0,032, gkd +0,133); τ là siêu tham số chính của logit adjustment ⇒ τ=1,5 khử nốt | head < tb + 0,010, hoặc benign giảm |
| vb | `la05_kdpx` | cận dưới của τ (la có thể sửa quá tay ở client lệch mạnh) | như trên |
| vc | `lau_kdpx` | dịch về prior đều thay vì prior gộp: F1 macro thưởng recall lớp hiếm | head < tb + 0,010 |
| vd | `la_lap_kdpx` | snapshot gộp (teacher) là trung bình proxy học trên dữ liệu lệch local; la trên CE proxy (kiểu FedLC) ⇒ teacher bớt lệch | head hoặc proxy không tăng |
| ve | `la_kdpx_kd2` | rd: teacher 0,885 vs private 0,815 ở ep50 ⇒ private chưa bám đủ teacher; KD ×2 (thật + Gaussian) | head < tb + 0,010 hoặc tệ nhất giảm |

Cổng gain+worst vs tb (khóa `@fast`). Arm PASS ⇒ tổ hợp + lặp s43; khi hướng hết arm thắng ⇒ đề xuất run cuối 50 epoch (protocol cuối, cần
thêm chế độ `--final`: train toàn bộ 80%, test20% mọi round).

**Kết quả (19/10 ~12:30Z): FAIL cả 5** (Δ −0,006…+0,001) ⇒ la τ=1 trên prior gộp + kdpx là điểm tối ưu của hướng; [results §Vòng 20–21](results.md).

## Vòng 21 — communication budget nhỏ nhất không giảm performance (owner 19/10 ~08:45Z; khóa trước push, song song vòng 20)

Cùng source tag `20261015-r20` (token mới `zl`, `ef`). Bằng chứng: q4 (int4, delta vs replica, làm tròn ngẫu nhiên, EF chỉ phía server/chiều xuống)
giữ la_kdpx (tc −0,004, byte proxy ÷7,6); q2 không EF hỏng với frz (−0,226: snapshot nhiễu). Đối chứng performance = tb (la_kdpx dense `--fast` s42, 0,8269).

| session | arm | cơ chế | dấu hiệu sai / cổng |
|---|---|---|---|
| wa | `q4_…_la_kdpx_zl` | thân gói (mã int4 + scale fp16 + BN fp16) nén LZMA2 không mất mát: delta dồn quanh 0 ⇒ entropy < 4 bit/tham số; train trùng bit với q4 | byte không giảm; noninferior vs tb |
| wb | `q4_…_la_kdpx_ef_zl` | + error feedback phía client chiều lên (gửi Q(delta + e_k), giữ e_k = phần dư): lỗi lượng tử không tích lũy ⇒ snapshot gần dense hơn | proxy/private không tốt hơn wa; noninferior vs tb |
| wc | `q2_…_la_kdpx_ef_zl` | 2-bit với EF hai chiều (server EF có sẵn): EF khử phần nhiễu đã làm q2 hỏng | noninferior vs tb (Δ ≥ −0,010, tệ nhất ≥ −0,010) |

Báo byte wire up/down (`R.bytes`) và tỉ lệ trước/sau LZMA (`q_zl_ratio_down`, `q_raw_up_mean`). Arm thắng = byte nhỏ nhất trong các arm
noninferior. Chưa làm: scale chung (SecAgg, ghi FL §7 d), lưu trạng thái codec để nối tiếp (bắt buộc trước run cuối dùng kênh nén).

**Kết quả (19/10 ~12:30Z):** wa PASS (−0,003, byte proxy 99,5 MB = 0,80 × q4), wb PASS (−0,005, byte bằng wa ⇒ `ef` bỏ), wc **sụp** (−0,514;
`q_residual_rel` 0,27 → 0,71: downlink 2-bit không theo kịp). Chọn `q4_zl`. Lưu trạng thái codec đã cài (resume bundle).

## Vòng 22 — kênh nén dưới int4 (khóa ~12:50Z 19/10 trước push; push 12:57Z; source tag `20261015-r22`)

Bằng chứng vòng 21: q4_zl giữ performance (−0,003); q2 hai chiều sụp **qua chiều xuống**: server lượng tử hóa (exact − replica) làm tròn gần nhất,
scale theo max hàng ⇒ ở 3 mức {−1,0,1}·max mọi phần tử < max/2 về 0, replica tụt lại (phần dư 0,27 → 0,71). Chiều lên làm tròn ngẫu nhiên (không lệch),
trung bình 20 client chia phương sai cho 20, + EF client ⇒ chịu được ít bit hơn. Token mới: `q3` (7 mức, mã 1..7 gói 2/byte, LZMA2 khử phần thừa),
`u2` (chiều lên 2-bit, chiều xuống giữ số bit của `qN`). Thử local 2 epoch smoke, byte proxy up+down so q4_zl: q3 −31 %, q4_u2 −32 %, q3_u2 −48 %.

| session | arm | cơ chế | dấu hiệu sai / cổng |
|---|---|---|---|
| xa (s42) | `q3_…_la_kdpx_zl` | 7 mức: ngưỡng về 0 là max/6 (q2: max/2) ⇒ downlink theo kịp; q4 có phần dư 0,008 | `q_residual_rel` tăng dần như q2; noninferior vs tb |
| xb (s42) | `q4_u2_…_la_kdpx_ef_zl` | chiều lên 2-bit không lệch + EF client; chiều xuống int4 như wa | private < tb − 0,010; noninferior vs tb |
| xc (s42) | `q3_u2_…_la_kdpx_ef_zl` | tổ hợp hai cái trên | **hoãn** (owner 19/10 ~12:50Z: tổ hợp chờ kết quả từng thành phần; chỉ push nếu xa và xb cùng PASS) |
| xd (s43) | `q4_…_la_kdpx_zl` | lặp seed của cấu hình đã chọn trước run cuối | noninferior vs te (la_kdpx `--fast` s43, 0,8128) |

Arm thắng = byte nhỏ nhất trong các arm noninferior. Bootstrap (state proxy ban đầu, 31,9 MB = 23 % tổng q4_zl) là init từ seed
(`torch.manual_seed(seed + 1000)`, codec dense không mất mát) ⇒ client tự dựng được, chỉ cần gửi seed: đổi kế toán byte, trùng bit, không cần
run; làm cùng run cuối nếu owner đồng ý.

**Kết quả (20/10 ~03Z):** xa `q3` PASS (−0,002, byte proxy 66,8 MB = −33 % so q4_zl), xb `q4_u2` **FAIL** (−0,080, proxy sụp; giả thuyết chiều lên
chịu 2-bit bị bác), xd q4_zl s43 PASS (+0,005). xc không push. [Results §Vòng 22](results.md).

## Vòng 23 — 2-bit với bước RMS (khóa ~03:05Z 20/10, push 03:14Z; source tag `20261015-r23`)

Bằng chứng: q2 hỏng ở cả hai chiều (wc downlink, xb uplink) còn q3 PASS. Chung một nguyên nhân là **scale theo max hàng**. Đo trên update proxy
thật (tb, ep6→ep15, `scratchpad an/levels.py`): crest max/RMS trung vị 3,2 (p90 5,2) ⇒ **93 %** phần tử < max/2. Sai số tương đối:

| bộ lượng tử | gần nhất (downlink) | ngẫu nhiên (uplink) |
|---|---:|---:|
| q2 scale max (FAIL) | 0,834 | 1,447 |
| **q2 bước RMS Lloyd-Max (1,224σ, cắt ±1)** | **0,501** | **0,593** |
| q3 scale max (PASS) | 0,399 | 0,578 |
| q4 scale max (PASS) | 0,179 | 0,254 |

Cơ chế: bước 1,224·RMS là lượng tử tối ưu MSE 3 mức cho nguồn Gauss (hằng số lý thuyết, không tune). Phần bị cắt nằm lại trong EF server (xuống);
chiều lên làm tròn ngẫu nhiên, nhiễu ≈ q3 đã PASS mà không cần EF client. Token `rs` (cả hai chiều; arm khác bit-exact, test `final`).
Byte dự kiến (smoke 2 epoch): −32 % so q3.

| session | arm | căn cứ | dấu hiệu sai / cổng |
|---|---|---|---|
| ya (s43) | `q3_…_la_kdpx_zl` | q3 PASS s42 (xa) ⇒ lặp seed trước khi chọn | noninferior vs te |
| yb (s42) | `q2_rs_…_la_kdpx_zl` | sai số uplink ≈ q3 (0,593 so 0,578); downlink 0,50 so 0,40 nhưng có EF server | `q_residual_rel` không giảm dần như q3, proxy sụp; noninferior vs tb |

Giới hạn bằng chứng: delta đo qua 9 round gộp (local chỉ giữ weights ep6/ep15), update một round có thể đuôi dày hơn.
Còn lại sau vòng này: bootstrap bằng seed (trùng bit, không cần run), entropy coder tốt hơn LZMA2 (không mất mát; chỉ đổi byte).

**Kết quả (yb audited 20/10 ~08Z):** q2_rs **FAIL** (−0,013 vs tb, tệ nhất −0,001, proxy −0,024); residual 0,094 → 0,078 không giảm như q3 (dấu hiệu sai trúng), proxy không sụp; byte −22 % so q3 (dự kiến −32 %). 2-bit loại. ya-v1 dừng (máy chậm); ya-v2 đã audit đủ ep15, q3_zl s43 PASS (+0,0032, tệ nhất +0,0054), byte 66,7 MB. Chốt q3_zl cả hai seed. [results §Vòng 23](results.md).

## Vòng 24 — sinh mẫu có chủ đích: lai ghép dòng own theo cặp lớp bất đồng (owner 21/10; khóa ~06:30Z trước push)

Owner 21/10: ngoài Gaussian, đặt **ưu tiên** để sinh dữ liệu có chủ đích hơn; kiểm build cũ, chưa có thì đề xuất và thử.
**Đã thử trước đây** (đều ở chế độ gpool nhãn cứng hoặc chỉ đổi lượng/độ thật): ưu tiên lớp hiếm local `fill` FAIL; lọc theo teacher `rej`
+0,011 s42 / +0,008 s43 (bị gkd thay); gate P1 FAIL; hỗn hợp mode P2 không tín hiệu; tinh chỉnh theo proxy (probe 17) chỉ offline +0,012…+0,021;
trên gkd: `cop` +0,003, `g512` +0,001, `gk2` 0,000. **Chưa build nào ưu tiên theo chỗ private bất đồng teacher.**

**Bằng chứng (probe 20, không train, weights xa ep15 s42, fixture pilot local; [results §Probe 20](results.md)):** private bất đồng snapshot 5,1 %
dòng group, 41 % số lỗi "teacher đúng, private sai" là benign↔timeDelay, phần lớn còn lại benign↔positionMirroring/dataReplay/suddenStop. Dòng own
của chính client cho **cùng cặp** (cos gộp 0,94) ⇒ xe biết được chỗ cần dạy mà không cần byte. Mẫu Gaussian gkd bất đồng 12,7 % nhưng **sai chỗ**
(constantSpeedOffset→randomSpeedOffset; cos với lỗi thật 0,09); ưu tiên trong Gaussian (top-KL, đối kháng, lớp/cặp từ dòng own) vẫn cos ≤ 0,12;
nhiễu covariance gộp quanh dòng own cũng về randomSpeedOffset ⇒ giá trị liên tục ở feature tốc độ (thật là atom) làm mẫu lệch. Lai ghép hai dòng
own theo cặp ưu tiên (Gx3) cos 0,55, "bất đồng đúng chỗ" 0,069 (Gaussian 0,011, dòng thật client khác 0,048); cùng lai ghép cặp đều 0,045.

**Cơ chế:** mỗi bước thêm 256 dòng: dòng own lớp a, mỗi feature lấy từ một dòng own lớp b với xác suất ρ = 0,3; (a, b) rút theo histogram
(teacher argmax a, private argmax b) trên 65.536 dòng calibration own đầu epoch (90 %) + 10 % đều trên cặp client có cả hai lớp. Dòng lai học
KL(snapshot ‖ private) như dòng Gaussian gkd (cùng trọng số `wpriv`, T 1); CE nhãn cứng chỉ trên dòng thật. Dòng lai nằm giữa hai lớp mà private
đang tách khác teacher ⇒ private nhận ranh giới toàn cục đúng ở vùng nó nhầm, vẫn giữ nhãn own trên dòng nguyên. Round 1 (chưa snapshot) không có
dòng lai ⇒ = control. Token `xo`; `xou` = cùng dòng lai, cặp đều (đối chứng ưu tiên). **R1:** 0 byte. **R2:** chỉ input-space + logit.
Không record nào rời xe. Chi phí: +256 dòng private fwd/bwd và snapshot fwd mỗi bước từ round 2, chấm 65.536 dòng own × 2 model mỗi epoch.

| session | arm (s42, ep1→15 `--fast`, `--workers 2`, source tag `20261015-r28`) | vai trò |
|---|---|---|
| xo | `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xo` | lai ghép có ưu tiên |
| xu | `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_xou` | đối chứng: cùng lai ghép, cặp đều |

**Cổng (ep15, private head trên group; đối chứng xa = cùng arm không token, s42 `--fast`, 0,8246):** xo PASS khi Δ ≥ +0,010, benign ≥ −0,005,
client tệ nhất ≥ −0,010. Ưu tiên có tác dụng khi xo − xou ≥ +0,005 (chỉ đọc nếu xo PASS). **Dấu hiệu sai:** xo < +0,010 ⇒ "bất đồng đúng chỗ"
của probe không dự báo utility (như cop) ⇒ dừng hướng, không quét ρ/floor; xo ≈ xou ⇒ lợi do lai ghép, không do ưu tiên. Chẩn đoán đọc kèm:
`xo_disagree` (bất đồng trên dòng lai), `xo_pairs` (cặp ưu tiên phải gồm benign), `xo_disagree_own`. Dự báo không chắc: cop có align 0,034 gấp 3
Gaussian mà chỉ +0,003. Test local `--part r24` (arm cũ bit-exact với r25 hai epoch, xo/xou epoch 1 = control) và `--part par` (compiled, 1 vs 2 worker).
Chạy trên odixeuit (ngoài 14 account của run cuối, 30h). Screening một seed; PASS mới lặp s43 + cân nhắc đưa vào run cuối (owner quyết).

**Kết quả 21/10 ~12Z: xo FAIL** gain (Δ +0,0076, benign +0,0063, tệ nhất +0,0113); cơ chế đúng chỗ nhưng lợi trong nhiễu seed ⇒ dừng,
không s43, không quét ρ/floor. [Results §Vòng 24](results.md).

## Vòng 25 — teacher hiệu chỉnh bằng đặc trưng ảo theo lớp (`cv`; owner 21/10 ~16Z: **chỉ cvl**, ngân sách ≤ 1,5× byte; cổng khóa, push r30) — **FAIL 22/10** (Δ −0,0136, benign −0,0185; dấu hiệu (a)+(c) ⇒ dừng; [results §Vòng 25](results.md))

**Bằng chứng** ([results §Vì sao sinh mẫu không ăn](results.md)): 77 % lỗi private headline là lỗi teacher cũng sai ⇒ mọi mẫu KL từ teacher
chạm tối đa 23 %; refit head teacher trên dòng thật gộp 0,842 → 0,919; CCVR từ moment lớp (cộng được) 0,893, phần lỗi private teacher đúng
21 → 47 % (probe 22). **Cơ chế** ([hồ sơ](../../research/methods/teacher-virtual-feature-calibration.md)): đầu mỗi round từ round 2, client gửi
(n_c, Σh, Σhhᵀ) của h256 snapshot trên 65.536 dòng calibration own; server rút N(μ_c, Σ) theo tỷ lệ lớp gộp, fit lại `head.5` từ trọng số
snapshot (Adam 3.000 × 4.096, lr 1e-3, jitter 1e-4); head đó thay `head.5` của cả teacher frz (KD dòng thật) và teacher gkd (KL dòng
Gaussian). Proxy train như cũ (vẫn trao đổi proxy q3 mỗi round; cv thêm một lượt thống kê, không thay). `cvf`: covariance riêng 16 lớp
(lên 2,12 MB/client/round); `cvl`: một covariance chung (148 kB); xuống 16,4 kB.

**Byte (sửa 21/10 ~16Z, theo `result.json` xa đã audit, mỗi client):** xa = 5,15 MB/15 epoch (bootstrap một lần 1,81 MB + q3 223 kB/round),
12,95 MB/50 round. cvf ×6,8 (15 ep) / ×9,1 (50) ⇒ **loại** theo ngân sách owner ≤ 1,5×. cvl ×1,45 (15 ep) / ×1,62 (50) / ×1,74 theo round;
gửi 16-bit sẽ ×1,25 / ×1,34 / ×1,41. Số +48 %/+3,3 % trước đây sai (byte cv một client so với kênh q3 cộng 20 client).
R1 B2 (231 kB/contact): round xa 223 kB vừa B2, round cvl 387 kB vượt ⇒ thống kê cần contact riêng hoặc bản 16-bit; báo kèm.

| session | arm (s42, ep1→15 `--fast`, `--workers 2`, source tag mới) | vai trò |
|---|---|---|
| cl | `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvl` | teacher hiệu chỉnh, covariance chung (khanhmay0304, tag r30, 6,3h) |
| ~~cf~~ | `…_cvf` | không chạy: ×6,8 byte |

**Cổng (khóa 21/10 ~16:15Z trước push; ep15, private head trên group, đối chứng xa s42 `--fast` 0,8246):** cl PASS khi Δ ≥ +0,010,
benign ≥ −0,005, client tệ nhất ≥ −0,010, và byte/client ≤ 1,5× xa ở 15 epoch (×1,45). PASS ⇒ trước run dài: bản 16-bit (×1,34 ở 50 round)
kiểm offline F1 bằng bản fp32, rồi s43; owner quyết đưa vào run cuối.
Chẩn đoán mỗi round (dòng own, nhãn own; không group/test): `cv_teacher_f1_orig/cal`, `cv_teachable_orig/cal`, `cv_server_seconds`.
**Dấu hiệu sai:** (a) `cv_teacher_f1_cal − orig` < +0,02 phần lớn round ⇒ moment không đủ khi backbone trôi ⇒ dừng; (b) teacher lên mà
private < +0,010 ⇒ nút thắt là kênh truyền (mẫu Gaussian lệch đa tạp) ⇒ hướng phụ đảo ngược theo moment, không quét biến thể cv;
(c) benign < −0,005 như pce. Test local `--part r25` (arm cũ bit-exact với r28 hai epoch; cv epoch 1 = control; tổng client = moment gộp
trực tiếp; fit tách lớp toy; head chỉ thay `head.5`) và `--part par --arms <cv>` (compiled, 1 vs 2 worker).

## Vòng 26 — teacher Project48 (`cvp`) + support copula (`cop`) (owner 22/10 ~13:40Z "thử nghiệm cả hai phương án"; cổng khóa ~14:55Z, push r31) — **FAIL 22/10** (Δ +0,0003, benign +0,0053, worst +0,0026; dấu hiệu (a) ⇒ dừng; [results §Vòng 26](results.md))

**Bằng chứng** ([P32–36](../../research/methods/teacher-support-two-seed-2026-10-22.md)): frozen-head hai seed, so phương pháp hiện hành (teacher gốc +
Gaussian): Project48 + copula **+0,0077 / +0,0051** mean, benign +0,010 / +0,007, worst +0,005 / −0,002 (PASS cả hai); Project48 một mình hại worst
(−0,017 / −0,025); copula với teacher gốc không thêm gì ⇒ lợi là tương tác teacher tốt hơn × support thật hơn. **Cơ chế:** như `cv` vòng 25 nhưng
thống kê lớp trên phép chiếu 48 chiều của h256 (16 hướng head snapshot + bù ngẫu nhiên cố định seed 23, QR CPU, mọi bên tự dựng, 0 byte), covariance
riêng từng lớp ở đó, head fit giới hạn trong phép chiếu (probe 23 Project48); support Gaussian của gkd thay bằng copula one-shot vòng 18 (`cop`).

**Byte (mỗi client, theo xa đã audit 5,16 MB / 15, 12,96 MB / 50):** cvp lên 78.400 + xuống 16.448 B mỗi round có snapshot (14 / 49 lần);
copula một lần 1,92 MB (fp32, đo P32) ⇒ **×1,63 (15 round), ×1,51 (50 round)** — vượt ngân sách 1,5×; bản copula mã không mất ×1,55 / ×1,47.
Screen giữ đúng cấu hình đã PASS ở harness; nén là bước R1 bắt buộc trước run dài (không phải cổng của screen này, khác vòng 25).

| session | arm (s42, ep1→15 `--fast`, `--workers 2`, tag r31) | vai trò |
|---|---|---|
| cp | `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_cvp_cop` | minhtrit06, 6,3 h; push 22/10 14:53Z |

**Cổng (khóa trước kết quả; ep15, private head trên group, đối chứng xa s42 `--fast` 0,8246):** cp PASS khi Δ ≥ **+0,010**, benign ≥ −0,005, client
tệ nhất ≥ −0,010. PASS ⇒ nén R1 (kiểm F1 bản nén offline bằng bản fp32) rồi s43; owner quyết đưa vào run cuối.
Chẩn đoán mỗi round như vòng 25 (`cv_teacher_f1_orig/cal`, `cv_teachable_orig/cal`, dòng own). **Dấu hiệu sai:** (a) teacher hiệu chỉnh − gốc < +0,02
phần lớn round ⇒ thống kê chiếu không đủ khi backbone trôi; (b) teacher lên mà private < +0,010 ⇒ frozen-head đánh giá cao kênh truyền online;
(c) benign < −0,005. Test local `--part r26` (arm cũ bit-exact với r30 hai epoch; cvp epoch 1 = control; byte 78.400 / 16.448; tổng client = moment
chiếu gộp trực tiếp; cơ sở trực chuẩn, chứa head; fit 0 bước trả head snapshot) và `--part par` (compiled, 1 vs 2 worker) **PASS**.

## Kiểm chứng m4b trên recipe hiện hành — build r32/r37 (owner 24–25/10; đã audit)

Mục tiêu: tách tác động của 13 feature m4b khỏi toàn bộ recipe đã chọn
`fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl`. Giữ seed 42, `--fast`, `--workers 2`, epoch 1→15,
FullDrift internal-holdout fixture và chỉ chấm `group` + `old`; không chấm official test20%, không dùng protocol `--final`.

| session | arm | input model |
|---|---|---|
| fa | `fd_dml_full66_gpool_q3_frz_gkd_la_kdpx_zl` | đủ 66 feature, không mask |
| fb | `fd_dml_drop53_gpool_q3_frz_gkd_la_kdpx_zl` | bỏ đúng `BLOCK + POSITION` (13 cột), model nhận 53 feature |
| control đã có | `fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl` | 66 feature, 13 cột đặt 0 |

DAGSNet cần số chiều chia hết cho `patch_len=6`; arm drop53 pad một số 0 **sau input model** để reshape 54 = 9×6.
Số pad không phải feature dữ liệu, không có tham số, không thay đổi 53 chiều đầu vào. Drop53 làm trục vị trí từ 11 còn 9;
đây là hệ quả trực tiếp của yêu cầu bỏ vật lý 13 feature. Checkpoint ghi `n_features=53,input_pad=1`; fast evaluator dựng cùng layout.

Kết quả hợp lệ: `fa` full66 r32-v1 và `fg` drop53 r37-v3 đều COMPLETE, pull + audit 25/10; `fb` r32 và `fg` r35
không hợp lệ do thiếu transform ở train/compile/eval rồi `compile_extra`. Artifact mới giữ cùng pattern với control m4b `xa`:
15 prediction/marker, weights round6/15, config/result/log/source và `AUDITED.json`. [Số liệu](results.md#kiểm-chứng-feature-trên-recipe-hiện-hành--seed42-ep15-audit-2510).

## Kiểm chứng feature trên FedAvg — build r33 (owner 24/10; chưa chạy)

Port theo FedAvg chuẩn sẵn trong harness hiện hành: một shared DAGSNet, mọi client tham gia, một local epoch rồi trung bình
tham số; AdamW, batch 512, seed 42, cosine T50 giữ nguyên. Chạy epoch 1→15, `--fast --workers 2 --holdout_only`, chỉ
chấm `group` và `old`. Ba arm chỉ khác xử lý input:

| session | arm | input |
|---|---|---|
| fc | `fd_fedavg_full66` | đủ 66 feature |
| fd | `fd_fedavg_drop53` | bỏ vật lý đúng 13 cột; model nhận 53 rồi pad nội bộ 1 số 0 để reshape |
| fe | `fd_fedavg_m4b` | giữ 66 feature; đúng 13 cột đặt 0 trong train và eval |

FedAvg dựng shared model từ `run.architecture`; full66/m4b giữ cấu hình 66, drop53 ghi
`n_features=53,input_pad=1`. Cùng seed khởi tạo `seed+1000`, aggregation, optimizer, data và evaluator; không tái dùng
kết quả FedAvg recipe cũ vì chúng không cùng internal-holdout contract.

Cập nhật 24/10: `fc` full66 và `fe` m4b đã rebuild thành `r37-v1` bằng wrapper không W&B và source r37 đã sửa; audit
AST/metadata/source SHA PASS. Cả hai đã push lên `hoangdang1711` sau khi cb/cc COMPLETE; kiểm ngay sau push đều RUNNING.
`fh` drop53 hợp lệ dùng r37-v3 trên `dangthiet`.

## Kiểm chứng feature trên 3 phương pháp cũ + centralized — build r34 (owner 24/10; đã build, chưa push)

Giữ đúng contract r33: seed42, 20 client, batch512, cosine T50, round 1→15, `--fast --holdout_only`, chỉ chấm
`group` + `old`, không chạm official test20%. Mỗi phương pháp có `full66`, `drop53`, `m4b`; khác nhau duy nhất ở xử lý input.
FedProx/FedTGP/CanPerFL dùng `--workers 2`; centralized dùng một process/GPU vì chỉ có một optimizer/model pooled.

| phương pháp | sessions | recipe khóa |
|---|---|---|
| FedProx | pa / pb / pc | FedAvg + proximal `μ=0.01` về snapshot đầu round |
| FedTGP | ta / tb / tc | prototype 544-d, client MSE weight 1, server ACL 100 bước lr1e-3, `τ=100`; báo cả shared-head và nearest-prototype |
| CanPerFL | ca / cb / cc | FedAvg 15 round; sau cùng Fed-FN chỉ fine-tune head 10 epoch, tối đa 30.000 dòng/client |
| Centralized | na / nb / nc | một shared DAGSNet trên pool train; một epoch pooled mỗi round, cùng AdamW/LR/batch/clip hiện hành |

Notebook: `fd-s42{pa,pb,pc,ta,tb,tc,ca,cb,cc,na,nb,nc}-20261015-r34-v1`. Source dataset dự kiến:
`hoangdang1711/mobihet-fulldrift-src-20261015-r34`. Tất cả notebook private, dùng Kaggle secret `wandb_key`, không nhúng key.
Build mới chỉ qua compile/static/schema/hash audit; máy local không có PyTorch/GPU nên chưa có runtime smoke hay kết quả khoa học.

## Sửa drop53 — build r35 (owner 24/10)

Không sửa frozen r32/r33. Rebuild đúng hai arm bị ảnh hưởng bằng source đã áp transform 66→53 ở train, compile probe và eval:

| session | arm | trạng thái |
|---|---|---|
| fg | `fd_dml_drop53_gpool_q3_frz_gkd_la_kdpx_zl` | source/notebook audit PASS; push v1 account `hoangdang1711` 24/10 11:34Z |
| fh | `fd_fedavg_drop53` | source/notebook audit PASS; đã build, chưa push |

Cả hai giữ seed42, round1→15, `--fast --workers 2 --holdout_only`; source private dùng chung
`hoangdang1711/mobihet-fulldrift-src-20261015-r35`, 49 file đã verify sau upload.

## Build r37 — bỏ W&B + sửa `compile_extra` drop53 (owner 24/10; main drop53 đã audit)

`r35` main drop53 dừng trong frozen-teacher compile gate: model 53 chiều nhưng `compile_extra()` lấy thẳng tensor 66 chiều.
`r37` áp đúng transform `mask()` trước mọi eager/compiled extra gate. Notebook không cài/import/login W&B, chỉ ghi Kaggle log và
artifact; source private đã upload/verify 49 file riêng cho `dangthiet` và `hoangdv1711`.

Wrapper v1/v2 không hợp lệ do xóa cell W&B kéo theo import ẩn `os/subprocess/sys`; v3 import runtime độc lập và là bản đầu tiên
được dùng. Bốn kernel v3 đã push: `fg` main-drop53, `fh` FedAvg-drop53 trên `dangthiet`; `pa` FedProx-full66, `pb`
FedProx-drop53 trên `hoangdv1711`. Cả bốn giữ seed42, round1→15, `--fast --holdout_only`; kết quả chỉ hợp lệ sau collect/audit.

`fg` r37-v3 COMPLETE và audit 25/10: 15/15 round, 180 metric checks, weights round6/15 strict-load + finite;
đây là kết quả drop53 duy nhất được dùng cho main recipe.

Mười arm còn lại đã rebuild `r37-v1`, chưa push: `pc`; `ta/tb/tc`; `ca/cb/cc`; `na/nb/nc`. Audit 24/10 PASS:
không chuỗi/flag W&B, code-cell AST hợp lệ, metadata private + 2×T4, đúng workers (FL=2, centralized=1), source-stage `fd.py`
trùng SHA live và chứa fix `compile_extra`. Metadata owner hiện là placeholder `hoangdang1711`; khi chia account phải generate suffix mới
với `--owner`, upload cùng source r37 vào đúng owner rồi mới push.

Phân lô `r37-v4` 24/10: `hoangdangnhi` = pc+na, `phamquang0606` = ta+nb, `qwerasdf` = tb+nc,
`ttalgies` = tc+ca, `hoangdang1711` = cb+cc. Owner sau đó xác nhận account đúng là `qwerasdf3004`; build v5 tb+nc,
source r37 verify đủ 49 file và push thành công. Như vậy đủ 10/10 arm đã push: 8 arm v4, tb+nc v5; mọi owner dùng
source đã verify, không W&B. qwerasdf cũ (`slug/hashlink null`) không dùng.

Centralized r37 cả ba mode lỗi round 1 sau compile: pooled scaler dùng dữ liệu 20 client nhưng guard giữ `skipped <= 16`
của một client; đồng thời `steps` bỏ sót batch ở ranh giới chunk. Recovery r38 đếm `batches` thật và dùng ngân sách tương đương
Fed path `skip_budget = 16*K` (K=20), không đổi model/loss/optimizer/update. na/nb/nc r38-v1 đã audit, source riêng từng owner
ready/49 file và push 24/10; cả ba RUNNING, chờ log round 1 xác nhận `CENTRALIZED AMP` + commit.

Startup check 25/10: nc dựng xong fixture, start pooled train và còn RUNNING sau hơn thời lượng 11,3 phút mà r37 chết;
na/nb cũng RUNNING. Điều này xác nhận false-fail cũ không tái diễn tại cùng mốc; kết quả vẫn pending tới COMPLETE + collect/audit.

## Run cuối 50 epoch — chế độ `--final` (owner 19/10; cài ~09:30Z, chưa chạy)

Protocol §3: train **mọi dòng train80%** (fixture `papers/final/fixture`, `fd_prepare.prepare_fd(final=True)` = `mh.prepare(holdout=False)`, như
build baseline q5), chấm **test20% (10.761.343 dòng) sau mọi round** cho private (head + gp), private_ema, proxy; headline round 50, không chọn
checkpoint. `--final` ghi `config.final` (fingerprint riêng, report khóa `@final`), chỉ nối tiếp parent `--final`; `gen_fd.py` plan dict
`'final': True` (cell setup chuẩn bị fixture final). Test `--part final` PASS.

Cập nhật 19/10 ~12:50Z: resume codec đã cài (`resume/round_NNN.pt` = server `exact` + `ef` mọi client; test `resume` PASS) ⇒ run cuối dùng
được kênh nén.

Ước lượng: train `--fast` ~12 phút/epoch + eval 21 model trên test20 (~40 s/model/GPU) ~7 phút ⇒ ~16h ⇒ **2 session nối `--parent`** (dense
nối tiếp chính xác). Kênh nén đã lưu `exact` (server) + `ef_state` (client) trong resume bundle; fixture q4/q3 đã kiểm nối tiếp. Predictions mỗi round ~nhiều trăm MB nén ⇒ ước 5GB/run (như q5).
Chỉ build run cuối khi vòng 20/21 đã chọn xong cấu hình và owner đồng ý hướng đã vét hết.

**Owner 20/10 ~03:30Z:** run cuối chạy **20, 50 và 100 client** như `fl_dagsnet_veremi` (batch 512 / 512 / 256; dữ liệu `trietbackup/veremi-fl-20client`,
`minhtriethihi/veremi-fl-50client`, `khanhmay0304/veremi-fl-100client`), chỉ sau khi vét hết. Cần trước khi build: kiểm `fd.py` với K = 50 / 100
(fixture, partition, moment Gaussian theo client, eval N model mỗi round) bằng test local; ước giờ (100c: gấp đôi số bước vì batch 256, eval 100 model).
Chỗ code còn cố định 20 client: `fd.py` `Run.__init__` (`self.K = 20`), `fd_prepare.py` (`clients=range(20)`, dataset `veremi-fl-20client`).
Eval là nút thắt ở 100c: hiện chấm private + private_ema + proxy = 2K + 1 model × 10,76M dòng mỗi round (ước ~40 s/model/GPU ⇒ 100c ~67 phút/round,
NILM-FL/PerFed-SKD đo 1.360–1.425 s/round cho 101 model).

**Owner 20/10 ~03:40Z:** giữ chấm private_ema (đánh giá hiệu quả module). Đề xuất của owner — **tách train và chấm**, đánh giá khả thi:
train chỉ train + commit `weights/round_NNN.pt` (đã chứa private, private_ema, proxy, kiến trúc, fingerprint, chuỗi sha) + resume + prior nhãn
client (luật gp); notebook chấm `fd_eval.py --rounds a-b` dựng lại 2K + 1 model (strict, kiểm sha theo marker), chấm test20 deterministic, ghi
`eval/round_NNN.json` + predictions; collector gộp. Lịch: account A train s1 (r1..x) → A chấm r1..x (kernel_sources của chính nó) song song account B
train s2 (`--parent`, `upload_parent.py`); chấm tách theo khoảng round chạy song song nhiều account. Weights/round 65 / 161 / 320 MB (20c / 50c / 100c),
50 round 3,3 / 8 / 16 GB (output session ≤ ~20 GB ⇒ 100c ~25 round/session). Kiểm trước: chấm từ weights == chấm trong vòng (trùng bit, deterministic,
smoke 20c và K ≠ 20, 1 vs 2 worker). Protocol giữ nguyên (test mọi round, không hồi tiếp vào train).
**Owner 20/10: run cuối phải có state dict đầy đủ để dựng lại model từng round.** `fd.py` đã ghi `weights/round_NNN.pt` mọi round (private ×K,
private_ema ×K, proxy, BN, kiến trúc, fingerprint); chỗ thiếu là **pull**: `collect_fd.py` chỉ lấy weights round 6/15/50 (regex dòng 32). Run cuối:
collector lấy weights (+ `resume` nếu kênh nén) **mọi round**; audit mỗi round: sha == marker, chuỗi `prev_sha`, `load_state_dict(strict=True)`
từ `weights_only=True`, đếm tham số, hữu hạn; notebook chấm từ chối round thiếu/sai sha. Lưu trữ local ~27 GB cả 3 kịch bản (DVC khi owner yêu cầu).

**Owner 20/10 ~15Z (phiên hiện tại):** session đầu **4 epoch**, pull và audit weights, rồi **mỗi round một notebook chấm** trên account khác, mỗi notebook chia client/EMA cho 2 GPU. Kernel chấm khởi chạy ổn thì được train tiếp song song, không cần chờ chấm xong. Đo communication cost trong harness hiện tại (không thêm contact). Triển khai và budget: [final pipeline](../../operations/final-pipeline.md).

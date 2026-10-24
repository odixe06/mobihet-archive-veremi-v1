# FullDrift — kết quả đã audit

**Chốt20/10:** q3_zl PASS cả seed42/43, chọn cho final; 2-bit FAIL. Bảng dưới giữ lịch sử theo vòng.
**Cập nhật22/10:** final seed42 đủ6 chuỗi×50 round train/chấm/audit; [report hoàn chỉnh](results/final/report.md),
[export VERIFIED](results/final/VERIFIED.json), [kernel/source provenance](results/final/provenance.json).
Report ghi riêng giới hạn contact R1, heterogeneous R2 và single-seed. Final r24 crash trước round1 là lịch sử startup,
không phải thiếu kết quả hiện hành. [Bàn giao](../../archive/operations/continuation-2026-10-22.md).

Synthetic sau cvl: [P29–31](../../research/methods/synthetic-gradient-diagnosis-2026-10-22.md): joint carrier P30 FAIL,
P31 không đủ quy lỗi chính cho mất chiều; proposal density chưa chạy. Probe xa ep15/fixture không trộn vào final50.

Contract khóa trước push: [design](design.md). Full 100% train80% trừ holdout nội bộ (train 40,0M dòng),
group/old 65.536 dòng; test chỉ epoch 6, mô tả. Sessions `annhannh331/mobihet-fd-s42{a,b}-20261015-v2`,
`anhtruong1909/mobihet-fd-s42c-20261015-v2`, mọi arm đủ epoch 6; audit executed source, input/fixture SHA,
holdout manifest (group overlap 0), marker chain, weights epoch 6 strict/finite, predictions → 10 metrics.
Bảng đủ: [report](results/report.md) · [gates.json](results/gates.json) · [10 metrics](results/metrics-all.csv) ·
[per-class](results/per-class-f1.csv).

## Epoch 6, group holdout (Δ so đối chứng cùng seed)

| arm | primary | Δ | benign | Δ benign | drift | test (mô tả) | cổng |
|---|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,7631 | — | 0,565 | — | 0,384 | 0,7715 | — |
| fd_dml_prox (μ=0,01) | 0,7624 | −0,001 | 0,662 | +0,098 | 0,076 | 0,7510 | FAIL primary |
| fd_dml_avgm (β=0,9) | 0,0537 | −0,709 | 0,000 | −0,565 | 0,295 | 0,0479 | FAIL, phân kỳ từ ep2 |
| fd_fedavg | 0,7724 | +0,009 vs DML | 0,525 | — | 0,385 | 0,7655 | mô tả |
| fd_gaussian (private gp) | 0,5882 | — | 0,297 | — | — | 0,6074 | — |
| **fd_gaussian_swappos** | **0,6404** | **+0,052** | 0,427 | +0,130 | — | 0,6540 | **có triển vọng** |
| private_ema Gaussian / swappos | 0,6126 / 0,6768 | +0,024 / +0,036 | | | | | **có triển vọng** |
| private_ema DML arms | | −0,026…−0,040 | | | | | FAIL |

## Đọc

- FedProx giảm drift 5 lần nhưng macro F1 không đổi; benign giữ tốt hơn rõ (+0,098).
- FedAvgM β=0,9, η=1 phân kỳ khi update local đã lớn (4.200 bước). Không dùng nếu chưa giảm server LR.
- FedAvg thuần ≈ DML proxy (+0,009). KD với private không cải thiện model triển khai ở full.
- Lỗi còn lại vẫn nằm trong khối {benign, dataReplay, positionMirroring, timeDelay}.
- Gaussian ở full: swappos ngăn suy giảm sau epoch 1 (0,635→0,640, đối chứng 0,617→0,588); ghép EMA được 0,677.
  EMA hại private DML vì private DML còn đang tăng qua các epoch.
- Một seed, epoch 6/50: chỉ là sàng lọc.

## Vòng 3 — mục tiêu client (audited 16/10 ~08:40Z)

Sessions `nguyenminhtran060113/mobihet-fd-s42{g,h}-20261015-r3-v1` COMPLETE, AUDITED (executed source, input/stage SHA,
data manifest, marker chain, ep6 weights strict/finite, predictions → confusion → 10 metrics). Seed 42, group ep6,
primary **private gp** (owner 16/10), đối chứng `fd_dml` vòng 1. Cổng khóa trước push ([design §Vòng 3](design.md)).

| arm | private gp | Δ | benign Δ | proxy head (Δ) | local F1 / acc private gp | client tệ nhất | test proxy | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,6655 | — | — | 0,7631 | 0,637 / 0,794 | 0,595 | 0,7715 | — |
| **fd_dml_m4b** | **0,7010** | **+0,036** | **+0,113** | **0,8146 (+0,052)** | 0,683 / 0,854 | 0,617 | 0,8116 | **PASS** |
| **fd_dml_swappos** | **0,6985** | **+0,033** | **+0,107** | **0,8121 (+0,049)** | 0,681 / 0,845 | 0,613 | 0,8099 | **PASS** |
| fd_dml_ntd | 0,6236 | −0,042 | −0,034 | 0,7645 (+0,001) | 0,610 / 0,768 | 0,400 | 0,7695 | FAIL |
| fd_dml_q4 | 0,6613 | −0,004 | 0,000 | 0,7651 (+0,002) | 0,633 / 0,789 | 0,577 | 0,7716 | PASS noninferior |

Đọc:
- **Bỏ khối feature receiver (m4b/swappos) là kỹ thuật mạnh nhất ở full, cho cả client lẫn proxy**, 0 byte thêm
  (R1) và chỉ là tiền xử lý phía client (R2): private +0,036, proxy +0,052. Pilot r20 từng cho proxy FAIL (−0,012…+0,006): full khác pilot (học thuộc receiver chỉ lộ ở
  4.200 bước/epoch). Benign tăng mạnh (+0,11) — khối {benign, …} là nút thắt cũ.
- FedNTD hại private (−0,042) và client tệ nhất (0,400): bỏ lớp đúng khỏi KL làm mất tín hiệu thầy ở lớp đúng;
  lệch prior không phải nguyên nhân chính của khoảng private–proxy.
- q4 không thua (private −0,004, proxy +0,002), byte nhỏ 7,65× ⇒ R1 dùng được.
- Một seed, ep6/50: sàng lọc. Cần horizon (ep15) và seed 43 trước khi đổi headline.

## Vòng 2 — horizon ep15 (audited 16/10 ~16:40Z)

Sessions `anhtruong1909/…-fd-s42{d,e}-20261015-r2-v1` COMPLETE, AUDITED như vòng 3 (`fd_dml@e15`, `fd_fedavg@e15`).
Replay ep1–6 của các arm `@e15` trùng vòng 1 tuyệt đối (Δ 0,0000) ⇒ đối chứng tái dùng hợp lệ.

Quỹ đạo ep6→ep15 (group): `fd_dml` proxy 0,763→0,765 (phẳng từ ep4), private gp 0,666→0,696, **benign proxy 0,741 (ep3)
→0,358**. Client view ep15 private gp: local acc 0,788, client tệ nhất local acc 0,432.

Đọc: kéo dài DML không mask không có tiềm năng ở proxy (phẳng, benign giảm đều theo epoch); private gp còn tăng chậm
(+0,03 trong 9 epoch). Đối chứng horizon cho m4b (vòng 5, 9).

## Vòng 4 — hidden KD và Gaussian m4b (audited 16/10 ~16:40Z)

Session `odixe0502/mobihet-fd-s42i-20261015-r4-v1` COMPLETE, AUDITED. Cổng private gp group ep6 ([design §Vòng 4](design.md)).

| arm | private gp | Δ | benign Δ | proxy head | local F1 / acc private gp | client tệ nhất | test primary | cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fd_dml (đối chứng) | 0,6655 | — | — | 0,7631 | 0,637 / 0,794 | 0,595 | 0,7715 (proxy) | — |
| fd_dml_hkd | 0,6710 | +0,006 | +0,001 | 0,7661 | 0,643 / 0,797 | 0,588 | 0,7745 (proxy) | FAIL |
| fd_gaussian (đối chứng) | 0,5882 | — | — | — | 0,614 / 0,790 | 0,458 | 0,6074 (private gp) | — |
| **fd_gaussian_m4b** | **0,6553** | **+0,067** | **+0,142** | — | 0,679 / 0,866 | 0,490 | 0,6671 | **PASS** |

Đọc:
- **Hidden-state KD (FedKD) gần như không có tác dụng** (+0,006 < +0,010): thêm kênh 256-d không đóng khoảng private–proxy.
  Cùng với NTD (−0,042) ⇒ khoảng private–proxy ở full không do kênh KD hẹp hay lệch prior, mà do private học shortcut
  receiver trên dữ liệu local (m4b đóng +0,036 với cùng kênh KD). Không nối/quét trọng số HKD.
- **m4b cho Gaussian lớn nhất từ trước tới nay** (+0,067, pilot +0,025; swappos full +0,052); vẫn dưới DML m4b (0,7010).

## Vòng 5 — m4b horizon + ghép (audited 16/10 ~17:00Z)

Session `nguyenminhtran060113/mobihet-fd-s42j-20261015-r5-v1` COMPLETE, AUDITED; replay ep1–6 của `fd_dml_m4b@e15` trùng
vòng 3 tuyệt đối. Cổng [design §Vòng 5](design.md). Group; test chỉ epoch cuối, mô tả.

| so sánh | private gp (Δ, benign Δ) | proxy head (Δ, benign Δ) | test private gp / proxy | cổng |
|---|---|---|---|---|
| **`fd_dml_m4b@e15` vs `fd_dml@e15`** (ep15) | **0,7323 (+0,036, +0,144)** | **0,8510 (+0,086, +0,409)** | 0,7478 / 0,8322 (đối chứng 0,7164 / 0,7769) | **PASS** horizon |
| `fd_dml_m4b_q4` vs `fd_dml_m4b` | 0,7010 (+0,000, −0,000) | 0,8157 (+0,001, +0,010) | 0,7234 / 0,8125 | PASS noninferior |

Quỹ đạo `fd_dml_m4b` ep6→ep15 (group): proxy 0,815→0,851 (còn tăng, `fd_dml` phẳng 0,763), private gp 0,701→0,732,
**benign proxy giữ 0,76–0,78** (`fd_dml` rơi 0,74→0,36). Client view ep15 private gp: local F1 0,708, local acc 0,867,
client tệ nhất local acc 0,756 (`fd_dml@e15` 0,432).

Đọc:
- **Học thuộc receiver là nguyên nhân proxy DML chững và benign sụp theo epoch ở full**: bỏ 13 feature receiver/vị trí thì
  proxy tiếp tục học tới ep15 và benign không rơi. Lợi m4b lớn dần theo horizon (proxy +0,052 ep6 → +0,086 ep15).
- So mô tả với full headline cũ (test, khác holdout/round): m4b ep15 private gp 0,748 = `h_dml` r41 0,748; proxy 0,832 > 0,801.
- Hai kỹ thuật cộng được: m4b (0 byte) + q4 (byte 7,65× nhỏ, không mất). Còn chờ seed 43 (vòng 6).

## Vòng 6 — seed 43 xác nhận m4b (audited 17/10 ~21:05Z)

Session `anhtruong1909/mobihet-fd-s43k-20261015-r6-v1` COMPLETE, AUDITED (marker chain, ep6 weights, predictions →
confusion → 10 metric). Cổng [design §Vòng 6](design.md), group ep6, seed 43 so đối chứng cùng seed.

| so sánh (s43) | private gp (Δ, benign Δ) | proxy head (Δ, benign Δ) | client tệ nhất global F1 private gp | test proxy | cổng |
|---|---|---|---|---|---|
| **`fd_dml_m4b` vs `fd_dml`** | **0,6940 (+0,031, +0,096)** | **0,7930 (+0,039, +0,199)** | 0,6046 (đối chứng 0,5774) | 0,7997 (0,7581) | **PASS** cả hai |

Đọc:
- **m4b lặp lại ở seed 43** (private +0,031 so s42 +0,036; proxy +0,039 so +0,052), benign proxy +0,20. Quy tắc headline
  [design §Vòng 6](design.md) thỏa (m4b@e15 PASS s42 + m4b vs dml s43 PASS private và proxy) ⇒ **headline DML chuyển sang m4b**.
- EMA private hại ở mọi arm s43 (−0,021…−0,047), như s42.

## Vòng 7 — seed 43 swappos + Gaussian m4b (audited 17/10 ~21:43Z)

Session `annhannh331/mobihet-fd-s43m-20261015-r7-v1` COMPLETE, AUDITED. Cổng [design §Vòng 7](design.md).

| arm (s43, ep6) | private gp (Δ, benign Δ) | proxy head (Δ, benign Δ) | client tệ nhất global F1 | private_ema gp | cổng |
|---|---|---|---|---|---|
| **`fd_dml_swappos` vs `fd_dml`** | **0,6970 (+0,034, +0,094)** | **0,7846 (+0,031, +0,200)** | 0,6174 (0,5774) | 0,6594 | **PASS** |
| `fd_gaussian_m4b` | 0,6568 (s42: 0,6553) | — | 0,5258 | 0,6355 (−0,021) | mô tả (đối chứng `fd_gaussian` s43 ở vòng 8) |

Đọc: swappos lặp lại ở seed 43 (+0,034 so s42 +0,033). Gaussian m4b gần như trùng s42; **EMA hại Gaussian-m4b
ở cả hai seed** (−0,010 s42, −0,021 s43) ⇒ nhánh EMA §4.3 của plan dừng. Cổng cặp Gaussian s43 chờ vòng 8 `r`.

## Vòng 8 `r` — đối chứng Gaussian seed 43 (audited 17/10 ~00:05Z)

Session `takids1909/mobihet-fd-s43r-20261015-r8-v1` COMPLETE, AUDITED (2/2 arm). Cổng `gain` [design §Vòng 8](design.md).

| so sánh (s43, ep6) | private gp (Δ, benign Δ) | client tệ nhất global F1 (đối chứng 0,5037) | private_ema gp (Δ so private) | cổng |
|---|---|---|---|---|
| **`fd_gaussian_m4b` vs `fd_gaussian`** | **0,6568 (+0,053, +0,119)** | 0,5258 | 0,6355 (−0,021) | **PASS** |
| **`fd_gaussian_swappos` vs `fd_gaussian`** | **0,6527 (+0,049, +0,121)** | 0,5103 | 0,6835 (+0,031) | **PASS** |

Đọc: bỏ/xáo đặc trưng receiver-position lặp lại cho Gaussian ở seed 43 (đối chứng `fd_gaussian` s43 0,6039; s42 PASS ở
vòng 3–4). EMA private giúp `fd_gaussian` (+0,026) và `swappos` (+0,031) nhưng hại `m4b` (−0,021) — cùng chiều cả hai seed.

## Vòng 8–10 — cập nhật client/trao đổi tri thức, s42 + s43 (audited 17/10 ~04:40Z)

15 session COMPLETE, AUDITED (n o p q s t u v y z ka kb ga gb + r ở trên); `w` (horizon ep15 s43) AUDITED 17/10 ~08:20Z (bảng dưới).
Cổng `gain+worst` [design §Vòng 8–10](design.md): private gp Δ ≥ +0,010, benign ≥ −0,005, client tệ nhất (global
macro-F1 private gp) ≥ −0,010. Group holdout ep6. Proxy head chỉ báo kèm. Máy local ngủ ~00:05–04:25Z: kernel Kaggle
không ảnh hưởng; collect sau khi thức (4 tiến trình, pull ~20 s/session, audit tuần tự).

| so sánh | s42: private gp (Δ, benign, worst; proxy Δ) | s43: private gp (Δ, benign, worst; proxy Δ) | cổng |
|---|---|---|---|
| **`gpool` vs `fd_dml_m4b`** | **0,7174 (+0,016, +0,003, −0,003; −0,003)** | **0,7175 (+0,024, −0,000, +0,018; +0,001)** | **PASS ×2** |
| **`gpool` vs `gown`** | +0,030, +0,004, +0,022 | +0,030, +0,001, +0,013 | **PASS ×2** |
| **`gpool` vs `replay`** | +0,033, +0,029, +0,012 | +0,032, +0,021, +0,021 | **PASS ×2** |
| `fd_dml_m4b` vs `_own` (lợi FL) | +0,043, +0,030, +0,138 | +0,040, +0,030, +0,132 | PASS ×2 |
| `fd_gaussian_m4b` vs `_own` | +0,051, +0,008, +0,044 | +0,052, +0,007, +0,091 | PASS ×2 |
| `kdcap` vs `fd_dml_m4b` | 0,6735 (−0,028, −0,027, −0,061; **+0,024**) | 0,6687 (−0,025, −0,023, −0,052; **+0,033**) | FAIL ×2 |
| `kdcap_frz` vs `kdcap_wu` | 0,5871 (−0,087, −0,123, −0,259) | 0,5765 (−0,088, −0,098, −0,188) | FAIL ×2 |
| `dyn` vs `fd_dml_m4b` | 0,6688 (−0,032, −0,020, −0,183; −0,044) | −0,034, −0,008, −0,145; −0,037 | FAIL ×2 |
| `cons` vs `cons0` (JS) | 0,7001 (+0,003, −0,003, +0,006) | — | FAIL |
| `scaf10` vs `sgd10` (LR* 0,1) | 0,6941 (−0,010, −0,039, −0,125) | — | FAIL |
| `scaf10` vs `fd_dml_m4b` | −0,007, −0,049, −0,149 | — | FAIL |
| `fd_dml_m4b_q4` vs `fd_dml_m4b` (noninferior) | (vòng 5 PASS) | −0,003, −0,002 | PASS |
| **`fd_dml_m4b@e15` vs `fd_dml@e15`** (horizon ep15, `w`) | (vòng 5: +0,036; proxy +0,086) | **0,7263 (+0,025; worst 0,634 vs 0,623; proxy 0,8356, +0,063; benign proxy 0,782 vs 0,484)** | **PASS ×2** |

`w` test (mô tả, ep15 s43): private gp 0,7446 vs 0,7194, proxy 0,8240 vs 0,7768. Quỹ đạo s43 giống s42: `fd_dml` proxy phẳng
~0,76 từ ep4 và benign proxy rơi 0,74→0,48; m4b proxy 0,793 (ep6) → 0,836 (ep15), benign giữ 0,77–0,78.

Mô tả (không cổng): LR SCAFFOLD chọn theo old-holdout proxy của `sgd*`: 0,7134 / 0,7585 / **0,8011** (0,01/0,03/0,1).
`sgd{1,3,10}` vs m4b −0,065/−0,028/+0,003; `scaf{1,3}` −0,128/−0,058. `wu` vs `kdcap` +0,000 (s42) / −0,004 (s43).
`cons` vs m4b −0,001. Rerun `fd_dml_m4b` (y, source r10) = vòng 3 **bit-exact** (Δ 0,0000).
Gaussian tổng hợp `gpool` thêm 2,78 MiB lên (moment, một lần) + 1,39 MiB xuống trên ~395 MiB/6 epoch (+1,1 %).

Chẩn đoán KD (rerun m4b, trung bình client): `w_raw = 1/(CE_priv+CE_proxy)` median 2,39 (ep1) → 4,14 (ep6), cap_hit
0,955 → 1,0; probe ‖w∇KL‖/‖∇CE‖ 0,79 → 1,23. `kdcap` đưa tỷ số về 0,23–0,27.

Đọc:
- **Hybrid private-only (`gpool`) là cải tiến client đầu tiên vượt m4b ở cả hai seed** (+0,016 / +0,024), và hơn cả
  moment riêng (+0,030) lẫn replay local cùng số forward (+0,032) ⇒ lợi đến từ **tri thức lớp gộp từ client khác**, không
  phải từ thêm CE/cân bằng lớp. Proxy không đổi (±0,003). Client tệ nhất không giảm.
- Lợi FL thật: DML-m4b hơn proxy không gộp +0,040–0,043 (client tệ nhất +0,13); Gaussian-m4b hơn moment riêng +0,05.
- **Giả thuyết "trọng số KD quá lớn" bị bác cho private**: cap chạm ~100 % bước (giả thuyết được kích hoạt), nhưng giảm KD
  làm private gp −0,025…−0,028 ở cả hai seed; ngược lại proxy head **+0,024 / +0,033**. KD mạnh có lợi cho private,
  nhưng KL proxy←private mạnh hại proxy (mô tả; proxy không phải primary).
- Frozen teacher: private **head** 0,7135 (s42) / 0,6978 (s43) cao hơn head m4b (0,586 / 0,579) và cả gp m4b
  (0,7010 / 0,6940), nhưng gp sập (0,587 / 0,577): teacher gộp đã mang prior toàn cục, luật gp cộng thêm
  `log π_g − log π_k` ⇒ sửa hai lần. FAIL theo luật khóa; head chỉ mô tả, cần vòng mới với rule khóa trước.
- FedDyn và SCAFFOLD (LR tốt nhất) giảm drift/không giúp, client tệ nhất rơi −0,13…−0,18 ⇒ dừng. JS consistency không thêm
  gì trên hai view (+0,003).

## Vòng 11 `la` — trọng số KD trên headline `gpool` (audited 17/10 ~14:00Z)

Session `accanniez/mobihet-fd-s42la-20261015-r11-v1` COMPLETE (13:51Z), AUDITED 2/2 arm. Đối chứng `fd_dml_m4b_gpool` s42
(vòng 9, private gp 0,7174, proxy head 0,8120). Cổng gain+worst [design §Vòng 11](design.md). Group ep6; test mô tả.

| arm | private gp (Δ, benign Δ) | client tệ nhất private gp (gpool 0,6137) | proxy head (Δ) | test proxy | cổng |
|---|---|---:|---:|---:|---|
| `gpool_kd2` (KD private ×2) | 0,7187 (+0,001, +0,004) | 0,6280 | 0,8027 (−0,009) | 0,8043 | FAIL |
| `gpool_kdpx` (cap chỉ KL proxy←private) | 0,7090 (−0,008, −0,019) | 0,6042 | **0,8434 (+0,031)** | 0,8271 | FAIL |

Đọc:
- **Tăng KD phía private không thêm gì** (+0,001): private đã hấp thụ hết những gì teacher nói trên dòng thật local.
  Khớp phân tích khoảng private–proxy (gpool ep6): 56 % khoảng F1 nằm ở lớp chiếm < 1 % dữ liệu local, nơi KD trên dòng
  local gần như không có mẫu.
- **Teacher tốt hơn không tự sang private**: cap KL proxy←private làm proxy +0,031 (lặp `kdcap` +0,024/+0,033) nhưng
  private −0,008. Hướng chỉnh trọng số KD đã cạn (kdcap, kd2, kdpx).

## Vòng 11 `lc` — horizon gpool ep15 + gw2 (audited 17/10 ~19:05Z)

Session `takids1909/mobihet-fd-s42lc-20261015-r11-v1` COMPLETE 18:59Z, AUDITED 2/2 arm. gpool:15 ep1–6 trùng run gpool ep6
(Δ 0,0). Cổng [design §Vòng 11](design.md).

| epoch | 1 | 3 | 6 | 9 | 12 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| `fd_dml_m4b` private gp | 0,6117 | 0,6651 | 0,7010 | 0,7167 | 0,7234 | 0,7323 |
| `fd_dml_m4b_gpool` private gp | 0,6654 | 0,6962 | 0,7174 | 0,7244 | 0,7325 | 0,7342 |
| Δ gpool − m4b | +0,054 | +0,031 | +0,016 | +0,008 | +0,009 | **+0,002** |
| proxy head m4b / gpool | 0,305 / 0,391 | 0,770 / 0,772 | 0,815 / 0,812 | 0,837 / 0,826 | 0,844 / 0,838 | 0,851 / 0,844 |

| so sánh | Δ private gp (benign Δ) | khác | cổng |
|---|---|---|---|
| `gpool@e15` vs `fd_dml_m4b@e15` (horizon) | +0,0019 (−0,001) | test ep15 private gp 0,7486 vs 0,7478; proxy test 0,8299 vs 0,8322 | **FAIL** (cần +0,010) |
| `gpool_gw2` vs gpool (ep6) | −0,0049 (+0,001) | client tệ nhất −0,0104; proxy −0,008 | **FAIL** |

Đọc:
- **gpool tăng tốc đầu, không tăng mức cuối**: lợi so m4b +0,054 → +0,016 (ep6) → +0,002 (ep15); m4b đuổi kịp khi proxy
  mạnh dần và KD mang tri thức quần thể vào private. Proxy của gpool thấp hơn m4b từ ep9 (−0,007…−0,011). Theo cổng khóa,
  gpool không qua horizon ⇒ cổng ep6 đo **tốc độ học**, chưa đo mức hội tụ; arm vòng 13 PASS ở ep6 phải lặp horizon ep15
  trước khi nhận. Ý nghĩa R1: gpool đạt 0,717 ở ep6, m4b cần ~ep9 (ít hơn ~3 lần trao đổi proxy cho cùng mức F1).
- Tăng liều Gaussian (gw2) không giúp, còn kéo client tệ nhất và proxy xuống.

## Vòng 13 probe — không train (17/10 ~17:05Z)

Kernel `nguyenminhtran060113/mobihet-fd-probe13-20261015-r13-v1` COMPLETE 17:03Z. Source đã chạy == manifest đã stage (48/48),
weights SHA khớp; luật gp tái hiện đúng số audit (0,7174 / 0,7175 / 0,7187 / 0,7091). Artifact:
`methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-probe13-20261015-r13-v1/runs/probe13.json`. Cổng khóa
[design §Vòng 13](design.md). Group ep6, private, mean 20 client.

**C — luật `gpe` (prior huấn luyện (π_k + q_k)/2) so gp, cùng weights:**

| weights | gp (benign, tệ nhất) | gpe (Δ, benign Δ, tệ nhất Δ) | head | cổng |
|---|---|---|---:|---|
| gpool s42 | 0,7174 (0,502, 0,614) | 0,6232 (−0,094, −0,035, −0,139) | 0,5999 | FAIL |
| gpool s43 | 0,7175 (0,499, 0,623) | 0,6223 (−0,095, −0,037, −0,157) | 0,6007 | FAIL |
| kd2 s42 | 0,7187 | 0,6187 (−0,100) | 0,5964 | mô tả |
| kdpx s42 | 0,7091 | 0,6295 (−0,080) | 0,6083 | mô tả |

**P3 — BN tính lại trên 65.536 dòng thật của client:** gp 0,6859 (−0,032) s42, 0,6741 (−0,043) s43; benign −0,027/−0,028
⇒ không có tín hiệu (cần ≥ +0,005 cả hai seed) → backlog.

**P1 coverage** (proxy gpool s42 ep6 trên 16 × 4.000 mẫu từ moment gộp full): nhận benign 0,207, dataReplay 0,488,
positionMirroring 0,022, timeDelay 0,000, suddenConstantSpeed 0,166; lớp còn lại 0,48–0,99. Gate {benign, dataReplay} phủ
**4,3 %** mẫu Gaussian (~11/256 mỗi bước).

**P2 mode urban/highway** (4 lớp khó; Mahalanobis giữa hai mode 5,2–5,5): proxy nhận mẫu 1 Gaussian → hỗn hợp: benign
0,205 → 0,212, dataReplay 0,493 → 0,492, positionMirroring 0,023 → 0,030, timeDelay 0 → 0. Không lớp nào ≥ +0,05 ⇒ **không mở P2a**.

**P4 ridge teacher-only** (σ 8,93; λ chọn trên old, lưới gần phẳng, chọn 1e-4·tr/d):

| teacher | group macro-F1 | benign | own (mean / tệ nhất) | byte lên / xuống |
|---|---:|---:|---|---|
| linear ridge (x53) | 0,264 | 0,568 | 0,193 / 0,125 | 9.396 / 3.456 |
| RFF M128 | 0,308 | 0,585 | 0,201 / 0,124 | 41.796 / 8.256 |
| **RFF M256** | **0,378** | 0,599 | 0,249 / 0,171 | 149.060 / 16.448 |
| RFF M1024 (mô tả) | 0,484 | 0,636 | 0,310 / 0,197 | 2.168.900 / 65.600 |

Pooled − own +0,129, M256 − linear +0,114 nhưng M256 **0,378 ≪ 0,7174** (private gp gpool) ⇒ **FAIL teacher gate**, không
train private P4. Lặp lại tiền lệ Build13 S13.

Đọc:
- **Private đã học prior gần π_k local, không phải prior trộn của hai số hạng CE.** gpe bớt sửa lớp hiếm ⇒ −0,094 cả hai seed,
  tệ nhất −0,14/−0,16. 256 mẫu Gaussian (dễ tách, nằm vùng khác dòng thật) không đổi prior của head trên dòng thật — đúng cảnh
  báo Q(x|y) ≠ p_k(x|y). gp giữ nguyên làm luật chính.
- **BN của private gpool là một phần hàm đã học trên batch trộn**; thay bằng thống kê chỉ dòng thật làm giảm 0,03–0,04.
  Probe không ủng hộ P3 dạng "BN real-only"; arm tách BN khi train (giữ hai miền) chưa được kiểm, để backlog.
- **P1 như khóa gần như không chạm private** (gate 4,3 % mẫu, λ 0,25, chia m): dự báo mc ≈ gpool; kết quả vẫn chấm theo cổng.
- Hỗn hợp theo scenario không làm mẫu "giống lớp" hơn trong mắt proxy; khó ở positionMirroring/timeDelay không do gộp mode.
- Teacher tuyến tính trên đặc trưng ngẫu nhiên của input kém xa DAGSNet (0,38 vs 0,72 ở M256, 0,48 ở M1024) ⇒ hướng thay
  proxy bằng thống kê ridge không đủ sức chứa ở ngân sách byte B2.

## Vòng 13 — train (s42 ep6 từ đầu, so gpool 0,7174; cập nhật dần)

Cổng [design §Vòng 13](design.md). Mỗi session một arm, `--workers 2`; AUDITED theo collector (source/fingerprint/marker/
predictions → 10 metric). Client tệ nhất gpool 0,6137.

| arm | session | private gp (Δ, benign Δ, tệ nhất) | proxy head (Δ) | test private gp | cổng |
|---|---|---|---:|---:|---|
| `gpool_rej` (B1) | mb | **0,7282 (+0,0108, +0,0098, 0,626 / +0,012)** | 0,804 (−0,008) | — | **PASS** (sát ngưỡng) |
| `gpool_q2` (C2) | mf | 0,7175 (+0,0001, +0,003, 0,621) | **0,688 (−0,124)** | 0,7343 | noninferior PASS (private); proxy sụt |
| `gpool_p1self` (control) | md | 0,7187 (+0,0014, +0,000, 0,623) | +0,004 | — | mô tả (≈ gpool) |
| `gpool_p1cm` (control) | me | 0,7172 (−0,0002, −0,001, 0,620) | +0,002 | — | mô tả (≈ gpool) |
| `gpool_p1` (P1) | mc | 0,7175 (+0,0001, +0,004, —) | −0,001 | 0,7348 | **FAIL** (≈ gpool; cổng cũng thua p1self/p1cm) |
| `gpool_fill` (A) | ma (v2) | 0,7162 (−0,0012, +0,002, tệ nhất −0,001) | 0,808 (−0,004) | 0,7335 | **FAIL** (≈ gpool) |

- rej theo epoch so gpool: r2 −0,007, r3 +0,011, r4 +0,011, r5 +0,015, r6 +0,011; 14/16 lớp tăng (constantSpeedOffset +0,050,
  suddenStop +0,017, randomSpeedOffset +0,019, benign +0,010, dataReplay +0,013); timeDelay −0,003 dù proxy nhận 0 % mẫu timeDelay
  (lớp đó mất hết mẫu Gaussian). Tỷ lệ nhận ~0,51 từ r4 (benign 0,17, positionMirroring 0,03). ⇒ lặp seed 43 + horizon ep15 (vòng 14).
- q2: 109.512 B/chiều (219.024 B/round, dưới B2 231.000 B), sai số decode tương đối 0,25–0,37/round (q4 ~0,02–0,05). Private không đổi
  ở ep6 nhưng proxy mất 0,124 ⇒ cần horizon (lc cho thấy private về sau tăng nhờ proxy mạnh dần).
- Luật gpe thua gp ở mọi arm (−0,09…−0,095), lặp kết quả C.
- P1 (KL proxy trên mẫu Gaussian lớp 0/4 được proxy nhận) không thêm gì: gate chỉ phủ 4,3 % mẫu (probe) ⇒ đúng dự báo.

## Vòng 12 `qa` — headline qua int4 thật (audited 17/10 ~20:25Z)

`gpool_q4` vs `gpool` ep6 s42 (noninferior): private gp 0,7173 (−0,0001, benign +0,002), proxy head 0,8158 (+0,004), test private
0,7341. **PASS noninferior** ⇒ headline DML-m4b + gpool qua QPacket int4 thật (208.268 B/chiều) không mất utility ở ep6.

## Chẩn đoán private chững ~0,73–0,75 (17/10 ~20:30Z; probe 14–15 đang chạy)

**Phân tích local trên confusion ep15 đã audit** (m4b s42 / gpool s42 / m4b s43; private gp vs proxy head, group):

| tỷ lệ lớp trong dữ liệu local | ô (client × lớp) | phần khoảng proxy − private | khoảng TB / ô |
|---|---:|---:|---:|
| < 0,01 % | 13 | 7,5–7,9 % | +0,21–0,22 |
| 0,01–0,1 % | 28 | 14,9–15,8 % | +0,19–0,21 |
| 0,1–1 % | 78 | 30,7–32,6 % | +0,14–0,15 |
| 1–10 % | 143 | 28,0–29,3 % | +0,07–0,08 |
| ≥ 10 % | 58 | 16,4–18,5 % | **+0,10–0,11** |

- Khoảng 0,109–0,119 **không** chỉ do lớp hiếm local: ~46 % nằm ở lớp chiếm ≥ 1 % dữ liệu local; lớp hụt nhất là **benign** (+0,25–0,27,
  chỉ 1 client có < 0,1 % benign), rồi dataReplay, timeDelay, suddenConstantSpeed, positionMirroring.
- Hướng lỗi (m4b s42): private **gán nhãn tấn công cho dòng benign** — benign recall 0,50 vs proxy 0,82 (benign→timeDelay 3.389 vs
  1.505, →dataReplay 1.846 vs 1.009, →positionMirroring 1.108 vs ~0). Precision thấp ở nhiều lớp: suddenStop 0,68 vs 0,999,
  suddenConstantSpeed 0,54 vs 0,97, dataReplay 0,34 vs 0,63 (trafficCongestionSybil→dataReplay 633). Group = receiver chưa thấy.
- Tương quan theo client: private F1 với số dòng local +0,50…+0,55; với entropy prior −0,13…−0,18.
- Giả thuyết đang kiểm: vùng quyết định private chỉ được neo trên giá đỡ dữ liệu local; dòng ngoài giá đỡ (receiver/vùng khác) bị
  gán lớp học được tại chỗ ⇒ false positive. Kiểm: probe 14 T1/T5, probe 15 `cov_real`/`pinit*` ([design](design.md)).

### Probe 14 (AUDITED 17/10 ~20:40Z: gp tái hiện audit ±0,002 cả 3 bộ; artifact `artifacts/kaggle/fd-probe14-20261015-p14-v1/`)

Trung bình 20 client, group (refit/bias = phép đo oracle, không là phương pháp):

| bộ weights ep15 | private gp | bias oracle | refit Linear own | **refit Linear gộp** | refit MLP gộp | proxy | proxy refit Linear gộp / bias oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| m4b s42 | 0,7323 | 0,7777 | 0,6864 | **0,8427** | 0,8754 | 0,8509 | 0,9020 / 0,8843 |
| gpool s42 | 0,7342 | 0,7792 | 0,6866 | **0,8413** | 0,8684 | 0,8435 | 0,8966 / 0,8815 |
| m4b s43 | 0,7263 | 0,7749 | 0,6809 | **0,8416** | 0,8741 | 0,8357 | 0,8999 / 0,8755 |

- **Backbone private đủ; Linear cuối (4.112 tham số) là nút thắt.** Refit Linear trên mẫu gộp: mọi client +0,05…+0,19 (tệ nhất 0,63 → 0,82).
  Refit own (cùng thủ tục) 0,68 < gp ⇒ lợi đến từ dữ liệu phân phối toàn cục, không từ thủ tục. Bias oracle chỉ +0,045 ⇒ không phải
  chỉ prior: hướng/vị trí siêu phẳng lệch theo dữ liệu local.
- Proxy cũng lệch ở head: refit gộp +0,05, bias oracle +0,03…+0,04.
- T1: private đồng ý proxy 0,885 trên dòng của mình, 0,773 trên dòng client khác, 0,752 trên group; KL 0,98 / 1,65 / 1,91.
- T5: 0,37 % dòng train trùng hệt khác nhãn (benign 74.331 ↔ timeDelay 66.238, dos 8.312), 95 % chỉ thấy khi gộp client; group không
  trùng train. kNN (Q 16.384 group): gộp 100k 0,557, gộp 1M 0,712, **own 100k 0,30** (0,20–0,39) ⇒ cùng số dòng, đa dạng client
  +0,25.

### Probe 16 (AUDITED 17/10 ~21:35Z: gp tái hiện audit cả 6 bộ; artifact `artifacts/kaggle/fd-probe16-20261015-p16-v1/`)

**A — backbone ep6 theo cách train** (refit head trên mẫu gộp = oracle):

| backbone ep6 s42 | gp gốc | Linear gộp | Linear own | MLP gộp |
|---|---:|---:|---:|---:|
| DML-m4b | 0,7010 | 0,8269 | 0,6483 | 0,8680 |
| DML-m4b không gộp (own) | 0,6579 | 0,8155 | 0,6275 | 0,8625 |
| Gaussian-m4b (không proxy) | 0,6553 | 0,8254 | 0,6361 | 0,8678 |
| Gaussian-m4b own (không FL) | 0,6048 | 0,8125 | 0,6058 | 0,8647 |

⇒ Backbone học chỉ từ dữ liệu local đã đủ (khác nhau ≤ 0,015 Linear, ≤ 0,005 MLP). Lợi FL gần như toàn bộ nằm ở tầng quyết định.

**B — dữ liệu fit Linear cuối của private ep15** (trung bình 20 client; prior-corrected π_g):

| private ep15 | gp | own | +Gaussian nhãn cứng | +Gaussian proxy nhận | +Gaussian nhãn proxy | +dòng thật client khác nhãn proxy (oracle) | gộp thật | bias trên Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m4b s42 | 0,7323 | 0,6855 | 0,7154 | 0,7632 | **0,7669** | **0,8306** | 0,8427 | 0,7124 |
| gpool s42 | 0,7342 | 0,6847 | 0,6340 | 0,6549 | 0,6641 | 0,8298 | 0,8413 | 0,6106 |

- Proxy gán nhãn gần bằng nhãn thật (0,831 vs 0,843) ⇒ thiếu **input thực tế** phủ phân phối client khác, không thiếu người gán nhãn.
- Gaussian gộp + nhãn proxy lấy ~1/3 khoảng (+0,035 so gp) trên backbone m4b; trên backbone gpool lại hại (backbone đã học mẫu Gaussian
  thành vùng riêng). Bias fit trên Gaussian không dùng được.
- C (CCVR proxy) lỗi thứ tự nhãn virtual (0,04) → sửa, chạy lại ở probe 17.

## Vòng 14 — lặp arm PASS vòng 13 (cập nhật dần)

| arm | session | private gp (Δ, benign Δ, tệ nhất Δ) | proxy (Δ) | test private | cổng |
|---|---|---|---:|---:|---|
| `gpool_rej` s43 ep6 | nb (audited 23:10Z) | 0,7257 (+0,0082, +0,005, +0,018) vs gpool s43 0,7175 | 0,787 (−0,007) | 0,7415 | **FAIL** gain (< +0,010); cùng chiều s42 (+0,0108), TB hai seed +0,0095 |
| `gpool_rej:15` s42 ep15 | na (audited 01:10Z) | **0,7455** (+0,0132 vs m4b@e15; +0,0113 vs gpool@e15; benign +0,008; tệ nhất 0,625 vs 0,629 / 0,633) | 0,846 (−0,005 / +0,002) | **0,7599** | **horizon PASS**; ep1–6 bit-exact = mb |
| `gpool_q2:15` s42 ep15 | nc (audited 01:15Z) | 0,7334 (−0,0008 vs gpool@e15, benign −0,001) | **0,725 (−0,118)** | 0,7499 | **noninferior PASS** (private); ep1–6 bit-exact = mf |

### Probe 15 (17/10 ~23:25Z; 1 epoch có kiểm soát từ m4b ep15 s42, 20 client; artifact `artifacts/kaggle/fd-probe15-20261015-p15-v1/`)

Private trên group, trung bình 20 client (gp cho DML/CE; head cho teacher cố định vì teacher mang prior toàn cục):

| biến thể | bước 0 | 64 | 256 | 1024 | 2048 | cuối | tệ nhất cuối |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cont` DML như train (gp) | 0,7323 | **0,7580** | 0,7527 | 0,7435 | 0,7400 | **0,7267** | 0,646 |
| `kdfix` teacher gộp cố định (head) | 0,6088 | 0,7513 | 0,7637 | 0,7683 | 0,7742 | **0,7697** | 0,676 |
| `cov_gauss` + KD trên Gaussian gộp (head) | 0,6088 | 0,7744 | 0,7808 | 0,7885 | 0,7917 | **0,7921** | 0,755 |
| `cov_real` + KD trên dòng thật client khác, oracle (head) | 0,6088 | 0,8092 | 0,8234 | 0,8349 | 0,8391 | **0,8422** | 0,820 |
| `pinit` khởi tạo = proxy, DML (gp) | 0,7121 | 0,8142 | 0,8104 | 0,7955 | 0,7861 | 0,7690 | 0,696 |
| `pinit_ce` khởi tạo = proxy, chỉ CE (gp) | 0,7121 | 0,8172 | 0,8050 | 0,7729 | 0,7594 | 0,7366 | 0,657 |
| `pinit_kdfix` (head) | 0,8509 | 0,8392 | 0,8342 | 0,8293 | 0,8263 | 0,8190 | 0,714 |
| `pinit_kdonly` (head) | 0,8509 | 0,8428 | 0,8392 | 0,8385 | 0,8368 | 0,8308 | 0,724 |

- **Quên do mục tiêu local:** private có tri thức mức proxy vẫn rơi 0,08 trong một epoch chỉ với CE local (`pinit_ce`); DML chậm hơn
  (0,769) vì teacher là bản proxy cũng trôi local. Ngay cả `cont` cũng đạt đỉnh ở bước 64 (0,758) rồi trôi về 0,727.
- **Teacher gộp cố định giữ ranh giới toàn cục:** `kdfix` head 0,770 (+0,043 so `cont`), `pinit_kdfix` giữ 0,819.
- **Phủ input là phần còn lại:** dòng thật client khác (không nhãn) + KD proxy cố định → 0,842 ≈ proxy 0,851; Gaussian gộp → 0,792.

### Probe 17 (17/10 ~23:45Z; artifact `artifacts/kaggle/fd-probe17-20261015-p17-v1/`)

Fit Linear cuối private trên own + mẫu tổng hợp nhãn proxy (trung bình 20 client):

| private | gp | Gaussian (cứng) | ce (mềm) | di_lo (mềm) | **di_hi (mềm)** | dòng thật client khác (oracle) | gộp thật |
|---|---:|---:|---:|---:|---:|---:|---:|
| m4b ep15 s42 | 0,7323 | 0,7625 | 0,7682 | 0,7728 | **0,7832** | 0,8306 | 0,8427 |
| m4b ep15 s43 | 0,7263 | 0,7647 | 0,7603 | 0,7667 | **0,7778** | 0,8206 | 0,8416 |
| m4b ep6 s42 | 0,7010 | 0,7445 | 0,7450 | 0,7492 | **0,7566** | 0,8072 | 0,8269 |

- Tinh chỉnh bằng proxy (CE + khớp BN) làm mẫu gần dữ liệu thật hơn (1-NN trung vị 5,07 → 3,9–4,4; dòng thật 2,05; nhãn 1-NN khớp
  0,43 → 0,65–0,69), utility +0,012…+0,021 so Gaussian — dưới ngưỡng +0,02 hai seed của hợp đồng ⇒ không đủ là kênh chính; ~30 % khoảng
  tới oracle.
- **CCVR trên proxy** (moment lớp gộp của đặc trưng 256-d proxy, full covariance): head proxy 0,896 / 0,892 vs 0,851 / 0,836
  (+0,045 / +0,056); diag 0,834 / 0,832 (thua proxy). Byte full ~2,1 MB/client ⇒ vượt R1; dạng nhỏ (LDA chung covariance ~148 kB) chưa đo.

- q2 tới ep15: private không đổi dù proxy yếu hơn 0,118 ⇒ khớp chẩn đoán probe 15: mức private do cân bằng trôi local quyết định, không do
  chất lượng teacher. Byte proxy 219.024 B/round (q4 416.536 B) với private noninferior ⇒ đòn bẩy R1 nếu target là private.

## Vòng 15 — teacher snapshot gộp cố định (s42 ep1→15; arm frz đọc head, control gp; cập nhật dần)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (benign, tệ nhất) | proxy ep15 / test | cổng |
|---|---|---|---|---|---|
| `fd_dml_m4b_frz:15` (head) | oa (audited 03:45Z) | 0,7464 / **0,7690** / 0,7728 | **+0,0367** vs m4b@e15 gp (benign **+0,166**, tệ nhất +0,045) | 0,8281 (−0,023) / 0,8170 | **PASS** gain+worst |
| `fd_dml_m4b_gpool_frz:15` (head) | ob (audited 04:20Z) | 0,7402 / **0,7717** / 0,7795 | **+0,0376** vs gpool@e15 gp (benign **+0,179**, tệ nhất **+0,092**) | 0,8155 (−0,028) / 0,8098 | **PASS** gain+worst |
| `fd_dml_m4b_gpool_frz_gkd:15` (head) | oc (audited 04:45Z) | 0,7808 / **0,8030** / **0,8046** | **+0,0688** vs gpool@e15 gp (benign **+0,200**, tệ nhất **+0,128**); +0,0312 vs gpool_frz head (benign +0,022, tệ nhất +0,036) | 0,8329 (−0,011) / 0,8207 | **PASS** cả hai cổng |

- Control: m4b@e15 gp 0,7323 / test 0,7478; gpool@e15 0,7342 / 0,7486; rej:15 0,7455 / 0,7599. m4b_frz cùng weights luật gp chỉ 0,6399
  (cộng prior hai lần, đúng dự báo). Lợi giữ suốt ep6→15 (+0,045 → +0,037), không teo như gpool. Benign tăng mạnh ⇒ sửa đúng lỗi
  false positive trên dòng benign của receiver chưa thấy (probe 14).
- **gkd (Gaussian gộp học KL từ snapshot cố định) là đóng góp thêm rõ** (+0,031 so gpool_frz) — khớp probe 15 cov_gauss. Khoảng private–proxy
  của chính run: 0,833 − 0,803 = 0,03 (m4b: 0,12). Private test 0,8046 (mô tả; tham chiếu `fl_dagsnet_veremi` 20c r50 khác protocol:
  NILM-FL private 0,696 / proxy 0,823, AFPHA 0,775, centralized 0,853).
- Proxy các arm frz thấp hơn 0,011–0,028 (private không còn kéo proxy theo hướng cũ); không ảnh hưởng mục tiêu private.

### Probe 18 (18/10 ~05:25Z; probe-14 đo trên weights vòng 15 s42; artifact `artifacts/kaggle/fd-probe18-20261015-p18-v1/`)

| ep15 | private head | bias oracle | refit Linear gộp | refit MLP gộp | proxy | proxy bias oracle / refit Linear gộp |
|---|---:|---:|---:|---:|---:|---:|
| m4b (probe 14, gp) | 0,7323 | 0,7777 | 0,8427 | 0,8754 | 0,8509 | 0,8843 / 0,9020 |
| m4b_frz | 0,7690 | 0,8333 | 0,8465 | 0,8807 | 0,8281 | 0,8829 / 0,8979 |
| gpool_frz | 0,7718 | 0,8275 | 0,8373 | 0,8683 | 0,8155 | 0,8817 / 0,8939 |
| **gkd** | **0,8030** | **0,8477** | 0,8554 | 0,8809 | 0,8329 | 0,8798 / 0,8949 |

- Backbone không đổi (refit gộp ≈ 0,84–0,86 như m4b) ⇒ lợi frz/gkd nằm ở tầng quyết định, đúng cơ chế.
- Khoảng còn lại của gkd tới refit gộp (+0,052) **phần lớn là bias theo lớp** (bias oracle +0,045). Proxy cũng lệch bias (+0,047…+0,066).
  ⇒ hướng tiếp: hiệu chỉnh bias 16 lớp ước federated (probe 19).

### Probe 19 (18/10 ~05:35Z; bias 16 lớp; artifact `artifacts/kaggle/fd-probe19-20261015-p19-v1/`) — **FAIL hợp đồng**

| ep15 s42 | proxy: none / F1 oracle / CE / CE cân bằng | private: luật chính | + bias proxy F1 / CE / bCE | bCE riêng trên `old` | F1 oracle riêng | bCE dòng own |
|---|---|---:|---|---:|---:|---:|
| m4b (gp) | 0,851 / 0,884 / 0,869 / 0,853 | 0,7323 | 0,602 / 0,611 / 0,594 (trên head) | 0,7397 | 0,7777 | 0,6594 |
| frz (head) | 0,828 / 0,883 / 0,823 / 0,844 | 0,7690 | 0,750 / 0,773 / 0,748 | 0,7893 | 0,8355 | 0,7272 |
| gkd (head) | 0,833 / 0,880 / 0,827 / 0,844 | 0,8030 | 0,793 / 0,806 / 0,783 | 0,8065 | **0,8488** | 0,7519 |

- Bias proxy (kể cả F1 oracle) chuyển sang private không có lợi (−0,021…+0,004) ⇒ lệch bias là riêng từng private.
- Bias fit bằng CE / CE cân bằng (lồi, federated được) chỉ +0,004…+0,020; lợi +0,045 chỉ có khi tối ưu **macro-F1 trực tiếp bằng nhãn
  toàn cục qua chính private đó** — không làm được federated với private khác kiến trúc (R2). Bias trên dòng own hại.
- Với proxy (model chung), tối ưu macro-F1 federated khả thi (client trả đếm confusion theo lưới bias) và cho +0,03…+0,05 — chỉ là kết quả
  phụ, proxy không phải mục tiêu.

## Vòng 16 — lặp seed 43 + gkd qua q2 (ep15; cập nhật dần)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (benign, tệ nhất) | proxy ep15 / test | cổng |
|---|---|---|---|---|---|
| `fd_dml_m4b_frz:15` s43 (head) | pd (audited 18/10 ~09:40Z) | 0,7308 / **0,7631** / 0,7718 | **+0,0369** vs m4b@e15@s43 gp (benign **+0,172**, tệ nhất +0,044) | 0,8262 (−0,009) / 0,8138 | **PASS** gain+worst |
| `fd_dml_m4b_gpool:15` s43 (gp, control) | pb (audited 18/10 ~09:40Z) | 0,7175 / **0,7367** / 0,7517 | control | 0,8299 / 0,8251 | — |
| `fd_dml_m4b_gpool_frz_gkd:15` s43 (head) | pa (audited 18/10 ~09:39Z) | 0,7720 / **0,7953** / 0,7986 | **+0,0586** vs gpool@e15@s43 gp (benign **+0,190**, tệ nhất **+0,118**); **+0,0690** vs m4b@e15@s43 gp (benign +0,187, tệ nhất +0,123) | 0,8275 (−0,002) / 0,8201 | **PASS** cả hai cổng |
| `fd_dml_m4b_gpool_q2_frz_gkd:15` s42 (head) | pc (audited 18/10 ~09:58Z) | 0,6753 / **0,5773** / 0,5815 | **−0,2256** vs gkd@e15 head (benign −0,272, tệ nhất −0,352) | 0,4790 (−0,354) / 0,4950 | **FAIL** noninferior |

- **gkd lặp đúng ở seed 43** (+0,0586 vs gpool; s42 +0,0688) ⇒ headline gkd xác nhận 2 seed; rd (gkd ep16→50) push 09:39Z.
- **q2 + frz hỏng**: private head giảm ep6 → ep15 (0,675 → 0,577) theo snapshot gộp q2 dao động (proxy 0,40–0,63). q2 với DML thường
  vẫn ổn (gpool_q2@e15 gp 0,7334) vì teacher private là proxy local được train lại. ⇒ frz cần snapshot gần dense; q4 (vòng 18 sf) là kiểm tiếp.
- frz lặp đúng ở seed 43: +0,0369 (s42 +0,0367), benign +0,172 (s42 +0,166) ⇒ snapshot cố định là hiệu ứng thật, không phải seed.
  Control s43 sát s42: m4b@e15 gp 0,7263 (s42 0,7323), gpool@e15 gp 0,7367 (s42 0,7342).

## Luật head vs gp (owner 18/10 ~06Z: luật chính xác hơn làm luật chính) — [bảng đầy đủ: design §Luật](design.md)

Cùng weights, group, mean 20 client: họ `frz` đọc **head** hơn gp ở mọi góc (gkd: global 0,803 vs 0,677; local F1 0,775 vs 0,541;
local acc 0,907 vs 0,501; client tệ nhất 0,761 vs 0,557). Arm DML cũ: gp hơn head ở global (+0,12), local F1 ≈, head hơn local acc +0,025.
⇒ luật chính theo nguồn teacher: frz → head, DML cũ → gp. gkd (head) hơn gpool (gp) ở global +0,069, local F1 +0,066, local acc
+0,015 (so gpool head), client tệ nhất +0,128. Rò prior local còn lại trong head (độ dốc over-prediction theo log π_k − log π_g):
m4b +0,74 → frz +0,18 → gkd +0,13; gp trên frz/gkd −0,41…−0,47 (sửa quá tay) ⇒ cơ sở cho arm `la` vòng 17.

## Vòng 17–18 — cải tiến trên gkd (s42 ep15, head; cổng gain+worst vs gkd@e15 head 0,8030; cập nhật dần)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (benign, tệ nhất) | proxy ep15 / test (Δ) | cổng |
|---|---|---|---|---|---|
| `gkd_la` | ra (audited 18/10 ~11:25Z) | 0,7952 / **0,8189** / 0,8190 | **+0,0160** (benign **+0,070**, tệ nhất **+0,037**) | 0,8294 / 0,8206 (−0,004) | **PASS** gain+worst |
| `gkd_kdpx` | rb (audited 18/10 ~11:10Z) | 0,7871 / **0,8087** / 0,8088 | +0,0058 (benign +0,010, tệ nhất +0,003) | 0,8486 / 0,8308 (**+0,016**) | **FAIL** gain (< +0,010) |
| `gkd:50` (rd, code cũ, nối tiếp ep15 oc) | rd (audited 19/10 ~21:20Z) | ep15 0,8030 → ep30 0,8100 → **ep50 0,8153** / test 0,8063 | +0,012 group / +0,002 test so ep15 | **0,8850** / 0,8488 (+0,052) | mô tả (50 round) |
| `gkd_la_kdpx` | rc (audited 18/10 ~12:55Z) | 0,7983 / **0,8225** / 0,8220 | **+0,0195** (benign **+0,075**, tệ nhất **+0,039**) | 0,8434 / 0,8305 (**+0,0105**) | **PASS** gain+worst |
| `gkd_cop` | sa (audited 18/10 ~13:00Z) | 0,7912 / **0,8060** / 0,8073 | +0,0031 (benign +0,004, tệ nhất −0,003) | 0,8296 / 0,8218 (−0,003) | **FAIL** gain |
| `gkd_pce` | sb (audited 18/10 ~14:25Z) | 0,7976 / **0,8077** / 0,7984 | +0,0047 (benign **−0,053**, tệ nhất −0,013) | **0,8917** / 0,8377 (**+0,059**) | **FAIL** gain |
| `gkd_t2` | sc (audited 18/10 ~14:20Z) | 0,7644 / **0,7972** / 0,7980 | −0,0058 (benign −0,036, tệ nhất −0,009) | 0,8334 / 0,8205 (+0,001) | **FAIL** gain |
| `gkd_gk2` | se (audited 18/10 ~14:20Z) | 0,7798 / **0,8028** / 0,8055 | −0,0002 (benign +0,003, tệ nhất −0,003) | 0,8246 / 0,8190 (−0,008) | **FAIL** gain |
| `gkd_g512` | sd (audited 18/10 ~15:40Z) | 0,7817 / **0,8044** / 0,8056 | +0,0014 (benign −0,007, tệ nhất −0,003) | 0,8358 / 0,8231 (+0,003) | **FAIL** gain |
| `q4_frz_gkd` | sf (audited 18/10 ~14:20Z) | 0,7813 / **0,8042** / 0,8059 | +0,0013 (benign 0,000, tệ nhất −0,002) | 0,8258 / 0,8203 (−0,007) | **PASS** noninferior |

- **la PASS, đúng cơ chế**: độ dốc rò prior local trong head (`prior_leak.py`) gkd +0,133 → la **+0,030** (corr 0,388 → 0,266);
  benign tăng mạnh nhất (+0,070) = bớt dự đoán thừa lớp tấn công phổ biến local trên dòng benign. Ứng viên tổ hợp vòng 19.
- **la_kdpx PASS, tốt nhất vòng 17**: gần cộng dồn (la +0,016, kdpx +0,006 → +0,0195; so la +0,0036); private và proxy cùng tăng; rò prior
  +0,032 (= la). kdpx một mình FAIL nhưng góp phần proxy (+0,011) mà không hại private.
- **cop FAIL dù cơ chế chạy đúng**: syn_accept (teacher đồng ý trên mẫu tổng hợp) 0,48 → **0,67** cả 15 epoch (khớp kiểm local 0,673),
  nhưng private chỉ +0,003 ⇒ độ thật của mẫu (theo teacher) không phải nút thắt của gkd; giữ Gaussian (rẻ hơn ~12× byte bootstrap).
- **q4 PASS noninferior với frz** (q2 FAIL −0,226): proxy mỗi chiều 478 → 62,5 MB/15 epoch (**7,6×**), tổng run kể bootstrap 992 → 161 MB
  (**6,2×**), private không giảm ⇒ int4 giữ snapshot đủ gần dense. Lưu ý FL: scale q4 riêng từng client, không cộng được dưới SecAgg
  ([luồng thông tin §7 d](../../research/federated-information-flow.md)).
- pce (proxy chỉ CE ⇒ teacher = FedAvg-m4b): proxy tốt nhất từ trước (0,892 group, +0,059) và private nhanh đầu (ep6 +0,017) nhưng chững,
  benign −0,053, test 0,798 < gkd ⇒ proxy mạnh hơn không làm teacher tốt hơn cho private; mục tiêu client ⇒ không chọn. Ghi nhận cho báo cáo proxy.
- g512 (512 dòng tổng hợp/bước) +0,001, cop (mẫu thật hơn) +0,003, gk2 (trọng số ×2) 0,000 ⇒ phần mẫu Gaussian của gkd đã bão hòa; probe 15
  dự báo cov_gauss +0,022/epoch không thành hiện thực khi tăng lượng/độ thật. Lợi còn lại đến từ khử rò prior (`la`).
- t2 (KD T=2) hại benign −0,036; gk2 (KL tổng hợp ×2) không đổi ⇒ trọng số/nhiệt độ KD không phải nút thắt.
- **gkd 50 round (rd)**: private tăng chậm (group +0,012, test +0,002 từ ep15), proxy tăng mạnh 0,833 → 0,885 (test 0,849; tham chiếu
  `fl_dagsnet_veremi` centralized 0,853, NILM-FL proxy 0,823, AFPHA private 0,775). la_kdpx ep15 (0,8225) đã hơn gkd ep50.
- kdpx: hướng dương mọi góc nhưng dưới ngưỡng; chênh seed gkd s42/s43 = 0,008 ⇒ +0,006 chưa tách được khỏi nhiễu seed. Proxy +0,016 (bằng
  dấu hiệu thiết kế). Không nới cổng; tổ hợp `la_kdpx` (rc) đọc riêng.

## Vòng 19 — code `--fast` (ep15, head; cổng khóa `@fast`; cập nhật dần)

| arm | session | private ep6 / **ep15** / test | proxy ep15 / test | ms/bước T4 | cổng |
|---|---|---|---|---:|---|
| `gkd` s42 `--fast` (đối chứng) | ta (audited 19/10 ~07:45Z) | 0,7819 / **0,8058** / 0,8066 | 0,8348 / 0,8223 | — | **PASS** equiv vs gkd@e15 cũ: Δ +0,0028 |
| `gkd` s43 `--fast` (đối chứng) | td (audited 18/10 ~18:40Z) | 0,7745 / **0,7984** / 0,8013 | 0,8244 / 0,8175 | 19,34 (cũ 24,54) | **PASS** equiv vs gkd@e15@s43 cũ: Δ +0,0031 |
| `gkd_la_kdpx` s42 `--fast` | tb (audited 18/10 ~18:35Z) | 0,7995 / **0,8269** / 0,8229 | 0,8526 / 0,8327 | 18,75 (cũ 25,19) | **PASS** gain+worst vs ta: **+0,0212** (benign +0,064, tệ nhất +0,042) |
| `gkd_la_kdpx` s43 `--fast` | te (audited 19/10 ~07:45Z) | 0,7874 / **0,8128** / 0,8145 | 0,8326 / 0,8200 | — | **PASS** gain+worst vs td: **+0,0144** (benign +0,067, tệ nhất +0,030) |
| `q4_frz_gkd_la_kdpx` s42 `--fast` | tc (audited 19/10 ~07:45Z) | 0,8003 / **0,8232** / 0,8207 | 0,8410 / 0,8278 | — | **PASS** noninferior vs tb: −0,0037 (tệ nhất −0,001) |

- **Vòng 19 PASS cả 5 cổng**: `--fast` tương đương cả 2 seed (+0,003); la_kdpx lặp 2 seed (+0,021 / +0,014, benign +0,06–0,07);
  q4 giữ la_kdpx (−0,004, byte proxy ÷7,6). Owner 19/10: 50 epoch chỉ theo protocol cuối, sau khi vét hết hướng ⇒ vòng 20 vét la/kdpx.
- **Tương đương end-to-end đầu tiên**: gkd s43 `--fast` vs code cũ +0,0031 (benign +0,011, tệ nhất −0,001) ⇒ `--fast` không đổi kết quả ngoài chênh seed.
- Cùng arm code cũ (rc) 0,8225 / test 0,8220 ⇒ `--fast` lệch +0,004 (≤ chênh seed 0,008); tốc độ −25,6 % đúng benchmark.

## Vòng 20–21 — vét la/kdpx + kênh nén (s42 ep15 `--fast`, head; đối chứng tb la_kdpx 0,8269; audited 19/10 ~12:30Z)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (tệ nhất; benign) | proxy ep15 / test | byte proxy 15 ep (up+down) | cổng |
|---|---|---|---|---|---:|---|
| `la15_kdpx` | va | 0,7969 / **0,8223** / 0,8208 | −0,0046 (+0,003; +0,003) | 0,8442 / 0,8286 | 956 MB | **FAIL** gain |
| `la05_kdpx` | vb | 0,7997 / **0,8208** / 0,8198 | −0,0062 (−0,009; −0,009) | 0,8472 / 0,8300 | 956 MB | **FAIL** gain |
| `lau_kdpx` | vc | 0,8062 / **0,8278** / 0,8240 | +0,0009 (**−0,042**; **−0,042**) | **0,8584** / 0,8380 | 956 MB | **FAIL** gain+worst |
| `la_lap_kdpx` | vd | 0,7975 / **0,8220** / 0,8218 | −0,0050 (+0,011; +0,011) | 0,8407 / 0,8285 | 956 MB | **FAIL** gain |
| `la_kdpx_kd2` | ve | 0,7919 / **0,8209** / 0,8201 | −0,0061 (+0,006; +0,006) | 0,8451 / 0,8281 | 956 MB | **FAIL** gain |
| `q4_…_la_kdpx_zl` | wa | 0,7976 / **0,8241** / 0,8226 | −0,0029 (+0,002; +0,002) | 0,8465 / 0,8295 | **99,5** (50,6 + 48,9) | **PASS** noninferior |
| `q4_…_la_kdpx_ef_zl` | wb | 0,7994 / **0,8216** / 0,8212 | −0,0054 (+0,006; +0,006) | 0,8475 / 0,8316 | 99,8 (50,8 + 49,0) | **PASS** noninferior |
| `q2_…_la_kdpx_ef_zl` | wc | 0,6821 / **0,3130** / 0,3222 | **−0,514** | 0,0578 / 0,0606 | 35,4 (21,4 + 14,1) | **FAIL** (sụp) |
| (tham chiếu) `q4_…_la_kdpx` | tc | 0,8003 / 0,8232 / 0,8207 | −0,0037 | 0,8410 / 0,8278 | 125,0 | PASS (vòng 19) |

- **Hướng la/kdpx vét xong, τ=1 với prior gộp là tối ưu**: 5 biến thể nằm trong 0,8208–0,8278 (dải chênh `--fast`/seed ~0,008), không biến thể nào
  qua +0,010. `lau` (đích prior đều) đẩy proxy cao nhất vòng 20 (0,858; pce 0,892 vẫn cao nhất) nhưng benign −0,042 ⇒ dịch về prior đều làm head bỏ lớp benign phổ biến.
  `la_lap` (la trên CE proxy) tốt cho client tệ nhất (+0,011) nhưng trung bình −0,005. τ 0,5 / 1,5 đều thua τ=1 ⇒ không có xu hướng đơn điệu cần thử tiếp.
- **Kênh nén chọn `q4_zl`**: LZMA2 không mất mát giảm thân gói còn **0,80** (xuống) ⇒ byte proxy 125 → **99,5 MB** (÷9,6 so dense), private −0,003.
  `ef` chiều lên không thêm gì (byte bằng, Δ −0,005 vs −0,003 trong nhiễu) mà thêm trạng thái client ⇒ bỏ.
- **q2 sụp kể cả có EF hai chiều**: `q_residual_rel` (phần dư server/|exact|) 0,27 ở ep1 → 0,71 ở ep15 (q4: 0,008) ⇒ downlink 2-bit (3 mức, làm tròn gần nhất)
  không theo kịp, replica client xa dần model thật, frz khuếch đại (bài học vòng 16). Bước giữa chưa thử: 3-bit, hoặc q2 chỉ chiều lên.
- Tổng byte run (kể bootstrap) dense 992 → q4 161 → **q4_zl 136 MB**. `proxy_bootstrap` (state proxy ban đầu gửi mỗi client, 31,9 MB) giờ là
  **23 %** tổng ⇒ đích tiếp theo.
- Thời gian session 212–227 phút cho 15 epoch (`--fast`, 2×T4).

## Vòng 22 — kênh dưới int4 (`--fast`, head; audited 20/10 ~03Z)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (tệ nhất) | proxy ep15 / test | byte proxy 15 ep (up + down) | `q_residual_rel` ep1 → ep15 | cổng |
|---|---|---|---|---|---:|---|---|
| `q3_…_la_kdpx_zl` s42 | xa | 0,7976 / **0,8246** / 0,8216 | −0,0024 (−0,004) vs tb | 0,8413 / 0,8280 | **66,8** (34,0 + 32,8) | 0,097 → 0,021 | **PASS** noninferior |
| `q4_u2_…_la_kdpx_ef_zl` s42 | xb | 0,7747 / **0,7474** / 0,7544 | **−0,0795** (−0,042) vs tb | **0,4544** / 0,4512 | 73,6 (21,6 + 52,0) | 0,049 → 0,068 (không giảm) | **FAIL** |
| `q4_…_la_kdpx_zl` s43 | xd | 0,7837 / **0,8177** / 0,8178 | +0,0050 (−0,003) vs te | 0,8427 / 0,8249 | 99,3 (50,7 + 48,6) | 0,040 → 0,009 | **PASS** noninferior |
| (tham chiếu) `q4_…_zl` s42 | wa | 0,7976 / 0,8241 / 0,8226 | −0,0029 | 0,8465 / 0,8295 | 99,5 | 0,042 → 0,008 | PASS (vòng 21) |

- **q3 giữ performance, byte proxy −33 % so q4_zl** (99,5 → 66,8 MB; tổng run 136 → 103 MB): phần dư server khởi đầu cao hơn q4 (0,097 so 0,042)
  nhưng giảm dần như q4 (0,021 ở ep15) ⇒ downlink 7 mức theo kịp, đúng dự báo (ngưỡng về 0 là max/6). LZMA2 nén mã 3-bit còn 0,54 (q4: 0,80).
- **Chiều lên 2-bit hỏng dù có EF và trung bình 20 client**: proxy sụp (0,454), phần dư server không giảm (0,068 suốt). Giả thuyết "chiều lên chịu ít bit
  vì làm tròn ngẫu nhiên không lệch" **bị bác**: không lệch nhưng phương sai theo scale max hàng quá lớn so với delta mỗi round. xc (`q3_u2`) không push.
- **q4_zl lặp s43 PASS** (+0,005 vs te) ⇒ kênh nén không giảm performance ở cả hai seed.
- Bootstrap (31,9 MB) giờ là **31 %** tổng byte của q3 (103 MB).

## Vòng 23 — 2-bit bước RMS + lặp q3 s43 (`--fast`, head; yb audited 20/10 ~08Z, ya-v2 audited 20/10 ~15Z)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (tệ nhất) | proxy ep15 / test | byte proxy 15 ep (up + down) | `q_residual_rel` ep1 → ep15 | cổng |
|---|---|---|---|---|---:|---|---|
| `q2_rs_…_la_kdpx_zl` s42 | yb | 0,7785 / **0,8136** / 0,8137 | **−0,0133** (−0,001) vs tb | 0,8284 / 0,8166 (−0,024) | 52,0 (26,8 + 25,2) | 0,094 → 0,078 (gần phẳng) | **FAIL** noninferior |
| `q3_…_la_kdpx_zl` s43 | ya-v2 | 0,7865 / **0,8160** / 0,8166 | **+0,0032** (+0,0054) vs te | 0,8434 / 0,8250 | **66,7** | 0,095 → 0,022 | **PASS** noninferior |
| (tham chiếu) `q3_…_zl` s42 | xa | 0,7976 / 0,8246 / 0,8216 | −0,0024 | 0,8413 / 0,8280 | 66,8 | 0,097 → 0,021 | PASS (vòng 22) |

- **Bước RMS chữa được sụp đổ của 2-bit nhưng không giữ performance.** q2 scale max (wc) phân kỳ (residual 0,27 → 0,71, proxy sụp); q2_rs ổn định
  (proxy 0,828, client tệ nhất ≈ tb) nhưng private −0,013, proxy −0,024. Dấu hiệu sai đã khóa trúng một nửa: **residual không giảm như q3**
  (0,078 ở ep15 so 0,021) ⇒ phần bị cắt (> 1,5 bước) dồn lại trong EF server mỗi round, downlink 3 mức không xả kịp; proxy không sụp.
- **Lợi byte nhỏ hơn dự kiến:** −22 % so q3 (66,8 → 52,0 MB), dự kiến −32 %; LZMA2 nén mã 2-bit kém (tỷ lệ downlink 0,74, q3 0,54).
  Đổi −0,011 F1 lấy 14,8 MB không đáng ⇒ **2-bit loại**; kênh = q3_zl nếu ya-v2 (s43) PASS, nếu không q4_zl (PASS cả 2 seed).

- **Vòng 23 đã audit đủ:** ya-v2 15/15 round, 87 file pull không thiếu, executed source khớp, 186 lượt kiểm predictions → confusion → 10 metric; weights ep6/15 strict và hữu hạn. q3_zl PASS cả s42/s43 ⇒ **chốt q3_zl**, không chạy tiếp nhánh 2-bit hay tổ hợp q3_u2 đã mất điều kiện.

## Probe 20 — mẫu tổng hợp có đúng chỗ private còn sai không (21/10 ~06Z; local, không train)

`fd_probe20.py` trên weights ep15 của xa (`q3_…_la_kdpx_zl` s42, audited), fixture pilot local (group 65.536 dòng; train mẫu 2,5 %), m4b.
Teacher = snapshot nhận được (`proxy[0]`, group F1 0,842), private head TB 0,824 (khớp audit 0,8246). "Cặp" = (teacher argmax, private argmax)
trên dòng hai model khác nhau; cos so với cặp của dòng group mà **private sai, teacher đúng** (3,1 % dòng). align = bất đồng × cos.
Mẫu tổng hợp 16.384 dòng/nguồn/client. [JSON](results/probe20-xa-s42-e15.json).

| nguồn | bất đồng | cos với lỗi thật | align | cặp bất đồng chính |
|---|---:|---:|---:|---|
| group (H) | 5,1 % | — | — | benign↔timeDelay 41 %, positionMirroring→benign 10 %, dataReplay↔benign 13 % |
| dòng own (O) | 4,2 % | 0,75 (gộp 0,94; client thấp nhất 0,38) | 0,032 | như H |
| dòng thật client khác (X, oracle) | 5,0 % | 0,96 | 0,048 | như H |
| Gaussian gkd (G) | 12,7 % | 0,09 | 0,011 | constantSpeedOffset→randomSpeedOffset 11 % |
| G top-25 % KL của tập 4× (Gk) | 33 % | 0,10 | 0,034 | randomSpeedOffset |
| G đẩy 5 bước tăng KL (Ga, Mahalanobis 0,94 → 116) | 51 % | 0,07 | 0,035 | randomSpeedOffset |
| G lớp/cặp theo bất đồng dòng own (Gq / Gp) | 16 / 15 % | 0,11 / 0,12 | 0,018 | randomSpeedOffset |
| copula vòng 18 (Gc) | 9,6 % | 0,36 | 0,034 | dataReplay↔benign |
| dòng own + nhiễu covariance gộp 0,5 / 1,0 (Gn) | 12 / 15 % | 0,12 / 0,07 | 0,014 / 0,011 | randomSpeedOffset |
| dòng own, 30 % feature thay giá trị Gaussian (Gsg) | 10 % | 0,28 | 0,028 | lẫn |
| dòng own, 30 % feature thay giá trị copula (Gs / lớp ưu tiên Gsp) | 8,0 / 12,3 % | 0,53 / 0,61 | 0,042 / 0,076 | dataReplay↔benign, positionMirroring↔benign |
| **lai hai dòng own, cặp ưu tiên ρ 0,3 / 0,5 (Gx3 / Gx5)** | 12,7 / 14,4 % | 0,55 / 0,53 | **0,069 / 0,076** | dataReplay→benign, positionMirroring→benign, benign→timeDelay |
| lai hai dòng own, cặp đều (Gxu) | 12,0 % | 0,38 | 0,045 | lẫn |

- **Lỗi còn lại của private nằm ở khối benign**, đúng lỗi đã chẩn đoán từ probe 14; dòng own của chính client cho biết cặp đó (miễn phí, 0 byte).
- **Mẫu Gaussian bất đồng nhiều nhưng sai chỗ**: phần KL của gkd rơi vào cặp tốc độ (randomSpeedOffset), không chạm khối benign ⇒ giải thích vì sao
  tăng lượng (g512), trọng số (gk2) hay độ thật biên (cop) không thêm gì. Ưu tiên **trong** Gaussian (KL, đối kháng, lớp, cặp) không đổi được chỗ đó.
- Nhiễu liên tục ở feature tốc độ (thật là atom) là thủ phạm: thêm nhiễu covariance gộp vào chính dòng own cũng kéo về randomSpeedOffset; thay 30 %
  feature bằng giá trị copula (biên thật) thì không.
- Ưu tiên cặp chỉ có tác dụng khi bộ sinh nằm trên dữ liệu thật: Gx3 0,069 vs Gxu 0,045, Gsp 0,076 vs Gs 0,042; mọi client cos Gx3 > G.
- Giới hạn: fixture pilot 2,5 %, một bộ weights, chỉ số phân bố — cop từng có align gấp 3 Gaussian mà utility +0,003. Quyết định bằng train (vòng 24).

## Vòng 24 — lai ghép dòng own theo cặp bất đồng (`--fast`, head; xo audited 21/10 ~12Z, xou ~12:20Z)

| arm | session | private ep6 / **ep15** / test | Δ ep15 (benign, tệ nhất) | proxy ep15 / test | cổng |
|---|---|---|---|---|---|
| `q3_…_la_kdpx_zl_xo` s42 | xo | 0,8026 / **0,8321** / 0,8279 | **+0,0076** (+0,0063, +0,0113) vs xa | 0,8377 / 0,8272 | **FAIL** gain (cần ≥ +0,010) |
| `q3_…_la_kdpx_zl_xou` s42 (cặp đều) | xu | 0,8092 / 0,8275 / 0,8253 | +0,0030 (+0,0098, +0,0041) vs xa; xo − xou +0,0046 | 0,8477 / 0,8311 | mô tả (không đọc vì xo FAIL) |
| (tham chiếu) `q3_…_zl` s42 | xa | 0,7976 / 0,8246 / 0,8216 | — | 0,8413 / 0,8280 | — |

- **Cơ chế chạy đúng như probe dự báo nhưng lợi dưới ngưỡng.** Mọi round 2–15, 20/20 client (trừ 1 client ep5) có cặp ưu tiên chứa benign;
  cặp nặng nhất ep15 = (teacher timeDelayAttack, private benign), rồi benign→positionMirroring / timeDelay / dataReplay — đúng khối lỗi của probe 20.
  Bất đồng trên dòng lai ổn định 0,09–0,12 (private không nuốt hết), bất đồng trên dòng own giảm 0,47 → 0,045.
- Δ +0,0076, benign +0,0063, client tệ nhất +0,0113: dương ở cả ba nhưng nằm trong biên lệch giữa seed (~0,008) ⇒ không phân biệt được với nhiễu
  một seed. Proxy −0,004. **Dấu hiệu sai đã khóa trúng:** "bất đồng đúng chỗ" của probe (align 0,069, gấp 6 Gaussian) không thành utility
  tương ứng — như cop (align gấp 3, +0,003). Theo khóa: dừng hướng, không quét ρ/floor; cổng ưu tiên xo − xou không đọc (xou chỉ mô tả).
- Test chỉ báo cáo, không dùng chọn (private test +0,006).

## Vì sao sinh mẫu có chủ đích không ăn — phân tích 21/10 ~13Z (không train)

Nguồn: dự đoán từng dòng đã audit (ep15 group, 20 private + proxy), [probe 21](results/probe21-xa-s42-e15.json) và
[probe 22](results/probe22-xa-s42-e15.json) trên weights xa ep15 (`fd_probe21.py`, `fd_probe22.py`; local, fixture pilot, train mẫu 2,5 %;
teacher head tái hiện 0,842 vs audit 0,8413). "Dạy được" = private sai, teacher đúng; "chung" = cả hai sai; trần = F1 private nếu sửa hết dòng dạy được.

**1 — Phần lỗi teacher dạy được đã cạn; lỗi chung không đổi qua mọi arm.**

| arm ep15 s42 | private | teacher | lỗi private | dạy được | chung | trần |
|---|---:|---:|---:|---:|---:|---:|
| gkd (oc) | 0,8030 | 0,8329 | 0,166 | 0,067 | 0,099 | 0,863 |
| la_kdpx (tb) | 0,8269 | 0,8526 | 0,134 | 0,032 | 0,102 | 0,868 |
| q4_zl (wa) | 0,8241 | 0,8465 | 0,134 | 0,030 | 0,104 | 0,862 |
| q3_zl xa | 0,8246 | 0,8413 | 0,136 | 0,031 | **0,105** | 0,860 |
| xo | 0,8321 | 0,8377 | 0,131 | 0,026 | **0,105** | 0,859 |

la/kdpx (PASS) ăn hết nửa phần dạy được (0,067 → 0,031); không arm nào chạm phần chung (~0,10 = **77 %** lỗi private ở xa). Mọi mẫu tổng hợp
học KL từ snapshot (Gaussian, cop, g512, gk2, xo) chỉ với tới 23 % còn lại ⇒ trần +0,036, xo lấy +0,0076.

**2 — Bất đồng ≠ lỗi: một nửa ngân sách ưu tiên của xo rơi vào cặp teacher đoán mò.** Cặp (teacher a, private b) trên group, gộp 20 private (xa):

| teacher → private | phần bất đồng | teacher đúng | private đúng |
|---|---:|---:|---:|
| benign → timeDelay | 36,6 % | 58 % | 33 % |
| timeDelay → benign | 13,2 % | 48 % | 50 % |
| posMirror → benign | 7,2 % | **91 %** | 6 % |
| dataReplay → benign | 6,6 % | 58 % | 35 % |
| benign → posMirror | 3,9 % | 21 % | 71 % |
| congestionSybil → benign / constPosOff → benign | 2,3 / 2,1 % | **86 / 96 %** | 4 / 1 % |

Cặp nặng nhất của xo (timeDelay ↔ benign) là đồng xu: xo sửa 55 và làm hỏng 221 dòng timeDelay mỗi client (ΔF1 −0,004); toàn bộ lợi đến từ
cặp teacher đáng tin (posMirror +0,081; xou cặp đều +0,036 ⇒ ưu tiên chỉ thêm ở đây). Probe 22 A: trọng số xo cho cặp có teacher đúng trên group
**0,506**; trọng số theo độ tin cậy đo bằng nhãn own (mass × max(0, 2ρ_own − 1), 0 byte) **0,665**; tương quan ρ own–group 0,43.

**3 — Lớp nút thắt là lớp teacher chưa biết, không phải lớp mẫu chưa phủ.** F1 private head ep15:

| arm | benign | posMirror | timeDelay | dataReplay | macro |
|---|---:|---:|---:|---:|---:|
| gkd / cop / g512 / gk2 / t2 | 0,68–0,72 | 0,29–0,34 | **0,18–0,19** | 0,63–0,65 | 0,797–0,806 |
| la_kdpx / xa / xo | 0,78–0,79 | 0,38–0,46 | **0,20–0,28** | 0,72–0,73 | 0,823–0,832 |
| teacher xa | 0,798 | 0,476 | 0,277 | 0,754 | 0,841 |

timeDelay của teacher 0,28 ở ep15 (gkd:50 lên 0,46 ở ep50; posMirror 0,37 → 0,60) ⇒ ở ep15 không nguồn KL nào có tri thức timeDelay để truyền.

**4 — Tri thức có trong feature; thiếu ở tầng quyết định của teacher** (probe 21, backbone đóng băng, refit trên dòng thật có nhãn gộp 20 client):

| | head như train | refit Linear cuối | refit cả head |
|---|---:|---:|---:|
| teacher (macro / timeDelay / posMirror) | 0,842 / 0,274 / 0,476 | 0,892 / 0,404 / 0,771 | **0,919** / 0,501 / 0,841 |
| private TB 5 client | 0,827 | 0,855 | 0,889 |

Refit teacher sửa ~50 % lỗi chung của private. (Cột bias oracle của probe 21 fit sai mục tiêu — làm private tệ đi — bỏ qua.)

**5 — Hiệu chỉnh teacher bằng đặc trưng ảo theo lớp (probe 22 B):** tổng Σh, Σhhᵀ theo lớp của h256 snapshot trên dòng own (cộng được dưới
SecureSum) → server rút đặc trưng ảo N(μ_c, Σ) theo tỷ lệ lớp gộp, fit lại Linear cuối từ trọng số đã train.

| teacher | F1 group | lỗi private teacher đúng | trần private | byte lên/client/round (4 B/giá trị) |
|---|---:|---:|---:|---:|
| như train | 0,842 | 21 % | 0,861 | — |
| LDA (covariance chung) | 0,848 | 38 % | 0,903 | 148 kB |
| weak6 (covariance riêng 6 lớp teacher yếu nhất trên dòng own) | 0,852 | 44 % | 0,925 | 938 kB |
| CCVR (covariance riêng 16 lớp) | **0,893** (timeDelay 0,45, posMirror 0,77) | **47 %** | 0,926 | 2,12 MB |
| diag | 0,816 | 43 % | 0,913 | — |

Khớp probe 17 (CCVR proxy +0,045 / +0,056). Trần là cận trên (chưa trừ dòng teacher mới sai mà private đúng). Byte: tổng bình phương
trên 65.536 dòng vượt tầm fp16 ⇒ 4 B/giá trị, nửa tam giác (khóa `bytes_per_client_fp16` trong JSON probe 22 là số fp16, không dùng).

**6 — Private chững khi teacher còn lên** (run cuối, test, 20c): zc gkd private 0,806 → 0,810 (r12 → r27), proxy 0,823 → 0,839; za la_kdpx
0,817 → 0,828, proxy 0,822 → 0,839. gkd:50 pilot: private +0,012, proxy +0,052 (ep15 → 50). Screening ep15 đo ở vùng teacher ≈ private.

## Vòng 25 — teacher hiệu chỉnh bằng moment lớp, covariance chung (`cvl`; `--fast`, head; audited 22/10 ~01:10Z)

| arm | session | private ep15 | Δ ep15 vs xa (benign, tệ nhất) | proxy ep15 | byte/client 15 ep | cổng |
|---|---|---|---|---|---|---|
| `q3_…_la_kdpx_zl_cvl` s42 | `khanhmay0304/…-s42cl-20261015-r30-v1` | 0,8110 | **−0,0136** (**−0,0185**, −0,0075) | 0,8449 (+0,0036) | 7,48 MB = **×1,451** xa (cv_stats 2,07 + head 0,23 MB) | **FAIL** gain + benign; byte đạt |
| (tham chiếu) `q3_…_zl` s42 | xa | 0,8246 | — | 0,8413 | 5,15 MB | — |

Quỹ đạo (private head group; chẩn đoán teacher trên dòng own, round ≥ 2):

| ep | teacher gốc → hiệu chỉnh | dạy được gốc → hiệu chỉnh | Δ private vs xa |
|---|---|---|---|
| 2 | 0,368 → 0,632 | 0,180 → 0,194 | +0,0150 |
| 3–4 | 0,697 → 0,735 · 0,736 → 0,768 | 0,175 → 0,230 · 0,165 → 0,148 | −0,017 · −0,015 |
| 5 | 0,782 → 0,789 | 0,178 → 0,139 | −0,010 |
| 6–15 | hiệu chỉnh **thấp hơn** gốc 0,002–0,020 | 0,12–0,18 → 0,06–0,11 | −0,008 … −0,019 |

F1 private theo lớp ep15 (cvl − xa): **lớp mục tiêu lên** posMirror +0,166 (0,378 → 0,544), timeDelay +0,106 (0,281 → 0,387),
suddenStop +0,059; **lớp head train đã tốt xuống** suddenConstantSpeed −0,212, zeroSpeedReport −0,122, constSpeedOff / randomSpeedOff /
constPosOff / reversedHeading −0,03 … −0,045, benign −0,019.

- **Dấu hiệu sai đã khóa trúng (a) và (c).** (a) teacher hiệu chỉnh − gốc < +0,02 ở 12/14 round; từ ep6 head refit từ moment thua head đã train
  (đo trên dòng own, head gốc có lợi in-sample nhưng kết quả private trên group xác nhận). (c) benign −0,0185 < −0,005. Theo khóa: **dừng**,
  không quét biến thể cv; bản 16-bit/s43 không chạy.
- Cơ chế: một covariance chung (LDA) buộc mọi lớp cùng hình dạng ⇒ head refit dời biên về phía lớp teacher yếu (đúng như probe 22 dự báo cho
  timeDelay/posMirror) nhưng đánh đổi các lớp tốc độ có phân bố hẹp/khác dạng. Probe 22 đã cho tín hiệu này: LDA chỉ +0,006 macro teacher
  (lợi 0,893 là của CCVR covariance riêng, bị ngân sách byte loại). Lợi sớm (ep2 +0,015) chỉ có khi head snapshot còn non.
- Test chỉ báo cáo, không dùng chọn: private test 0,7978 vs xa 0,8216 (−0,024).

## Vòng 26 — teacher Project48 mỗi round (`cvp`) + support copula (`cop`) (`--fast`, head; audited 22/10 ~19:15Z)

| arm | session | private ep15 | Δ ep15 vs xa (benign, tệ nhất) | proxy ep15 | byte/client 15 ep | cổng |
|---|---|---|---|---|---|---|
| `q3_…_la_kdpx_zl_cvp_cop` s42 | `minhtrit06/…-s42cp-20261015-r31-v1` | 0,8249 | **+0,0003** (+0,0053, +0,0026) | 0,8536 (+0,0123) | 8,20 MB = **×1,59** xa (cv_stats 1,10 + head 0,23 + copula 1,92 MB) | **FAIL** gain |
| (tham chiếu) `q3_…_zl` s42 | xa | 0,8246 | — | 0,8413 | 5,15 MB | — |

Audit `collect_fd.py`: 88 file, source/fingerprint, chuỗi marker, weights strict load, predictions → confusion theo client → 10 metric khớp.
Quỹ đạo private head group (cp / xa): ep1 0,741 / 0,665 · ep2 0,682 / 0,652 · ep6 0,802 / 0,798 · ep10 0,818 / 0,814 · ep13 0,823 / 0,824 ·
ep15 0,825 / 0,825. Teacher trên dòng own (gốc → hiệu chỉnh): ep2 0,289 → 0,618, ep3 0,689 → 0,746, ep4 0,755 → 0,779, ep5 0,781 → 0,797,
**ep6–15 chênh −0,008 … +0,015** (ep15 0,842 → 0,844); dạy được 0,07–0,11 → 0,06–0,09. `kd_syn_accept` 0,67 (xa 0,46): copula được gate nhận nhiều hơn.

F1 private theo lớp ep15 (cp − xa): **posMirror +0,120** (cơ chế P33 chở được online), suddenConstantSpeed +0,096; **timeDelay −0,078**,
accelerationMultiplication −0,046, **suddenStop −0,025** (harness dự báo +0,053), constPosOff −0,023, constSpeedOff −0,015, randomSpeedOff −0,012,
benign +0,005. Holdout `old` (development, mô tả): private 0,8002 vs 0,8175 (−0,017). Test chỉ báo cáo: private 0,8074 vs 0,8216 (−0,014).

- **Dấu hiệu sai đã khóa trúng (a):** teacher hiệu chỉnh − gốc < +0,02 ở 11/14 round (chỉ ep2–4 ≥ +0,02). Online, head đã train của teacher gốc đuổi
  kịp từ ep6; lợi teacher của harness frozen-head (snapshot xa ep15, hiệu chỉnh một lần) không tồn tại trong vòng lặp. (b) không áp dụng vì teacher
  không lên; (c) benign +0,0053 đạt. Theo khóa: **dừng**, không nén R1, không s43, không quét biến thể cvp/cop.
- Support copula đổi phân bố lỗi: lên posMirror/suddenConstantSpeed, xuống timeDelay/accelMult/suddenStop ⇒ macro ≈ 0. Proxy group +0,012 nhưng
  proxy old/test thấp hơn xa (0,821 / 0,822 vs 0,831 / 0,828).
- Harness P36 (cả hai seed PASS) **không dự báo được online**: +0,0077 frozen-head → +0,0003 online. Cùng kiểu với vòng 18 `cop` (+0,0031) và
  vòng 25 `cvl` (P22 teacher lên → online −0,0136).

## Nghiên cứu synthetic22/10 — Probe23–28, đóng phiên để compact

[Kết quả/cổng đầy đủ](../../research/methods/synthetic-learning-recovery-2026-10-22.md),
[bàn giao](../../archive/operations/continuation-synthetic-2026-10-22.md),
[audit build cũ](../../research/methods/synthetic-build-history-audit-2026-10-22.md).
CCVR Project48 giảm calibration packet, teacher PASS hai seed Kaggle nhưng private transfer/inversion **FAIL**;
class-weighted local chưa đủ gain, pooled-real support oracle chỉ chẩn đoán R3-ineligible, gradient-only teacher **FAIL**.
Terminal source/input/predictions đã audit; đây là frozen-head mechanism screen, không online ep15/r50 hoặc R1/R2 proof.
Owner dừng để compact; chưa launch phương pháp mới, run cuối giữ `gkd + la + kdpx`, proxy `q3_zl`.

## Kiểm chứng feature trên recipe hiện hành — seed42 ep15 (audit 25/10)

Cùng `--fast --holdout_only`, 20 client, seed42, round1→15; headline là macro-F1 `private/head/group`.
`m4b` dùng đối chứng đã audit vòng 22 (`xa`); hai artifact mới dùng cùng collector/pattern với đối chứng.

| input | kernel | private group | proxy group | Δ private vs m4b | trạng thái |
|---|---|---:|---:|---:|---|
| full66 | `hoangdang1711/…s42fa…r32-v1` | 0,7728 | 0,7859 | −0,0518 | 15/15, **AUDITED** |
| drop53 | `dangthiet/…s42fg…r37-v3` | 0,8138 | 0,8358 | −0,0108 | 15/15, **AUDITED** |
| m4b (control) | `annhannh331/…s42xa…r22-v1` | **0,8246** | **0,8413** | — | 15/15, **AUDITED** |

Mỗi artifact mới: 88 file lọc, đủ 15 prediction + marker, weights round6/15, config/result/log/source;
executed source khớp notebook push, input/source/data fingerprint khớp, weights strict-load + finite, 180 lượt
prediction→confusion→10 metric khớp. `drop53` checkpoint rebuild đúng `n_features=53,input_pad=1`.
Kết luận mô tả một seed: bỏ vật lý 13 feature tốt hơn full66 nhưng vẫn thấp hơn đặt 0 m4b 0,0108 macro-F1 private.

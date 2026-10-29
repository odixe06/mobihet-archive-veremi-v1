# Continue — bàn giao sống, 23/10/2026 ~07:30Z

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này.
Mỗi phiên sửa trực tiếp trang này; snapshot cũ chuyển vào [archive/operations](../archive/operations/README.md) kèm ngày, không tạo
`continuation-*.md` mới cạnh trang này.

## Cập nhật Kaggle 27/10/2026

CLI `nckh` đã nâng **2.2.5**, pin/skill hai cây và helper giữ version kernel đã cập nhật.
Chi tiết kiểm chứng và giới hạn: [Kaggle](kaggle.md#kaggle-cli-225--cập-nhật-27102026).
Chỉ cập nhật tooling; không thay trạng thái nghiên cứu/train hay thực hiện Git/DVC/backup.

## Thiết lập host 25/10/2026

Checkout mới ở `/home/odixe/project/mobihet`, env `nckh`; cấu hình máy hiện tại:
[knowledge/local-env](../../knowledge/local-env.md). Skill `.agents` / `.claude` đã đồng bộ và sửa frontmatter.
Đã thêm 16 account token; user login CLI sau, xem [Kaggle](kaggle.md). Remote DVC đã chuyển sang rclone
`~/Cloud/OneDrive`; push/cloud-read/pull file thử đã PASS, xem [backup](backup.md).
Chỉ thiết lập môi trường/kiểm tra; không launch Kaggle, không resume nghiên cứu, không Git push hoặc backup toàn bộ.
Bảng công việc bên dưới là bàn giao lịch sử 23/10, không xác nhận tiến trình còn chạy trên host mới.

### Kiểm tra lại host 26/10/2026

Đã sửa tài liệu môi trường theo Lubuntu/LXQt và số đo hiện tại; cấu hình RAM/swap/GPU,
runtime, Conda và dataset ORICO tại [local-env](../../knowledge/local-env.md).
Nguồn dataset local theo owner là ORICO, không phải bản dataset OneDrive; remote DVC giữ nguyên.
Không khởi động train/collector/backup trong phiên này. Kiểm process local không thấy job MobiHet;
có tác vụ Python `rcd` của dự án khác và hai rclone mount — không can thiệp.
Không kiểm lại trạng thái mọi kernel Kaggle remote; bảng 23/10 bên dưới vẫn là lịch sử.

## Tiến trình trong phiên thiết lập 25/10

Không job train/collector/backup nào được khởi động. Chỉ kiểm token/quota Kaggle, không xác nhận trạng thái mọi kernel remote.

## Bàn giao tiến trình lịch sử 23/10

| việc | chủ | trạng thái | ghi chú |
|---|---|---|---|
| Copy dvc-storage Google Drive → OneDrive (`copy-current-dvc.sh`) | phiên Codex của owner | chạy từ ~09:45Z | **không sửa** [backup](backup.md), skill repo-backup, `.dvc/config`, `scripts/dvc_*` khi phiên đó còn chạy |
| Kaggle | — | không kernel nào chạy | vòng 26 `minhtrit06/mobihet-fd-s42cp-20261015-r31-v1` COMPLETE 19:01Z, collect + audit xong, **FAIL** |
| Kaggle | — | không kernel nào chạy | P37 `minhtran0601/mobihet-fd-probe37-20261023-v1` COMPLETE, pull + audit 23/10 ~07:30Z |

## Backup 23/10 — dừng giữa chừng theo owner (owner tự push + cập nhật [backup](backup.md))

Remote DVC `onedrive`. **Đã đẩy:** 9 file `.pt` probe P24–36 (`docs/methods/full-drift/results/*.pt.dvc`, mới, đã `git add` do autostage;
`dvc status -c` in sync); `fixture-cache.dvc` và `input.dvc` merge xong (manifest mới trong working tree). **Chưa:** `kaggle.dvc` (manifest
cũ giữ nguyên; object mới đẩy được một phần, file `.tmp` dở đã xóa). Lỗi: `ENOMEM` ngắt quãng khi ghi/đọc `/mnt/d` lúc swap WSL đầy;
`scripts/dvc_merge_push.py` đọc/copy khối lớn — bản thử chunk 256 KB + retry chỉ ở scratchpad, chưa đưa vào repo. Git chưa commit/push.

## Run cuối seed42 — xong, không chạy lại

6 chuỗi × 50 round đã train + chấm + audit; [report đầy đủ](../methods/full-drift/results/final/report.md) (10 metric, per-class, mọi round,
private/private_ema/proxy, cost, so sánh baseline), [VERIFIED](../methods/full-drift/results/final/VERIFIED.json),
[provenance kernel/version/source](../methods/full-drift/results/final/provenance.json). Phương pháp `gkd + la + kdpx`, proxy `q3_zl`:
za 20c / ze 50c / zi 100c; đối chứng `gkd` dense: zc / zg / zk. Seed43 final chưa chạy. Không repull/re-audit các kernel final.
Contract/timing: [final-pipeline](final-pipeline.md). Tái sinh report chỉ khi artifact đổi:

```bash
B=methods/full-drift/builds/fd-probe/scripts
conda run -n nckh python "$B/report_final.py" && conda run -n nckh python "$B/report_final_md.py" && conda run -n nckh python "$B/check_final_report.py"
```

## Nghiên cứu synthetic — khoảng private ↔ proxy

Khoảng r50 proxy − private: 0,028 / 0,025 / 0,024 (20/50/100c); private chững khi proxy còn lên. Backbone private đủ (refit head trên dữ liệu
thật gộp ≈ proxy) ⇒ thiếu ở tầng quyết định vì private chỉ thấy input own; tri thức vùng client khác chỉ tới qua mẫu tổng hợp.

| mốc | kết quả | hồ sơ |
|---|---|---|
| P23–28 | Project48 teacher PASS hai seed; transfer/inversion FAIL; support thật (oracle) + Project48 +0,013 | [recovery](../research/methods/synthetic-learning-recovery-2026-10-22.md) |
| P29–31 | synthetic không tái tạo hướng cập nhật private; joint carrier P30 FAIL; teacher-nullspace nhỏ | [gradient diagnosis](../research/methods/synthetic-gradient-diagnosis-2026-10-22.md), [identifiability](../research/methods/synthetic-transfer-identifiability-2026-10-22.md) |
| P32 (22/10 10:40Z) | copula × Project48 +0,0024 mean (**FAIL** cổng +0,005) nhưng worst +0,0218; ablation: lợi support thật cần **4 nhóm feature cùng thật** (`snd_kin`, `sess`, `geo`, `prof`), mỗi nhóm Gaussian hóa mất 42–75 % lợi | [support realism](../research/methods/synthetic-support-realism-2026-10-22.md) |
| **P33/P33b (22/10 11:20Z)** | chẩn đoán cách client học (không train): 71 % lợi support thật = recall **posMirror + suddenStop** vs benign; dòng own cân bằng lớp **không** thêm gì (own_bal −0,0015) ⇒ cần biến thể ở receiver client khác; trong không gian đặc trưng private, Gaussian/copula nửa số mẫu lệch manifold (precision 0,49/0,45 vs real 0,86), copula nhòe atom làm suddenStop tệ hơn Gaussian | [client learning](../research/methods/client-learning-diagnosis-2026-10-22.md) |
| **P34 (22/10 11:45Z)** | **FAIL**: `pcop` (atom chính xác + trạng thái first + profile categorical + định danh extractor) +0,0023 mean, không hơn copula; mọi biến thể copula +0,0020…+0,0025; định danh trên primitive copula làm posMirror tệ hơn (precision 0,34 → 0,22); V4-S/G61 dưới Project48 vẫn ≈ Gaussian | [structured support](../research/methods/structured-support-generator-2026-10-22.md) |
| **P35 (22/10 13:35Z)** | chẩn đoán: lợi PS cần **mẫu thật của chính lớp tấn công** (copula + PS thật lấy lại 74 %, benign giả chỉ mất 10 %); head chịu lệch rất ít (nhiễu σ 0,05 z mất 44 % lợi); bộ sinh học được fit trên pool: GMM 8/32 +0,0044, MADE +0,0026 — **không đủ năng lực** (+0,008) | [capacity oracle](../research/methods/support-capacity-oracle-2026-10-22.md) |
| **P36 (22/10 14:20Z)** | owner "thử cả hai": (a) **Project48 + copula vs phương pháp hiện hành PASS hai seed** frozen-head (+0,0077 / +0,0051, worst +0,005 / −0,002); (b) GMM diag lớp tấn công **FAIL** (+0,0004 / +0,0008 so p48_cop, 3,3 MB EM) — đóng GMM. R1 stack cvp + cop ×1,63 / ×1,51 (15 / 50 round, copula fp32) ⇒ cần nén trước run dài | [two-seed](../research/methods/teacher-support-two-seed-2026-10-22.md) |
| **Vòng 26 (online, 22/10 19:15Z)** | **FAIL**: `cvp` + `cop` ep15 0,8249 vs xa 0,8246 (Δ +0,0003, benign +0,0053, worst +0,0026), 8,20 MB/client ×1,59; dấu hiệu (a): teacher hiệu chỉnh ≈ gốc từ ep5; posMirror +0,120 nhưng timeDelay −0,078, suddenStop −0,025 ⇒ harness frozen-head không dự báo online | [results §Vòng 26](../methods/full-drift/results.md) |
| **P37 (23/10, Kaggle)** | synthetic đổi trọng số private/proxy, full model, code production. **Không ứng viên** (tách BN / head-only < production ở ep6, ep15); oracle dòng thật chỉ +0,0034 / +0,0099 so production ⇒ cổng online +0,010 ở trần oracle; copula +0,002; BN trộn không hại; proxy không đổi. Kèm tài liệu tham khảo | [P37](../research/methods/synthetic-weight-dynamics-2026-10-23.md) |

**Bước tiếp:** P37 xong, theo luật khóa không có lần thử online nào đủ điều kiện ⇒ **owner quyết đóng hướng synthetic** (hoặc nêu hướng khác).
Không mở biến thể BN/head/cop/GMM/cv/xo. Backup Git/DVC: owner tự làm (mục "Backup 23/10" dưới).

Chạy lại P32 (local, ~10 phút, cùng weights/fixture P24–31):

```bash
cd methods/full-drift/builds/fd-probe/source/proj
W=../../artifacts/kaggle/fd-s42xa-20261015-r22-v1/runs/fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl_s42_e15/weights/round_015.pt
FX=../../../../../../archive/methods/shared-baselines/builds/pilot-baselines/artifacts/fixture
conda run -n nckh python ../../../../../../scripts/run_local_guarded.py python fd_probe32.py --weights $W --fixture $FX --out <out.json>
```

## Ràng buộc giữ nguyên

- **R1:** tổng byte/client ≤ 1,5× xa ở **cả 15 và 50 round**, ≤ 231.000 B/contact kể cả bootstrap; byte nhỏ của một carrier không chứng minh
  toàn phương pháp đạt (tính upload/downlink/header/ACK/crypto/chờ contact). Copula one-shot hiện ~1,9 MB/client — chưa nén.
- **R2** mixed architectures chưa kiểm; mọi model hiện là DAGSNet.
- Test20 chỉ báo cáo; không chọn HP/round/recipe bằng test. old/group là development set đã dùng nhiều lần.
- Candidate mới: cơ chế đổi → đối chứng → falsifier → cổng dừng, giữ benign ≥ −0,005, worst ≥ −0,010; online ep15 +0,010, s43/horizon riêng.
- Kaggle: owner dùng cho dự án khác ⇒ đọc quota live trước push; credentials account-scoped, không in key; notebook private 0600.
- Không Git/DVC push/backup/cleanup khi owner chưa yêu cầu. HEAD `f6ed8ed`; có thay đổi chưa commit của nhiều phiên — giữ nguyên.

## Việc tay

Không có. Lịch sử bàn giao: [22/10 report final + P29–31](../archive/operations/continuation-2026-10-22.md),
[synthetic trước compact 03Z](../archive/operations/continuation-synthetic-2026-10-22.md),
[snapshot trước đủ report](../archive/operations/continuation-before-final-report-2026-10-22.md),
[20–21/10](../archive/operations/continuation-2026-10-20-21.md). Snapshot lịch sử không phải trạng thái hiện tại.

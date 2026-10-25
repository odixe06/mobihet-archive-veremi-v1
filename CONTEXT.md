# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS kỹ thuật](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [handoff hiện tại](docs/operations/continue.md).
[Mục lục](docs/README.md) · [Phương pháp](docs/proposals.md) · [Index file](docs/file-index.md).

## Bài toán

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn; private model độc lập, thiết kế hỗ trợ client khác kiến trúc.
Giữ 66 feature, dataset/partition/scaler gốc; mọi model đang chạy là DAGSNet. Không raw/perturbed record,
per-record activation/gradient rời client. Heterogeneous R2 thực nghiệm và contact/deadline R1 chưa chứng minh.

## Trạng thái và quyết định hiện hành — 25/10/2026 11:32+07

- **Bàn giao sống duy nhất:** [continue](docs/operations/continue.md) — đang chạy gì, bước tiếp, lệnh, việc tay. Sửa tại chỗ;
  snapshot cũ vào [archive/operations](docs/archive/operations/README.md). Cách đọc/ghi context: skill `project-context`.
- **Đang chạy:** phiên Codex của owner copy dvc-storage Google Drive → OneDrive (remote DVC mặc định đã là `onedrive`, xem [backup](docs/operations/backup.md));
  **không sửa** phần trạng thái chuyển trong backup, skill repo-backup, `.dvc/config`, `scripts/dvc_*` khi phiên đó còn chạy. Không job local/Kaggle nào chạy.
- Phương pháp chốt: `gkd + la + kdpx`, proxy q3_zl (PASS seed42/43). [Kết quả/cổng](docs/methods/full-drift/results.md), [thiết kế](docs/methods/full-drift/design.md).
- **Run cuối seed42 xong:** 6 chuỗi (q3_zl la_kdpx + đối chứng gkd × 20/50/100 client) × 50 round, chấm toàn test20 mỗi round, headline r50,
  đã audit — [report](docs/methods/full-drift/results/final/report.md), [VERIFIED](docs/methods/full-drift/results/final/VERIFIED.json). Không repull/re-audit.
- **Kiểm chứng feature main seed42 ep15:** full66 `fa` và drop53 sửa `fg` đã pull + audit 25/10; control m4b `xa` đã audit trước đó.
  Macro-F1 private/head/group: 0,7728 / 0,8138 / 0,8246. [Kết quả và provenance](docs/methods/full-drift/results.md#kiểm-chứng-feature-trên-recipe-hiện-hành--seed42-ep15-audit-2510).
- **Feature sweep 25/10:** không còn kernel chạy; 18/18 arm có kết quả audited. 17 arm dùng ep15; theo quyết định owner,
  FedTGP-m4b `tc` dùng đúng nhãn ep14 vì hết cửa sổ sau prepare ~98 phút. Không cần rerun; so sánh FedTGP mang lệch horizon một round.
  [Số liệu + provenance](docs/methods/full-drift/results.md#kiểm-chứng-feature-baseline--audit-2510-1515-arm-dùng-được-một-arm-ep14).
- **Nghiên cứu synthetic (khoảng private ↔ proxy r50 ~0,024–0,028):** vòng 24 `xo` FAIL, vòng 25 `cvl` FAIL, P23–31 (teacher Project48 PASS nhưng
  transfer FAIL; joint carrier FAIL). **P32 22/10:** copula × Project48 +0,0024 (FAIL +0,005) nhưng worst-client +0,022; lợi support thật cần
  4 nhóm feature cùng thật (`snd_kin`, `sess`, `geo`, `prof`). **P33/P33b 22/10 (cách client học):** 71 % lợi support thật là recall posMirror +
  suddenStop vs benign ở receiver lạ; dòng own cân bằng lớp không thêm gì ⇒ cần biến thể client khác, mẫu phải nằm trên manifold đặc trưng private
  (Gaussian/copula precision ~0,47 vs real 0,86). **P34 FAIL** (atom chính xác + trạng thái first + định danh extractor +0,0023, không hơn
  copula; V4-S/G61 vẫn ≈ Gaussian). **P35:** lợi cần mẫu thật của chính lớp tấn công hiếm (benign giả không hại), head chịu lệch rất ít (σ 0,05 z
  mất 44 %); bộ sinh học được fit trên pool (GMM +0,0044, MADE +0,0026) không đủ năng lực. **P36 (owner: thử cả hai):** so **phương pháp hiện hành**,
  Project48 + copula PASS hai seed frozen-head (+0,0077 / +0,0051); GMM lớp tấn công FAIL. **Vòng 26 online (cvp + cop) FAIL** Δ +0,0003 (posMirror +0,12 nhưng timeDelay/suddenStop xuống;
  teacher hiệu chỉnh ≈ gốc từ ep5) ⇒ harness frozen-head không dự báo online; dừng nhánh. **Owner 23/10: thử một lần cuối** — **P37** (Kaggle, full model):
  tách BN / head-only cho synthetic thấp hơn production ⇒ không ứng viên; oracle dòng thật chỉ +0,010 ⇒ cổng online ở trần oracle. Chờ owner đóng hướng.
  [P37 + tài liệu tham khảo](docs/research/methods/synthetic-weight-dynamics-2026-10-23.md),
  [P36](docs/research/methods/teacher-support-two-seed-2026-10-22.md), [P35](docs/research/methods/support-capacity-oracle-2026-10-22.md), [P34](docs/research/methods/structured-support-generator-2026-10-22.md), [P33](docs/research/methods/client-learning-diagnosis-2026-10-22.md),
  [P32](docs/research/methods/synthetic-support-realism-2026-10-22.md),
  [P29–31](docs/research/methods/synthetic-gradient-diagnosis-2026-10-22.md).
- **R1 khóa:** ≤ 1,5× xa ở **cả 15/50 round**, ≤ 231.000 B/contact kể cả bootstrap; byte nhỏ của một carrier không chứng minh toàn protocol.
  R2 mixed architectures chưa kiểm. Test20 chỉ báo cáo, không chọn HP/round/recipe.
- Kaggle: owner dùng cho dự án khác ⇒ quota hụt bất thường là do dự án đó, đọc quota live trước push ([kaggle](docs/operations/kaggle.md)).
  Owner 23/10: dùng mọi account thoải mái nếu còn slot GPU và quota ≥ 1,15 × giờ session, không cần nhường dự án khác.
  `QUEUED` = chờ GPU; `RUNNING`/`COMPLETE`/W&B không chứng minh kết quả — collect + audit artifact. `datasets create` có thể báo lỗi dù ready.
- Freeze mọi source/notebook đã PUSHED; sửa bằng tag mới. Không nối adaptation FAIL y nguyên. Không Git/DVC push/backup/cleanup khi owner
  chưa yêu cầu. Credentials account-scoped, giữ global login thuyanh316; inline W&B chỉ ở private notebook 0600. Không rõ yêu cầu ⇒ hỏi owner.

## Phiên sau

1. Đọc [continue](docs/operations/continue.md); kiểm job còn chạy (bảng "Đang chạy") trước khi sửa file nào.
2. P37 FAIL theo luật khóa ([P37](docs/research/methods/synthetic-weight-dynamics-2026-10-23.md)) ⇒ owner quyết đóng hướng synthetic hoặc hướng khác. Không mở lại cvp/cop/GMM/cv/xo hay BN tính lại hậu kỳ.
3. Giữ mọi cổng cũ, seed/horizon/R1/R2 riêng.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Tiếp tục ngay: đang chạy gì, bước tiếp, lệnh, việc tay? | [Handoff](docs/operations/continue.md) |
| Synthetic/CCVR, khoảng private–proxy, client học gì từ mẫu, synthetic đổi trọng số thế nào, tài liệu tham khảo, build cũ không được lặp? | [P37](docs/research/methods/synthetic-weight-dynamics-2026-10-23.md), [P36](docs/research/methods/teacher-support-two-seed-2026-10-22.md), [P35](docs/research/methods/support-capacity-oracle-2026-10-22.md), [P34](docs/research/methods/structured-support-generator-2026-10-22.md), [P33](docs/research/methods/client-learning-diagnosis-2026-10-22.md), [P32](docs/research/methods/synthetic-support-realism-2026-10-22.md), [P29–31](docs/research/methods/synthetic-gradient-diagnosis-2026-10-22.md), [P23–28](docs/research/methods/synthetic-learning-recovery-2026-10-22.md), [veto build cũ](docs/research/methods/synthetic-build-history-audit-2026-10-22.md) |
| Train/chấm tách, timing, source/tests, cost? | [Pipeline cuối](docs/operations/final-pipeline.md) |
| Kết quả nào PASS/FAIL, vì sao chọn phương pháp, vòng 24 sinh mẫu có chủ đích? | [Results](docs/methods/full-drift/results.md), [design](docs/methods/full-drift/design.md), [danh mục](docs/proposals.md) |
| Protocol, thông tin rời client, so sánh/cost? | [Protocol](docs/research/protocol.md), [luồng FL](docs/research/federated-information-flow.md), [so sánh](docs/research/comparison-fl-dagsnet-veremi.md), [cost](docs/research/computational-cost.md) |
| Dataset/kiến trúc cố định? | [Knowledge](knowledge/README.md) |
| Account/quota, lỗi kernel, layout, Git/DVC? | [Kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md), [artifact index](docs/operations/artifact-index.md) |
| Quyết định/kết quả/bàn giao lịch sử? | [Archive](docs/archive/README.md), [bàn giao 22/10](docs/archive/operations/continuation-2026-10-22.md), [CONTEXT trước gọn tài liệu](docs/archive/operations/context-before-final-recovery-2026-10-20.md), [bàn giao17–20/10](docs/archive/operations/continuation-2026-10-17.md), [nhật ký 20–21/10](docs/archive/operations/continuation-2026-10-20-21.md) |

Hub giữ trạng thái/quyết định/route. Số đo và commands ở docs; dữ kiện cố định ở knowledge; procedure dùng lại ở skills.

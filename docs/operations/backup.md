# Git / DVC — backup và khôi phục

## Host hiện tại — kiểm push/pull 25/10/2026

- OneDrive được mount bằng **rclone VFS** tại `/home/odixe/Cloud/OneDrive`; không dùng OneDrive Desktop.
- Remote DVC mặc định `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/dvc-storage`.
- Đã thử DVC 3.67.1: push một object mới 92 byte, đọc trực tiếp bằng `rclone cat` từ cloud khớp nội dung,
  xóa cache của **repo thử riêng**, pull lại và so byte thành công.
  [Bằng chứng](registry/dvc-remote-check-2026-10-25.json); không sửa manifest hoặc đẩy toàn bộ dữ liệu dự án.
- Trước mỗi push kiểm mount: `findmnt --mountpoint /home/odixe/Cloud/OneDrive` phải báo `onedrive: fuse.rclone`.
  Nếu mount mất, dừng; không tạo thư mục thay thế vì có thể ghi nhầm vào ổ local.
- Mount dùng `--vfs-cache-mode full`; push thành công chỉ xác nhận ghi qua VFS. Với dữ liệu mới cần kiểm upload
  cloud riêng bằng rclone; lần thử nhỏ trên đã xác minh cloud thật. Không xóa VFS upload cache.
- Kiểm lại khi cần, từ gốc repo trong env `nckh`:

  ```bash
  python scripts/check_dvc_remote.py \
    --cloud-root onedrive:NCKH/veremi_build/mobihet/dvc-storage \
    --output /tmp/mobihet-dvc-remote-check.json
  ```

  Mỗi lần thử ghi một object nhỏ mới và giữ nó. Không coi phép thử nhỏ là audit toàn bộ backup lịch sử.
- Bundle mới dùng thư mục cha remote: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/mobihet.bundle`.
  Chính sách giữ ref bundle và checkpoint đã nghỉ ở dưới vẫn áp dụng. Chưa tạo/push bundle trong phiên thiết lập.

## MobiHet — trạng thái backup 24/10/2026 22:26 (UTC+7)

- **Đích push mới:** DVC mặc định `onedrive`, `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/dvc-storage`.
  Git vẫn GitHub; helper/skill `.agents` và `.claude` đã cập nhật. Nguồn Google Drive của MobiHet đã dọn sau xác minh.
- **Git:** source/docs/notebooks, manifest merge giữ entry remote-only và index phục hồi checkpoint tại `ed97aff` đã push.
  Hồ sơ này cập nhật bằng commit tài liệu riêng trên `main`; kiểm GitHub khớp HEAD sau push.
- **DVC:** 12 target ưu tiên / 4.651 object nguồn / 83,81 GB đã đạt kiểm cloud cuối; ghi mới 7,53 GB.
  `input.dvc` `07f88b88…dir` (3.029 entry), `kaggle.dvc` `29327f34…dir` (12.718 entry).
  Audit đủ 99 manifest / 31.161 object / 395,83 GB đạt size/QuickXor/AV lúc 01:40, có proof MD5 nguồn.
  Sau audit chủ động nghỉ thêm 32 ClassTail / 678.346.016 B, có Kaggle version cố định và MD5 toàn file;
  scope còn giữ trên OneDrive là 31.129 object. Không đẩy lại checkpoint chủ động nghỉ vì manifest lịch sử còn tham chiếu.
  Tổng kiểm cloud mới lúc 22:17 đạt 102.539 object MobiHet hiện tại/lịch sử, gồm đủ 31.129 object hiện tại cần giữ;
  từng object khớp size/QuickXor, không bị AV chặn, đối chiếu proof MD5 đã kiểm.
- **DVC lịch sử trên Google Drive:** nhóm chính 30.326 object / 337,43 GB đã kiểm cloud và dọn đủ nguồn G:.
  Nhóm còn lại 72.313 object / 109,76 GB đã kiểm cloud cuối toàn bộ và dọn đủ 72.313 ID nguồn file; gate nguồn và cloud kiểm lại lúc nối giám sát cuối.
  Chỉ xóa nguồn sau xác minh đúng ID/MD5 nguồn và cloud size/QuickXor/AV; tái dùng proof đã đạt, không phục hồi lại weights đã nghỉ.
- **Dọn local hoàn tất:** 122 prediction `.npz` / 44.091.802.287 B và 158 `.pt` / 45.196.692.006 B,
  tổng **280 file / 89.288.494.293 B** trong artifacts fd-probe. Nhóm `.pt` gồm 117 checkpoint trung gian đã chấm và 41 staging chấm đã dùng xong.
  Đối chiếu manifest đã push, provenance/AUDITED, hash local và cloud mới; không writer và stat không đổi trước unlink.
  Weights khớp SHA256 trong terminal evaluator AUDITED. Giữ 5 checkpoint r50 còn local, input/probe đang dùng, source/docs/config/metrics và manifest.
  Report sau dọn PASS: 6 scenario, 300 round, 1.800 dòng metric / 18.000 giá trị, 576 dòng per-class.
  Dung lượng filesystem WSL đã được giải phóng; thu hồi dung lượng vật lý C:/VHDX là bước riêng.
- **Checkpoint cũ đã nghỉ:** đợt archive 6.512 / 150,41 GB; 12 build đóng 8.830 / 289,72 GB; Build1–6 bổ sung 928 / 31,97 GB;
  ClassTail bổ sung 32 / 678,35 MB. Các đợt đã hoàn tất dọn theo phạm vi được duyệt; giữ 650 final hash bảo vệ và tham chiếu ngoài build đóng.
  886 cache/probe/preflight không có Kaggle được owner duyệt xóa riêng, không cam kết phục hồi.
  Index hiện tại có 8.862 object; 32 mới kiểm MD5 toàn file, 8.830 cũ GET+size đủ và mẫu MD5 từng notebook.
- **Index phục hồi đã push đúng repo:** Build1–6 `4864483` (928 object), Build7–8 `379ceed` (91), Build9–21 `ee89582` (5.535).
  Cả ba repo đã chuyển DVC mặc định sang cùng đích OneDrive; giữ nguyên manifest và JSON phục hồi.
  Đã thử `dvc get` một file từ manifest gốc tại mỗi commit mới, cả ba khớp MD5 toàn file trên OneDrive.
  Link Markdown/JSON/helper và lệnh pull ở mục [Khôi phục](#khôi-phục). Helper kiểm MD5 toàn file khi pull, từ chối nội dung khác.
  Bản index/helper/metadata dự phòng: `D:\OneDrive\NCKH\veremi_build\mobihet\archive-recovery.zip`, 9.571.637 B, đã kiểm SHA256 cloud;
  giữ các entry phục hồi trước đó.
- **Chốt nguồn MobiHet:** đã dọn đủ nguồn file và 260 thư mục rỗng trên G:; bản Git bundle cũ đã chuyển và kiểm trước khi dọn.
  `D:\OneDrive\NCKH\veremi_build\mobihet\mobihet.bundle` đã verify và kiểm cloud; giữ cả 12 ref tip của bundle G: cũ.
  Khi cập nhật bundle, giữ các ref `refs/backup/google-drive/*` bằng mirror riêng và so ref tip trước khi thay;
  `--all` từ checkout hiện tại đơn thuần có thể bỏ các ref chỉ còn trong bundle lịch sử.

Chỉ backup/push/dọn khi owner yêu cầu; yêu cầu backup không cho phép xóa local. Quy trình: skill
[repo-backup](../../.claude/skills/repo-backup/SKILL.md). Bảng mọi manifest và số file: [artifact-index](artifact-index.md).
Lịch sử từng đợt 14–20/10 (đã đẩy gì, đã xóa gì): [backup-history](../archive/operations/backup-history-2026-10-14-to-20.md).

## Đích

- Git: `git@github.com:odixe06/mobihet.git`, nhánh `main`. Kiểm `git ls-remote origin refs/heads/main` = `git rev-parse HEAD`.
- DVC remote mặc định `onedrive`: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/dvc-storage` (layout DVC 3 `files/md5`).
  Phạm vi MobiHet đã kiểm cloud ghi ở trạng thái đầu trang; checkpoint nghỉ được phục hồi qua index Kaggle.
- Bundle cho các lần backup sau: `/home/odixe/Cloud/OneDrive/NCKH/veremi_build/mobihet/mobihet.bundle` (`--all`, verify trước khi thay atomic).
  Giữ các ref lịch sử trong bundle hiện có khi cập nhật; nguồn bundle G: đã dọn sau xác minh.
- Repo lưu trữ cũ: `odixe06/mobihet-archive-b9-b21`, tag `archive-3` (giữ nguyên). Kiểm cloud sync OneDrive riêng với việc ghi vào thư mục D:.

## Đợt 21/10/2026 — lịch sử

- **21/10 ~17:30Z** (owner yêu cầu backup + dọn): DVC fd-probe qua `dvc_merge_push.py`:
  `kaggle.dvc` `381c1e32…dir` (10.777 entry, đẩy mới 71,56 GB), `input.dvc` `6ea020e5…dir` (2.644, 0 GB mới — trùng weights),
  `fixture-cache.dvc` `eb9eacad…dir` (23). `dvc status -c` 3 manifest: new/missing = 0, `deleted` 4.990 = chỉ thiếu cache local.
  Không đưa vào đợt này: hai output chấm pull dở/chưa audit (`fd-e03-12-s42zk02-…-f2` còn `.part`, `fd-e18-26-s42zi-…-f1`).
- Git `bcccbe5` (manifest + code/docs) rồi commit ghi dọn; local = remote (`git ls-remote`); bundle `--all` verify rồi thay trên Drive.
- **Dọn local 21/10 ~17:37Z** (owner yêu cầu; từng file kiểm MD5 có trên remote cùng size, weights kiểm sha256 = hợp đồng chấm đã audit):
  xóa 40 stage `artifacts/input/fd-e*` + `*-parent`, 190 predictions `.npz` của chấm đã audit, 178 weights train của round đã chấm audit;
  WSL dùng 221 → 112 GB. Giữ: weights round chưa chấm, eval JSON/metrics/AUDITED, `fd-src-*`, manifest. Ổ C: (VHDX) chưa trả lại — owner
  tự `sudo fstrim -v /` + `wsl --shutdown` + compact VHDX.
- `docs/methods/full-drift/results/per-class-f1.csv` 14 MB > 10 MiB vẫn ở Git như các commit trước; chuyển DVC nếu cần đúng quy tắc.
- Các output chờ pull/audit và manifest chưa merge tại mốc 21/10 đã được xử lý trong đợt hiện hành; xem trạng thái 24/10 đầu trang.

## Đẩy

1. Không `dvc add`: manifest ở gốc build + `artifacts/` bị git-ignore nên DVC 3 từ chối. Dùng
   `python scripts/dvc_merge_push.py <build>/<name>.dvc`: gộp `.dir` cũ theo relpath (file local ghi đè cùng relpath, giữ entry chỉ có trên remote),
   copy object tmp → MD5 → rename, ghi `.dvc` cuối cùng.
2. Kiểm `dvc status -c <manifest>`: **new/missing = 0** mới là đủ; `deleted` chỉ là thiếu cache local (đúng thiết kế sau khi dọn).
3. Git: không file > 10 MiB, quét credential trước commit; `private_push/` và runtime cache không lên Git/DVC.
   Commit không mang co-author/attribution Claude.
4. Bundle `--all` → `git bundle verify` → thay file tại đường dẫn OneDrive ở mục Đích.

## Khôi phục

- **Checkpoint đã dọn khỏi DVC/local nhưng có Kaggle:** chọn đúng repo và MD5 trong bảng; index chứa notebook/output, size và phiên nếu proof có lưu.
  Mỗi helper tải riêng object được chọn, kiểm size + MD5 toàn file trước khi ghi vào cache.

| Repo | Index notebook/output | JSON tra MD5 | Helper pull |
|---|---|---|---|
| MobiHet hiện tại — 8.862 object nghỉ | [Markdown](kaggle-retired-weights-index.md) | [JSON](kaggle-retired-weights.json) | [restore_kaggle_retired_weights.py](../../scripts/restore_kaggle_retired_weights.py) |
| Archive Build1–6 — 928 object | [Markdown](https://github.com/odixe06/mobihet-archive-b1-b6/blob/main/KAGGLE_OUTPUT_INDEX.md) | [JSON](https://github.com/odixe06/mobihet-archive-b1-b6/blob/main/KAGGLE_OUTPUT_INDEX.json) | [restore_kaggle_weight.py](https://github.com/odixe06/mobihet-archive-b1-b6/blob/main/scripts/restore_kaggle_weight.py) |
| Archive Build7–8 — 91 object | [Markdown](https://github.com/odixe06/mobihet-archive-b7-b8/blob/main/KAGGLE_OUTPUT_INDEX.md) | [JSON](https://github.com/odixe06/mobihet-archive-b7-b8/blob/main/KAGGLE_OUTPUT_INDEX.json) | [restore_kaggle_weight.py](https://github.com/odixe06/mobihet-archive-b7-b8/blob/main/scripts/restore_kaggle_weight.py) |
| Archive Build9–21 — 5.535 object | [Markdown](https://github.com/odixe06/mobihet-archive-b9-b21/blob/master/KAGGLE_OUTPUT_INDEX.md) | [JSON](https://github.com/odixe06/mobihet-archive-b9-b21/blob/master/KAGGLE_OUTPUT_INDEX.json) | [restore_kaggle_weight.py](https://github.com/odixe06/mobihet-archive-b9-b21/blob/master/scripts/restore_kaggle_weight.py) |

Chạy từ gốc repo cần khôi phục; thay `<owner>` bằng phần trước `/` của trường `kernel`, `<md5>` bằng hash trong JSON.
Helper account-scoped dùng từ checkout MobiHet; không đổi tài khoản Kaggle toàn cục.

```bash
MOBIHET=/home/odixe/project/mobihet
KAGGLE_AS="$MOBIHET/.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"

# Repo MobiHet hiện tại:
conda run -n nckh python "$KAGGLE_AS" <owner> -- python scripts/restore_kaggle_retired_weights.py --md5 <md5>

# Hoặc repo archive tương ứng, sau khi clone/cd vào đúng repo:
conda run -n nckh python "$KAGGLE_AS" <owner> -- python scripts/restore_kaggle_weight.py --md5 <md5>
```

Mặc định ghi `.dvc/cache/files/md5`; `--cache <path>` chọn cache khác. Chỉ `dvc checkout <manifest>` khi đã phục hồi đủ object cần thiết.
Không push lại nguyên build đã nghỉ. Index cũ có `version: null` sẽ dùng phiên Kaggle hiện tại; sai MD5/thiếu file thì dừng, không ghi đè cache.
886 cache/probe/preflight owner duyệt xóa không có Kaggle không nằm trong cam kết phục hồi này.

- `dvc pull <manifest>` theo [artifact-index](artifact-index.md); kiểm dung lượng đĩa trước. Không `dvc add` workspace partial sau khi pull một phần.
- Fresh checkout: lớp `.compat` theo [project-layout](project-layout.md#lớp-tương-thích-compat-và-fresh-checkout).
- Sau khi pull build cũ, tạo lại alias run canonical từ [registry/artifact-aliases.json](registry/artifact-aliases.json):

```bash
python3 - <<'PYALIAS'
from pathlib import Path
import json, os
for a in json.loads(Path('docs/operations/registry/artifact-aliases.json').read_text())['aliases']:
    p, target = Path(a['path']), Path(a['target'])
    if target.exists() and not p.exists() and not p.is_symlink():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(os.path.relpath(target, p.parent), target_is_directory=True)
PYALIAS
```

- Một số `runs.dvc` là snapshot rỗng lịch sử: pull `kaggle.dvc` cùng build rồi tạo alias, không dùng snapshot rỗng thay output Kaggle.
- Chỉ lấy metadata: `python scripts/restore_dvc_metadata.py <canonical.dvc> --proof <proof.json>` (vẫn partial, không `dvc add`).

## Dọn đĩa (chỉ khi owner yêu cầu)

Chỉ xóa phần đã có trên remote (new/missing = 0) và không cần cho round/collect tiếp theo; giữ code/docs/notebook/manifest/private_push.
Trả dung lượng về Windows (không bắt buộc): `sudo fstrim -v /`, PowerShell Administrator `wsl --shutdown`, rồi diskpart compact VHDX.

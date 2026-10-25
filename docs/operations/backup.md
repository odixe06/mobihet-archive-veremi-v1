# Git / DVC — backup và khôi phục

## OneDrive — trạng thái 24/10/2026 01:40 (UTC+7)

- **Push DVC mặc định đã chuyển sang `onedrive`**:
  `/mnt/d/OneDrive/NCKH/veremi_build/mobihet/dvc-storage`. Git vẫn GitHub; bundle `mobihet.bundle` cạnh `dvc-storage`.
  Helper/skill `.agents` và `.claude` đã cập nhật. Không push mới sang Google Drive.
- **Đang chuyển, chưa backup xong toàn dự án:** chỉ `NCKH` + `Lưu trữ` → `D:\OneDrive`, Files On-Demand.
  Nhóm MobiHet ưu tiên **30.326 object / 337,43 GB đã đạt kiểm cuối size/QuickXor/AV; đủ 30.326 ID nguồn G: đã dọn**.
  DVC còn lại 72.313 object / 109,76 GB: đủ bản local đã kiểm MD5; cloud đã xác minh khoảng 106,5 GB; đang nối kiểm cloud, không copy lại toàn bộ.
  Đã kiểm cloud cuối riêng 500 file nặng / 53,98 GB và dọn nguồn G: (500 HTTP204); chưa đánh dấu cả nhóm hoàn tất.
  Tiếp theo FL 73,38 GB; NCKH khác 47.209 file / 259,90 GB, gồm 142,26 GB Nextgen owner duyệt chuyển nốt nếu đủ quota.
  Gate quota thực tế đã đạt sau dọn checkpoint; chưa copy hai nhóm tiếp theo.
- **Đã dọn weights:** archive cũ 6.512 object / 150,41 GB (gồm 886 cache/probe/preflight được owner duyệt xóa);
  đợt mới **8.830 object / 289,72 GB thuộc 12 build đã đóng** hoàn tất: G: GET404, OneDrive cloud path404 và native absence đủ 8.830.
  Giữ checkpoint cuối từng run, input/smoke/reference, metrics/predictions/source/config và các tham chiếu còn dùng; không có payload đợt mới trong WSL để dọn.
  [Index checkpoint nghỉ](kaggle-retired-weights-index.md), [MD5 → kernel/version/output](kaggle-retired-weights.json),
  `scripts/restore_kaggle_retired_weights.py` pull version cố định và kiểm MD5 toàn file. Đủ 8.830 output đã kiểm truy cập/size;
  mỗi kernel có mẫu MD5 toàn file và smoke restore thực tế đạt. Không tuyên bố đã tải/hash toàn bộ 8.830 output Kaggle.
  Index/helper/manifest phục hồi đã vào `archive-recovery.zip` sẵn có trên OneDrive, kiểm SHA256 trước dọn.
- **Lưu trữ:** đủ 80.722 đích có proof nội dung + kiểm cloud cuối; 80.399 ID nguồn work đã dọn (~149,94 GB).
  Tái dùng bản OneDrive cũ, chỉ bổ sung thiếu/khác; giữ export Word/Excel + link, ánh xạ đổi tên, bản khác nội dung có hậu tố,
  và 5 ảnh mới trong `SnippingTool/Screenshots` (~843 KB). Đã dọn đủ 494 file bị AV chặn được owner duyệt trên cả hai bên.
- **337 file chia sẻ / 51,36 MB giữ nguồn cho owner xóa tay:** 310 ở Lưu trữ, 27 ở NCKH ngoài DVC.
  Tài khoản work đúng `minhtriet0502.work@gmail.com` không có quyền DELETE; 246 thuộc `minhtriet0502@gmail.com`, 91 thuộc bốn tài khoản khác.
  Owner 24/10: tự xóa nguồn sau xác minh khi tài khoản hiện tại có quyền; thiếu quyền thì giữ, ghi rõ vị trí, không gỡ shared parent.
  Danh sách chi tiết ngoài repo: `D:\Google-Drive-can-xoa-sau-20261024.csv` (310 nguồn Lưu trữ đã backup; thêm 27 NCKH khi kiểm xong).
  Không còn chờ quyết định/OAuth cá nhân để tiếp tục copy; các lỗi quota/mạng/hash vẫn phải xử lý, không tự coi là ngoại lệ quyền.
  Giữ các root ngoài phạm vi, shortcut và cây target không sở hữu. Không mirror từ G: sau khi đã dọn; dùng snapshot đã chốt.
- **Copy giữ 12 luồng**, trần 160 MiB/s, lô 4 GiB, backlog 24 GiB; giữ D: ≥20 GiB / C: ≥8 GiB.
  D: hiện khoảng 39,9 GiB; OneDrive cloud còn khoảng 474,8 GB khi nối phiên. Giới hạn Google Drive Desktop giữ nguyên theo owner.
  Mẫu mới 12 luồng: trung bình 27,9 MB/s/136 giây, 0 lỗi; 8 luồng 12,0 MB/s/250 giây; mẫu 4 luồng cũ 5,1 MB/s.
  Khác file/lô nên không coi là benchmark cùng dữ liệu. MD5 staging phải khớp nguồn trước publish; cloud size/QuickXor/AV mới phải đạt trước xóa đúng ID G:.
  Không tải lại toàn bộ payload cloud để tính MD5. Chỉ yêu cầu online-only sau cloud đạt. 48 kiểm thử runtime đạt.
- **Git/DVC MobiHet ưu tiên 24/10:** GitHub `main` đã nhận `9921c80`, remote SHA khớp; gồm code/docs/source/notebook và index checkpoint đã nghỉ.
  DVC 12 target hiện tại có 4.651 object nguồn duy nhất / 83,81 GB; ghi mới 7,53 GB, còn lại tái dùng bản OneDrive. Đủ nguồn và 3 `.dir` mới đã đạt kiểm cloud cuối size/QuickXor/AV.
  Gộp partial workspace giữ entry chỉ có trên remote: `input.dvc` 3.029 entry, `kaggle.dvc` 12.718; bỏ file `.tmp`/`.part`/`.partial` và `private_push`.
  Commit manifest/helper `c98d243`: `input.dvc` `07f88b88…dir`, `kaggle.dvc` `29327f34…dir`. Đủ **99 manifest / 31.161 object chưa nghỉ đã đạt kiểm cloud cuối size/QuickXor/AV**, gồm entry chỉ có trên remote; có proof MD5 nguồn cho đủ object.
  Copy G: và push DVC chạy đồng thời; 99 fingerprint đã đối chiếu và cập nhật cho HEAD mới, không đổi inventory ID nguồn G: đã chốt.
  Bundle mới sẽ công bố sau khi luồng cũ xử lý xong `NCKH/veremi_build/mobihet/mobihet.bundle` để tránh ghi đè bản mới bằng snapshot cũ.
- **Owner đã duyệt bước sau chuyển:** push toàn bộ thay đổi dự án qua Git/DVC OneDrive; kiểm Git remote = commit local,
  manifest/object DVC đủ (checkpoint chủ động nghỉ dùng Kaggle index), bundle `--all` verify và bản công bố khớp; ghi cloud sync riêng.
  Sau backup chỉ dọn **artifact/cache local đã xác minh và không còn job cần dùng**; giữ source và toàn bộ Markdown.
  Không đổi HEAD/manifest khi luồng chuyển còn khóa fingerprint; không đẩy lại checkpoint chủ động nghỉ chỉ vì manifest lịch sử còn tham chiếu.
- **Index phục hồi nằm trong đúng repo**, có JSON MD5 → notebook/output/aliases và helper kiểm MD5 toàn file trước xuất bản cache; đã thử pull thực tế thành công từng repo, remote SHA đã khớp:
  [Build1–6](https://github.com/odixe06/mobihet-archive-b1-b6/blob/main/KAGGLE_OUTPUT_INDEX.md) `583cd31` (928 object / 31,97 GB, phiên cố định, GET+size đủ và mẫu MD5 đủ 13 notebook);
  [Build7–8](https://github.com/odixe06/mobihet-archive-b7-b8/blob/main/KAGGLE_OUTPUT_INDEX.md) `fba9e46` (91 object);
  [Build9–21](https://github.com/odixe06/mobihet-archive-b9-b21/blob/master/KAGGLE_OUTPUT_INDEX.md) `e681b37` (5.535 object).
  Hai index cũ dùng proof GET+size 22/10, không lưu phiên ở kiểm gốc; helper lấy phiên hiện tại và từ chối nếu MD5 khác. 886 file local cũ không có Kaggle ghi riêng, không hứa khôi phục.
  Build1–6 rà 1.248 hash / 46,58 GB: giữ checkpoint cuối/prototype và 214 object / 12,83 GB chưa khớp rõ; 928 mới chưa xóa, chờ kiểm cloud cuối nhóm chuyển.
  Index/helper/manifest mới đã bổ sung vào `archive-recovery.zip` sẵn có; SHA256 cloud đạt, mọi entry phục hồi cũ giữ nguyên.
  [Index tổng hợp lịch sử](kaggle-archive-output-index.md) giữ thông tin kiểm cũ, index tại từng repo là nơi pull theo MD5.
  Google 24/10 ~01:34: vẫn vượt quota khoảng 266,98 GB; không suy nguồn chưa backup là được xóa. Thùng rác đã EMPTY_TRASH HTTP204 và query `trashed=true` rỗng; kiểm lại ở audit cuối nếu phát sinh.
  Object 1,42 GB native đã kiểm MD5 nhưng cloud 404 dù OneDrive báo synced đã đẩy riêng thành công, hash cloud đạt; nhóm DVC còn lại đủ 109,76 GB, đang kiểm cloud cuối 72.313 object.
  Copy FL 73,38 GB đã chạy, khoảng 18–54 MB/s theo lô; NCKH khác vẫn chờ sau đó. OneDrive cloud còn khoảng 450,54 GB lúc 01:39, đủ kế hoạch đã chọn; D: khoảng 74 GiB.
  Kiểm cloud cuối chạy song song copy nhóm tiếp theo; source cleanup chờ đúng future kiểm thành công của nhóm đó. 58 kiểm thử runtime đạt, gồm cổng chặn xóa khi kiểm cloud lỗi.
  Chưa đánh dấu toàn bộ công việc hoàn tất.
- Chỉ hồ sơ này giữ trạng thái chuyển; log/proof/script tạm tại `/home/odixe/.local/state/drive-migration-20261022/`, xóa sau hoàn tất/audit cuối.
  Worker systemd tự nối; gỡ startup unit tạm `mobihet-drive-migration-20261022.service` và trả power/WSL guards khi xong.
  **Việc tay:** giữ máy bật và mạng liên tục; chỉ xử lý nguồn thiếu quyền trong danh sách CSV khi cần. Các nguồn đủ quyền được tự dọn sau xác minh.

Chỉ backup/push/dọn khi owner yêu cầu; yêu cầu backup không cho phép xóa local. Quy trình: skill
[repo-backup](../../.claude/skills/repo-backup/SKILL.md). Bảng mọi manifest và số file: [artifact-index](artifact-index.md).
Lịch sử từng đợt 14–20/10 (đã đẩy gì, đã xóa gì): [backup-history](../archive/operations/backup-history-2026-10-14-to-20.md).

## Đích

- Git: `git@github.com:odixe06/mobihet.git`, nhánh `main`. Kiểm `git ls-remote origin refs/heads/main` = `git rev-parse HEAD`.
- DVC remote mặc định `onedrive`: `/mnt/d/OneDrive/NCKH/veremi_build/mobihet/dvc-storage` (layout DVC 3 `files/md5`).
  Dữ liệu cũ đang chuyển từ G:; kiểm object cụ thể trước khi gọi backup đầy đủ.
- Bundle cho các lần backup sau: `/mnt/d/OneDrive/NCKH/veremi_build/mobihet/mobihet.bundle` (`--all`, verify trước khi thay atomic).
  Bundle lịch sử trên G: vẫn giữ tới khi copy/kiểm xong.
- Repo lưu trữ cũ: `odixe06/mobihet-archive-b9-b21`, tag `archive-3` (giữ nguyên). Kiểm cloud sync OneDrive riêng với việc ghi vào thư mục D:.

## Lần gần nhất và phần còn lệch

- **25/10 feature sweep artifacts:** GitHub branch `codex/feature-validation-arms` nhận commit `e04d2ba` với 15
  `executed.redacted.ipynb`, collector audit và kết quả/handoff của FedAvg, FedProx, FedTGP, CanPerFL, Centralized.
  Bốn arm synthetic r39 `sn/sp/sq/sb` và code/report tương ứng không nằm trong commit. Raw artifact vẫn local/ignored;
  `kaggle.dvc` không đổi, chưa DVC push, chưa cleanup.
- **25/10 feature-validation branch:** GitHub branch `codex/feature-validation-arms` nhận commit `36f34d6` (code, notebooks,
  docs và kết quả audit). Hai artifact main `fa` full66 + `fg` drop53 hiện chỉ ở workspace local (~334 MB);
  `kaggle.dvc` chưa đổi, chưa DVC push, chưa cleanup.
- **21/10 ~17:30Z** (owner yêu cầu backup + dọn): DVC fd-probe qua `dvc_merge_push.py` có điều tiết (dừng khi ổ cache Drive D: < 15 GB trống;
  cache Drive for Desktop = `D:\ggdrive_cached`, `df` của G: phản ánh D:, không phải quota cloud; dvc-storage trên Drive 757 GB):
  `kaggle.dvc` `381c1e32…dir` (10.777 entry, đẩy mới 71,56 GB), `input.dvc` `6ea020e5…dir` (2.644, 0 GB mới — trùng weights),
  `fixture-cache.dvc` `eb9eacad…dir` (23). `dvc status -c` 3 manifest: new/missing = 0, `deleted` 4.990 = chỉ thiếu cache local.
  Không đưa vào đợt này: hai output chấm pull dở/chưa audit (`fd-e03-12-s42zk02-…-f2` còn `.part`, `fd-e18-26-s42zi-…-f1`).
- Git `bcccbe5` (manifest + code/docs) rồi commit ghi dọn; local = remote (`git ls-remote`); bundle `--all` verify rồi thay trên Drive.
- **Dọn local 21/10 ~17:37Z** (owner yêu cầu; từng file kiểm MD5 có trên remote cùng size, weights kiểm sha256 = hợp đồng chấm đã audit):
  xóa 40 stage `artifacts/input/fd-e*` + `*-parent`, 190 predictions `.npz` của chấm đã audit, 178 weights train của round đã chấm audit;
  WSL dùng 221 → 112 GB. Giữ: weights round chưa chấm, eval JSON/metrics/AUDITED, `fd-src-*`, manifest. Ổ C: (VHDX) chưa trả lại — owner
  tự `sudo fstrim -v /` + `wsl --shutdown` + compact VHDX.
- `docs/methods/full-drift/results/per-class-f1.csv` 14 MB > 10 MiB vẫn ở Git như các commit trước; chuyển DVC nếu cần đúng quy tắc.
- **Còn lại sau đợt này:** output chấm COMPLETE chưa pull (za r45–50, ze r9–28, ze r49–50, zk r3–12, zk r13–22, zi r1–8), zi r18–26 (pull đủ, chưa audit),
  train S2 zi03/zk04 và chấm sau đó ⇒ đợt sau merge-push `kaggle.dvc` + `input.dvc`.

## Đẩy

1. Không `dvc add`: manifest ở gốc build + `artifacts/` bị git-ignore nên DVC 3 từ chối. Dùng
   `python scripts/dvc_merge_push.py <build>/<name>.dvc`: gộp `.dir` cũ theo relpath (file local ghi đè cùng relpath, giữ entry chỉ có trên remote),
   copy object tmp → MD5 → rename, ghi `.dvc` cuối cùng.
2. Kiểm `dvc status -c <manifest>`: **new/missing = 0** mới là đủ; `deleted` chỉ là thiếu cache local (đúng thiết kế sau khi dọn).
3. Git: không file > 10 MiB, quét credential trước commit; `private_push/` và runtime cache không lên Git/DVC.
   Commit không mang co-author/attribution Claude.
4. Bundle `--all` → `git bundle verify` → thay file tại đường dẫn OneDrive ở mục Đích.

## Khôi phục

- Checkpoint cũ đã chuyển sang Kaggle-only: xem [index riêng](kaggle-retired-weights-index.md), khôi phục object cần dùng vào cache trước.
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

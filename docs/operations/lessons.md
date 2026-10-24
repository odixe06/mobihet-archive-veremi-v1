# Bài học vận hành MobiHet

Đọc cùng [LESSONS kỹ thuật](../../LESSONS.md) sau [CONTEXT](../../CONTEXT.md). Lỗi kernel cụ thể (triệu chứng → cách tránh) ở
[sổ lỗi kernel](notebook-errors.md); quy trình account/quota ở [kaggle](kaggle.md), Git/DVC ở [backup](backup.md).
Bài học ở đây là quy tắc rút ra khi chạy thí nghiệm và hạ tầng; bài học về phương pháp/train/đánh giá ghi ở LESSONS.

## Kaggle, notebook và đường pipeline

- **24/10 bỏ W&B khỏi notebook:** cell W&B cũ đồng thời import `os`, `subprocess`, `sys`; xóa nguyên cell làm setup lần lượt
  lỗi `NameError: os` rồi `NameError: sys`. Import runtime của notebook phải độc lập với integration tùy chọn; sau khi bỏ
  dependency, audit tên/import bằng execution hoặc startup log, không chỉ AST parse. Frozen source giữ tag; wrapper-only fix tăng suffix.
- **24/10 Windows multi-account:** mỗi lệnh `kaggle_as.py` với OAuth snapshot mint access token mới; poll dồn nhiều lệnh có thể chạm
  HTTP 429 ở `GenerateAccessToken` dù kernel vẫn chạy. Gộp status/log vào ít lần gọi, backoff thay vì poll liên tục; 429 không phải
  trạng thái kernel. Token cache trong snapshot không thay thế được token mới nếu đã hết hạn.

- **20/10 final startup crash:** test `--smoke --final` vẫn đọc pilot fixture, không chứng minh đường production final-only.
  Gaussian legacy còn hash cứng pilot manifest ⇒ r24 zc/zi crash trước round1. Test bắt buộc: không tồn tại pilot,
  Gaussian đọc final manifest; đối chiếu context pilot với legacy. Notebook phải lan truyền exit code subprocess,
  không biến `rc=1, epochs=0` thành Kaggle COMPLETE. Source đã push giữ nguyên; sửa ở tag mới.
- **20/10 upload source:** `datasets create` có thể exit1/JSON parse error dù server đã tạo dataset ready.
  Đọc status, verify manifest names/sizes trước retry; không upload/push trùng từ exit code đơn lẻ.
- **Sửa 20/10: CLI hiện tại đọc được log live bằng `kaggle kernels logs <owner/slug> --follow` (SSE).** Đã đọc W&B auth và
  `Prepared real pilot … old cut: 0` khi r24 vẫn RUNNING; lệnh không `--follow` có thể vẫn rỗng. Dùng account-scoped, đọc có timeout;
  exit 124 do hết thời gian đọc không phải kernel ERROR. Các ghi chú cũ "chỉ có W&B khi đang chạy" không còn đúng ở môi trường này.
  [Bằng chứng startup](../archive/operations/final-r24-2026-10-20/final-startup-2026-10-20.json). Log/status chưa thay terminal artifact audit.
- Fixture được fingerprint phải chính là fixture driver đọc. `--final` trước 20/10 chọn dữ liệu final ở Run nhưng main hash pilot; chưa có run final production. Dùng chung `fixture_path`, và evaluator từ chối manifest khác training. Smoke dùng pilot không phát hiện được sai đường dẫn production.
- Exact-resume fixture phải cover state của cả local/control và FL arm. CrossClientRank pooled replay PASS
  đã bỏ sót CE empty-byte state, khiến production verifier dừng dù r3 hợp lệ;
  [repro và recovery](notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.

## Collect, poll và máy local

- **25/10 collector phải theo layout và config của artifact, không theo giả định build cũ:** `collect_fd.py` cũ hard-code package
  ở repo root, DAGSNet 66 feature và cell W&B bắt buộc; vì vậy không thể audit r37 W&B-free/drop53 dù run hợp lệ. Import package
  từ `.compat`, cho W&B key optional, rebuild từ `weights['architecture']`, và chạy Kaggle client/PyTorch bằng đúng interpreter riêng.
  Thiếu `PUSHED` chỉ được dựng lại sau status account-scoped; vẫn bắt buộc tải executed notebook và so source.
- Máy local ngủ (WSL) làm poll nền dừng, Kaggle vẫn chạy. Sau khi thức đừng để watcher mở collector cho mọi session cùng
  lúc (mỗi audit đỉnh ~4 GB): chia nhóm session rời nhau cho 2–4 tiến trình `collect_fd.py --only`, pull ~20 s/session.
- Collector có thể thoát 0 mà chưa pull nếu thiếu `PUSHED`. Kiểm `AUDITED.json` và completion cụ thể; chỉ dựng lại marker sau khi
  xác nhận đúng kernel từ status account-scoped, rồi vẫn đối chiếu executed source trước audit (ya-v2, 20/10).
- **Không tạm dừng (SIGSTOP) tiến trình pull; giới hạn audit song song theo RAM còn trống.** 22/10: dừng pull để ưu tiên kernel khác làm đứt
  stream (`ChunkedEncodingError`); cùng lúc audit lớn (~2,6–4 GB mỗi audit 100 client) với tác vụ khác của owner đẩy RAM trống còn ~600 MiB
  và bộ canh RAM dừng hai collector. Ưu tiên bằng thứ tự hàng đợi, tối đa 2 audit cùng lúc trên máy 10 GB; pull nối tiếp được từ `.part`.

## Git, DVC và lưu trữ cloud

- SHA của `.git/index` khác chưa có nghĩa nội dung staged đổi. Kiểm staged tree và HEAD riêng; ghi cả byte identity và content identity,
  không khôi phục index chỉ để ép SHA trùng snapshot cũ (20/10).

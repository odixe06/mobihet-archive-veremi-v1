# Máy local — Lubuntu / LXQt, kiểm tra ngày 2026-10-26

Thông tin phần cứng/runtime được đo lại trên máy hiện tại ngày 2026-10-26.

## Cấu hình đã kiểm tra

| Thành phần | Giá trị |
|---|---|
| OS | Lubuntu 26.04.1 LTS (nền Ubuntu), desktop LXQt; kernel 7.0.0-38-generic, x86_64; Linux native |
| CPU | Intel Core i5-12500H, 16 luồng khả dụng |
| RAM | 16.087.364 KiB ≈ 15,34 GiB tổng (khoảng 15 GB theo cách gọi của owner) |
| Swap | `/swapfile`, 8.388.604 KiB ≈ 8 GiB; không tính vào RAM vật lý |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute capability sm_86 |
| Driver NVIDIA | 595.99.02; `nvidia-smi` báo hỗ trợ CUDA tối đa 13.2 |
| Conda | `/home/odixe/Tools/miniconda3`, cài đặt hiện tại |
| Checkout | `/home/odixe/project/mobihet` |
| Env | `nckh`, Python 3.12.15; khai báo trong [environment.yml](../environment.yml) |
| Dataset local ưu tiên | `/run/media/odixe/ORICO/nextgen`, ổ ORICO mount exFAT tại `/run/media/odixe/ORICO` |

CUDA Version trong `nvidia-smi` là khả năng của driver, không phải xác nhận CUDA Toolkit hoặc runtime PyTorch.
`/etc/os-release` báo Ubuntu là hệ điều hành nền; session `Lubuntu`, `XDG_CURRENT_DESKTOP=LXQt`,
`lubuntu-default-settings` và `lxqt-session` xác nhận môi trường Lubuntu/LXQt.
Đường dẫn `/mnt/*`, Miniforge và mô tả WSL 8 GB trong hồ sơ cũ không phải môi trường hiện tại.

Dataset dùng ổ local ORICO theo owner ngày 26/10; không dùng bản dataset trên OneDrive.
Remote backup DVC vẫn dùng rclone OneDrive theo [backup](../docs/operations/backup.md), độc lập với nguồn dataset.
Kiểm `findmnt -T /run/media/odixe/ORICO/nextgen` và tồn tại file trước khi đọc; ổ tháo rời có thể chưa mount.

## Runtime local đã kiểm tra

PyTorch **2.13.0+cu130**, CUDA runtime **13.0**, cuDNN **9.20**; `torch.cuda.is_available()` = True.
Phép nhân ma trận CUDA và DAGSNet 395.024 tham số forward/backward (batch 4) trên RTX 3050 đã kiểm ngày 25/10, kết quả hữu hạn.
Ngày 26/10 kiểm lại phiên bản package, CUDA availability và thông số GPU; không chạy lại benchmark/train.
Phiên bản chi tiết: [local-runtime.json](local-runtime.json).
Chưa có benchmark hiệu năng mới; không suy thời gian train từ smoke test.

Kaggle CLI cập nhật riêng ngày 2026-10-27: **2.2.5**, `kagglesdk` giữ **0.1.38**;
`pip check` PASS, package khác không đổi. Pin tại `environment.yml`; hardware/CUDA
ở trên vẫn là số đo ngày 26/10. Quy trình tương thích: [Kaggle](../docs/operations/kaggle.md).

## Chạy local và Kaggle

Theo chủ dự án ngày 2026-10-25: task ngắn chạy local, tận dụng GPU/cấu hình máy;
task kéo dài vài giờ chạy Kaggle. Ước lượng thời gian/RAM trước khi bắt đầu.
Local phù hợp test code, forward/backward và probe vài phút trên fixture nhỏ.
Multi-hour train/diagnostics, 2-worker/2-GPU và kiểm chứng sm_75 chạy Kaggle.
Không coi tốc độ local, compile sm_86 hoặc một GPU là bằng chứng hiệu năng/đúng trên 2×T4.

```bash
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh
conda activate nckh
cd /home/odixe/project/mobihet
python scripts/run_local_guarded.py python <script> <args>
```

Mỗi lần chỉ chạy một cây test nặng. Watchdog hiện tại dừng khi toàn máy còn dưới 700 MiB MemAvailable;
không tự nới ngưỡng khi bị dừng. Giảm fixture/batch, đóng ứng dụng theo quyết định owner
hoặc chuyển job lên Kaggle.
Giữ ngưỡng watchdog đang được dự án sử dụng.

RAM/VRAM/đĩa còn trống thay đổi theo thời gian, luôn đo lại:

```bash
free -h
swapon --show
nvidia-smi --query-gpu=memory.free --format=csv
```

Không nạp nguyên split 43 triệu dòng: stream parquet theo batch/row-group, đặt giới hạn mẫu rõ ràng.
GPU 4 GiB không chứa được train fp16 ~5,3 GiB; batch lớn, nhiều model eval và compile có thể OOM.
Khi cần tương thích T4 dùng AMP float16; không suy hỗ trợ BF16/TF32 của Ampere sang Turing.
Dữ liệu/checkpoint DVC không được tải cùng Git clone; kiểm tồn tại trước probe thật.

## Python và compatibility aliases

Dùng interpreter của `nckh` cho Python và Kaggle CLI. Với heredoc qua `conda run`, thêm
`--no-capture-output` khi cần quan sát stdout; có thể chạy trực tiếp interpreter của env.
Alias tương thích của repo được dựng lại bằng `python scripts/restore_compat_aliases.py`.
File `mobihet-compat.pth` trong site-packages của env trỏ đến `.compat` của checkout hiện tại.
Giữ nguyên pinned image Kaggle trong `knowledge/runtime.json`; môi trường local là môi trường khác.

## Dựng lại môi trường

```bash
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh
cd /home/odixe/project/mobihet
conda env create -f environment.yml
conda activate nckh
python scripts/restore_compat_aliases.py
python -c "import sysconfig; from pathlib import Path; Path(sysconfig.get_paths()['purelib'], 'mobihet-compat.pth').write_text(str(Path.cwd() / '.compat') + chr(10))"
```

Lệnh tạo env dành cho máy chưa có `nckh`; không chạy lại để ghi đè env đang dùng.

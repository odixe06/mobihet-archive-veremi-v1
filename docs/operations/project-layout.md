# Cấu trúc repo, code chung và tên thí nghiệm

Một file cho mọi câu hỏi "file/code này nằm đâu, tên cũ là gì". Máy đọc: [registry/](registry/) (map layout, alias, kiểm package 14/10).
Git/DVC và khôi phục artifact: [backup](backup.md) · [chỉ mục manifest](artifact-index.md).

## Cây thư mục

```text
core/                              code Python chung, tên theo nhiệm vụ
methods/<method>/builds/<build>/    build đang làm
  source/                          implementation/snapshot frozen
  notebooks/                       source/config/version đã push (public, không key)
  private_push/                    notebook mang key W&B, git-ignored, mode 0600
  scripts/                         generator/collector/report của build
  probes/                          probe/calibration/fixture proof
  artifacts/                       input/output/checkpoint — git-ignored, theo dõi bằng *.dvc
  *.dvc                            manifest DVC (md5/size/nfiles)
archive/methods/<method>/builds/    build đã kết thúc, cùng cấu trúc
archive/tools/, archive/logs/      helper/migration và log lịch sử
docs/                              README (mục lục), proposals, methods/, research/, lessons/, operations/, archive/
scripts/                           công cụ chung đang dùng
knowledge/                         dữ liệu/kiến trúc/runtime cố định
.compat/                           layout cũ (ẩn, không Git): mh/ (40 hardlink tới core), papers/, knowledge, logs
```

Tên build mô tả cơ chế/phạm vi; nhãn V2–V10 chỉ là provenance (bảng cuối trang). Build kết thúc → `archive/methods/`,
thiết kế/kết quả → `docs/archive/`, bài học dùng tiếp → `LESSONS.md` / `docs/lessons/`. Archive không phủ định cả họ phương pháp;
file archive còn trong source fingerprint của snapshot cũ thì không xóa, không đổi byte.

## Lớp tương thích `.compat` và fresh checkout

- Python local tìm `mh` qua `/home/odixe/Tools/miniconda3/envs/nckh/lib/python3.12/site-packages/mobihet-compat.pth` (một dòng: đường dẫn `.compat`).
  `.compat/mh/*.py` là hardlink tới `core/` ⇒ byte/SHA 40 module không đổi, `mh.pilot.ROOT = .compat`.
- Script local dùng `ROOT/'.compat/papers/...'`, `ROOT/'.compat/mh'`; đường dẫn Kaggle (`WORK/'papers/...'`) giữ nguyên.
  Build archive không sửa: chạy lại cần tạm `ln -s .compat/papers papers`.
- Fresh checkout: `python3 scripts/restore_compat_aliases.py` (symlink + hardlink từ [map](registry/compatibility-aliases.json),
  không ghi đè source, không pull data), rồi tạo lại file `.pth` trong env `nckh`.
- Cấu hình Lubuntu/LXQt, RAM/swap/GPU, env và đường dẫn dataset local hiện tại: [local-env](../../knowledge/local-env.md).
- Source frozen giữ byte và tên file gốc khi dựng runtime; tên canonical có thể khác tên import. Không sửa scientific code để che lỗi đường dẫn.
- Di chuyển tài liệu: `python scripts/move_docs.py moves.json [--dry]` (mv + sửa mọi link Markdown tương đối; chuỗi đường dẫn trong script phải sửa tay).

## Code chung và tên cũ `mh/*`

Code chung vật lý nằm trong `core/`; implementation riêng nằm trong build active hoặc archive. `mh/` chỉ là facade để snapshot cũ
import và hash đúng 40 tên file; không có các phiên bản thuật toán V2/V3/V4 cùng chạy trong core. 19 module chung, 1 module Hybrid
full build dùng trực tiếp, 20 implementation của build đã kết thúc. Namespace `scripts` tìm helper lịch sử trong `.compat/scripts`;
helper mới dùng `scripts.project_paths.current_path` khi đọc đường dẫn cũ. [Core](../../core/README.md).

### Chung toàn dự án

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/__init__.py` | [__init__.py](../../core/__init__.py) |
| `mh/archs.py` | [legacy_model_factory.py](../../core/legacy_model_factory.py) |
| `mh/channel.py` | [rsu_contact.py](../../core/rsu_contact.py) |
| `mh/dagsnet.py` | [dagsnet.py](../../core/dagsnet.py) |
| `mh/metrics.py` | [classification_metrics.py](../../core/classification_metrics.py) |
| `mh/mobility.py` | [vehicle_mobility.py](../../core/vehicle_mobility.py) |
| `mh/model_packet.py` | [model_codec.py](../../core/model_codec.py) |
| `mh/new_fl.py` | [artifact_driver.py](../../core/artifact_driver.py) |
| `mh/pilot.py` | [pilot_runtime.py](../../core/pilot_runtime.py) |
| `mh/prepare.py` | [prepare_data.py](../../core/prepare_data.py) |
| `mh/teacher_base.py` | [teacher_training.py](../../core/teacher_training.py) |
| `mh/teacher_driver.py` | [teacher_driver.py](../../core/teacher_driver.py) |
| `mh/v2_math.py` | [vector_codecs.py](../../core/vector_codecs.py) |
| `mh/v2_methods.py` | [shared_prior_and_proxy.py](../../core/shared_prior_and_proxy.py) |
| `mh/v2_run.py` | [federated_driver.py](../../core/federated_driver.py) |
| `mh/v4_schema.py` | [feature_support_schema.py](../../core/feature_support_schema.py) |
| `mh/v6_run.py` | [expanded_fixture_runtime.py](../../core/expanded_fixture_runtime.py) |
| `mh/v7_basis.py` | [covariance_basis.py](../../core/covariance_basis.py) |
| `mh/v7_nystrom.py` | [gaussian_moments.py](../../core/gaussian_moments.py) |

### Riêng build đang làm

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/v8_hybrid.py` | [hybrid_models.py](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/source/mh/hybrid_models.py) |

### Implementation lưu trữ

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/new_methods.py` | [distribution_matching.py](../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/source/mh/distribution_matching.py) |
| `mh/p2.py` | [fragmented_snapshot_driver.py](../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/source/mh/fragmented_snapshot_driver.py) |
| `mh/v10_energy.py` | [gaussian_base_energy.py](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/source/mh/gaussian_base_energy.py) |
| `mh/v10_run.py` | [joint_energy_driver.py](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/source/mh/joint_energy_driver.py) |
| `mh/v3_dense.py` | [dense_proxy_control.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/dense_proxy_control.py) |
| `mh/v3_math.py` | [proxy_carrier_codecs.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_codecs.py) |
| `mh/v3_methods.py` | [proxy_carrier_methods.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_methods.py) |
| `mh/v3_run.py` | [proxy_carrier_driver.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_driver.py) |
| `mh/v4_mala.py` | [mala_transport.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/mala_transport.py) |
| `mh/v4_run.py` | [support_mala_driver.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/support_mala_driver.py) |
| `mh/v4_support.py` | [support_conditioned_gaussian.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/support_conditioned_gaussian.py) |
| `mh/v5_anchor.py` | [kernel_anchor_targets.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/kernel_anchor_targets.py) |
| `mh/v5_books.py` | [gradient_latent_books.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/gradient_latent_books.py) |
| `mh/v5_run.py` | [synthetic_books_driver.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/synthetic_books_driver.py) |
| `mh/v6_geometry.py` | [geometry_moments.py](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/source/mh/geometry_moments.py) |
| `mh/v6_schema.py` | [geometry_feature_schema.py](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/source/mh/geometry_feature_schema.py) |
| `mh/v7_run.py` | [covariance_sketch_driver.py](../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/source/mh/covariance_sketch_driver.py) |
| `mh/v8_run.py` | [hybrid_pilot_driver.py](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/source/mh/hybrid_pilot_driver.py) |
| `mh/v9_anchor.py` | [anchor_residual_teacher.py](../../archive/methods/anchor-function-residual/builds/anchor-residual/source/mh/anchor_residual_teacher.py) |
| `mh/v9_run.py` | [anchor_residual_driver.py](../../archive/methods/anchor-function-residual/builds/anchor-residual/source/mh/anchor_residual_driver.py) |

## Tên đợt thí nghiệm V2–V10

V2–V10 là **đợt kiểm nghiệm**, không phải chuỗi phiên bản của một thuật toán: một đợt có thể so nhiều cơ chế, đợt sau có thể dùng lại
driver/codec của đợt trước. Không suy V4 thay V3.

| nhãn lịch sử | tên build hiện tại | nội dung |
|---|---|---|
| V2 | [countsketch-subspace-ratio](../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/README.md) | CountSketch, seeded subspace, density ratio |
| V3 | [proxy-carriers](../../archive/methods/proxy-compression/builds/proxy-carriers/README.md) | Coordinate EF, rolling subnetwork, orthogonal projection, PowerSGD |
| V4 | [support-mala](../../archive/methods/gaussian-sampling/builds/support-mala/README.md) | Support-conditioned Gaussian và MALA |
| V5 | [anchor-synthetic-books](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/README.md) | Kernel anchors, gradient/latent synthetic books |
| V6 | [geometry-g59-g61](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/README.md) | Geometry primitive G59/G61 |
| V7 | [nystrom-rank4-rank8](../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/README.md) | Covariance Nyström rank4/8 |
| V8 | [hybrid-pilot](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/README.md) | Gaussian + DML hybrid pilot |
| V9 | [anchor-residual](../../archive/methods/anchor-function-residual/builds/anchor-residual/README.md) | Anchor-function residual teacher |
| V10 | [gaussian-base-energy](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/README.md) | Gaussian-base joint energy/MH |

## Khi thêm/đóng build

1. Build mới đặt tên theo cơ chế/phạm vi; thiết kế/kết quả ở `docs/methods/<method>/`; lỗi vận hành ở [sổ lỗi](notebook-errors.md).
2. Build đóng: kiểm dependency, chuyển archive, giữ bài học; không sửa source/notebook đã push.
3. Sau khi chuyển file: `python scripts/update_file_index.py`, `conda run -n nckh python scripts/check_package_layout.py`,
   `python scripts/check_project_layout.py` (còn mismatch DVC migration 14/10 đã biết, không sửa DVC để làm xanh).
4. Skill theo loại việc: bảng ở `CLAUDE.md` §7; `.claude/skills` là bản chính, `.agents/skills` là bản Codex đồng nội dung.

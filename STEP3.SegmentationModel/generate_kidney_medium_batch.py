# -*- coding: utf-8 -*-
"""
批量生成 HealthyCT 肾脏 CT 中型肿瘤（DiffTumor）
运行目录：STEP3.SegmentationModel
输出：
  kidney_medium_batch/
    nifti/       每例的生成 CT、生成标签、输入 CT
    preview/     每例一张自动切片预览图
    batch_preview.png  前 12 例拼图
    batch_log.csv     成功/失败日志

说明：
- 当前输出 NIfTI 使用占位 affine（单位矩阵），适合数组查看与同数组叠加检查；
  不要直接用于物理空间测量或论文定量分析。正式使用前需补齐正确的空间元数据。
"""

import csv
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.transforms import Compose

from main import _get_transform
from TumorGeneration.utils import synt_model_prepare, synthesize_medium_tumor


# =========================
# 1. 路径和生成参数
# =========================
DATA_ROOT = Path(r"E:\thesisAndCode\data\HealthyCT\extracted\healthy_ct")
CT_DIR = DATA_ROOT / "CT" / "kidney_ct"
LABEL_DIR = DATA_ROOT / "label" / "kidney_label"

OUT_DIR = Path("kidney_medium_batch")
NIFTI_DIR = OUT_DIR / "nifti"
PREVIEW_DIR = OUT_DIR / "preview"

# 每例生成 1 个中型肿瘤；设为 None 表示处理全部匹配病例
MAX_CASES = None
DDIM_STEPS = 50
PREVIEW_CASES = 12
SEED = 20261023

NIFTI_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 兼容较新 PyTorch 的 checkpoint 加载行为
# 只对可信的本地官方模型权重使用
# =========================
_original_torch_load = torch.load
def compatible_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = compatible_torch_load


def save_nifti_array(array, path):
    """数组查看用保存函数；此处 affine 是占位值，不用于物理空间测量。"""
    array = np.asarray(array)
    nii = nib.Nifti1Image(array, np.eye(4))
    nib.save(nii, str(path))


def make_center_crop(image, label, crop_size=96):
    """
    image/label shape: (1, X, Y, Z)
    以器官前景质心为中心裁剪；如果尺寸不足，先在右侧补到 crop_size。
    """
    image = np.asarray(image)
    label = np.asarray(label)

    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(f"预处理后维度不符合预期：image={image.shape}, label={label.shape}")

    # 标签在 LoadImage_train 中应已将非背景映射为 1
    foreground = np.argwhere(label[0] > 0)
    if foreground.size == 0:
        raise ValueError("当前病例的肾脏掩码没有前景体素")

    center = np.round(foreground.mean(axis=0)).astype(int)

    # 保证三个空间维度至少为 crop_size
    pads = [(0, 0)]
    for dim in image.shape[1:]:
        pads.append((0, max(0, crop_size - dim)))
    image = np.pad(image, pads, mode="constant", constant_values=0)
    label = np.pad(label, pads, mode="constant", constant_values=0)

    spatial_shape = np.array(image.shape[1:])
    center = np.minimum(center, spatial_shape - 1)
    starts = center - crop_size // 2
    starts = np.maximum(starts, 0)
    starts = np.minimum(starts, spatial_shape - crop_size)
    starts = starts.astype(int)

    x, y, z = starts.tolist()
    image_crop = image[:, x:x+crop_size, y:y+crop_size, z:z+crop_size]
    label_crop = label[:, x:x+crop_size, y:y+crop_size, z:z+crop_size]

    if image_crop.shape != (1, crop_size, crop_size, crop_size):
        raise ValueError(f"CT crop 尺寸异常：{image_crop.shape}")
    if label_crop.shape != (1, crop_size, crop_size, crop_size):
        raise ValueError(f"label crop 尺寸异常：{label_crop.shape}")

    return image_crop, label_crop


def save_preview(input_ct, generated_ct, generated_mask, out_png, case_name):
    """自动选择肿瘤体素最多的轴向切片，并保存三联图。"""
    ct_in = np.asarray(input_ct).squeeze()
    ct_gen = np.asarray(generated_ct).squeeze()
    mask = np.asarray(generated_mask).squeeze()

    tumor_counts = np.sum(mask == 2, axis=(0, 1))
    if tumor_counts.max() > 0:
        z = int(np.argmax(tumor_counts))
    else:
        z = ct_gen.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ct_in[:, :, z], cmap="gray", origin="lower")
    axes[0].set_title("Input CT")
    axes[1].imshow(ct_gen[:, :, z], cmap="gray", origin="lower")
    axes[1].set_title("Generated CT")
    axes[2].imshow(ct_gen[:, :, z], cmap="gray", origin="lower")
    tumor = np.ma.masked_where(mask[:, :, z] != 2, mask[:, :, z])
    axes[2].imshow(tumor, cmap="autumn", alpha=0.8, origin="lower", vmin=2, vmax=2)
    axes[2].set_title(f"Tumor overlay (z={z})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(case_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return z, int(tumor_counts[z])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    if not CT_DIR.exists():
        raise FileNotFoundError(f"找不到 CT 目录：{CT_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"找不到 label 目录：{LABEL_DIR}")

    ct_files = sorted(CT_DIR.glob("*.nii.gz"))
    label_map = {p.name: p for p in LABEL_DIR.glob("*.nii.gz")}
    pairs = [(ct, label_map[ct.name]) for ct in ct_files if ct.name in label_map]

    print(f"CT 文件数：{len(ct_files)}")
    print(f"能按文件名匹配的 CT-label 对数：{len(pairs)}")
    if not pairs:
        raise RuntimeError("没有找到同名 CT/label 文件对，请检查两个目录的文件命名。")

    if MAX_CASES is not None:
        pairs = pairs[:MAX_CASES]
    print(f"本次计划处理：{len(pairs)} 例")
    print(f"输出目录：{OUT_DIR.resolve()}")

    # 使用训练 transform 的前 6 步：加载、加通道、RAS、重采样、强度归一化、空间补齐。
    args = SimpleNamespace(organ_type="kidney", fg_thresh=0.5)
    train_transform, _ = _get_transform(args)
    pre_transform = Compose(train_transform.transforms[:6])

    print("\n===== 加载 Kidney DiffTumor 模型（只加载一次） =====")
    vqgan, early_tester, noearly_sampler = synt_model_prepare(
        device=device,
        vqgan_ckpt="TumorGeneration/model_weight/AutoencoderModel.ckpt",
        diffusion_ckpt="TumorGeneration/model_weight/",
        fold=0,
        organ="kidney",
    )
    print("模型加载完成，开始批量生成。")

    log_path = OUT_DIR / "batch_log.csv"
    fields = ["case", "status", "tumor_voxels", "tumor_fraction_percent", "preview_png", "error"]
    preview_records = []
    success_count = 0
    fail_count = 0

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, (ct_path, label_path) in enumerate(pairs, start=1):
            case = ct_path.name.replace(".nii.gz", "")
            print(f"\n[{i}/{len(pairs)}] 正在处理：{case}")

            try:
                # 每例使用不同但可复现的随机种子
                case_seed = SEED + i
                torch.manual_seed(case_seed)
                np.random.seed(case_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(case_seed)

                sample = {
                    "image": str(ct_path),
                    "label": str(label_path),
                    "name": case,
                }
                data = pre_transform(sample)
                image = np.asarray(data["image"], dtype=np.float32)
                label = np.asarray(data["label"], dtype=np.float32)

                # 与此前已通过的固定中心裁剪测试一致
                image_crop, label_crop = make_center_crop(image, label, crop_size=96)
                image_crop = np.clip(image_crop, 0.0, 1.0)
                label_crop = (label_crop > 0).astype(np.float32)

                ct_tensor = torch.from_numpy(image_crop.copy()).unsqueeze(0).to(device)   # (1,1,96,96,96)
                mask_tensor = torch.from_numpy(label_crop.copy()).unsqueeze(0).to(device) # (1,1,96,96,96)

                with torch.no_grad():
                    generated_ct, generated_mask = synthesize_medium_tumor(
                        ct_tensor,
                        mask_tensor,
                        "kidney",
                        vqgan,
                        noearly_sampler,
                        ddim_ts=DDIM_STEPS,
                    )

                gen_ct_np = generated_ct.detach().float().cpu().numpy().squeeze().astype(np.float32)
                gen_mask_np = generated_mask.detach().float().cpu().numpy().squeeze().astype(np.uint8)
                input_ct_np = ct_tensor.detach().float().cpu().numpy().squeeze().astype(np.float32)

                tumor_voxels = int(np.sum(gen_mask_np == 2))
                tumor_fraction = 100.0 * tumor_voxels / gen_mask_np.size

                # 每例单独保存，方便断点续跑后检查结果
                case_dir = NIFTI_DIR / case
                case_dir.mkdir(parents=True, exist_ok=True)
                save_nifti_array(input_ct_np, case_dir / "input_ct.nii.gz")
                save_nifti_array(gen_ct_np, case_dir / "generated_ct.nii.gz")
                save_nifti_array(gen_mask_np, case_dir / "generated_mask.nii.gz")

                preview_path = PREVIEW_DIR / f"{case}.png"
                z, slice_tumor_voxels = save_preview(
                    input_ct_np, gen_ct_np, gen_mask_np, preview_path, case
                )

                success_count += 1
                print(
                    f"成功：肿瘤体素={tumor_voxels}，占裁剪体积={tumor_fraction:.3f}%，"
                    f"预览切片 z={z}（该切片肿瘤体素={slice_tumor_voxels}）"
                )
                writer.writerow({
                    "case": case,
                    "status": "success",
                    "tumor_voxels": tumor_voxels,
                    "tumor_fraction_percent": f"{tumor_fraction:.6f}",
                    "preview_png": str(preview_path),
                    "error": "",
                })
                f.flush()

                if len(preview_records) < PREVIEW_CASES:
                    preview_records.append((case, preview_path))

                del ct_tensor, mask_tensor, generated_ct, generated_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                fail_count += 1
                error_text = f"{type(e).__name__}: {e}"
                print(f"失败：{error_text}")
                traceback.print_exc()
                writer.writerow({
                    "case": case,
                    "status": "failed",
                    "tumor_voxels": "",
                    "tumor_fraction_percent": "",
                    "preview_png": "",
                    "error": error_text,
                })
                f.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # 将前若干成功病例的预览图合成一张拼图
    if preview_records:
        cols = 3
        rows = int(np.ceil(len(preview_records) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
        axes = np.asarray(axes).reshape(-1)

        for ax, (case, png_path) in zip(axes, preview_records):
            img = plt.imread(str(png_path))
            ax.imshow(img)
            ax.set_title(case, fontsize=9)
            ax.axis("off")
        for ax in axes[len(preview_records):]:
            ax.axis("off")
        fig.suptitle("DiffTumor Kidney Medium Tumor Batch Preview", fontsize=14)
        fig.tight_layout()
        montage_path = OUT_DIR / "batch_preview.png"
        fig.savefig(montage_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n拼图已保存：{montage_path.resolve()}")
    else:
        print("\n没有成功病例，未生成拼图。")

    print("\n========== 批量生成结束 ==========")
    print(f"计划病例：{len(pairs)}")
    print(f"成功：{success_count}")
    print(f"失败：{fail_count}")
    print(f"日志：{log_path.resolve()}")
    print(f"生成数据：{NIFTI_DIR.resolve()}")
    print(f"单例预览：{PREVIEW_DIR.resolve()}")


if __name__ == "__main__":
    main()

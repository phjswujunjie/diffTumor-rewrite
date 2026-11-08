"""
用 flowmatching/runs_author/{real,fm,ddpm}/model.pt 做滑窗推理，
把同一病例、同一层面的 CT + 真值轮廓 + 预测轮廓画成对比图。

【用途】
    训练跑完后快速目检三个模型的分割效果，不用打开 3D Slicer。
    输出两类文件：
        <case>_compare.png          3 个肿瘤最大层面 × 3 个模型
        <case>_<arm>_tumor.nii.gz   原始分辨率上的预测肿瘤掩膜

【图例】
    灰阶   CT（1mm 网格，HU 窗 [-175,250] 归一化到 [0,1]）
    绿线   真值肿瘤轮廓
    红线   模型预测肿瘤轮廓
    标题里的 dice 是该层面上的 2D Dice，只用于快速对照，
    正式指标以 val_metrics.csv 里的 3D Dice/NSD 为准。

【注意】
    如果模型只训练了几步（例如 smoke），预测会是噪声或整个肝脏，
    这是正常现象；本脚本的意义是验证「推理 -> 后处理 -> 可视化」链路。

用法：
  python flowmatching/preview_seg.py --cases liver_26,liver_129
  python flowmatching/preview_seg.py --cases liver_26 --runs <其他 run 目录>
"""

import os

# 官方 VQGAN/分割权重加载时的兼容开关
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # 无界面环境，必须用 Agg 后端
import matplotlib.pyplot as plt
from monai import data
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

HERE = Path(__file__).resolve().parent
STEP3_DIR = HERE.parent
# STEP3 在路径最前面，避免 import 到 STEP2 的同名 flowmatching 包
sys.path.insert(0, str(STEP3_DIR))
sys.path.insert(0, str(HERE))

# 复用训练时的验证 transform 和后处理函数，保证预览与验证指标同口径
from train_seg_author import _get_transform  # noqa: E402
from monai_trainer import resample, convert_to_one_hot, denoise_pred  # noqa: E402


def build_case_dicts(cases, args):
    """从 fold 的 val txt 里挑出指定病例，拼成 MONAI 需要的数据字典。

    val txt 每行是 "imagesTr/xxx.nii.gz labelsTr/xxx.nii.gz"；
    这里额外拼上器官伪标签路径：
        organ_pseudo_swin_new/liver/liver_26.nii.gz
    """
    val_txt = Path(args.val_txt).expanduser()
    wanted = set(cases)
    out = {}
    for line in val_txt.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name = Path(parts[1]).name          # liver_26.nii.gz
        stem = name.split(".")[0]           # liver_26
        if stem not in wanted:
            continue
        out[stem] = {
            "image": args.data_root + parts[0],
            "label": args.data_root + parts[1],
            "organ_pseudo": args.pseudo_root + name,
            "name": stem,
        }
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"val txt 里找不到病例: {sorted(missing)}")
    return out


def load_model(ckpt, device):
    """加载分割模型。

    网络结构必须与训练时完全一致，否则 load_state_dict 会失败：
        3D U-Net, channels=(16,32,64,128,256), strides=(2,2,2,2),
        num_res_units=2, 输入 1 通道, 输出 3 类（背景/肝脏/肿瘤）
    """
    model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2).to(device)
    ck = torch.load(ckpt, map_location=device)
    # save_checkpoint 保存的键是 "state_dict"
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, int(ck.get("epoch", -1))


@torch.no_grad()
def infer_case(model, item, orig_shape, device):
    """对一个病例做滑窗推理 + 后处理。

    返回两个版本：
        processed_1mm  : 1mm 网格，用于画图（与 ct / gt 同网格）
        processed_orig : 原始分辨率，用于保存 NIfTI（与 val_epoch 同口径）

    为什么要在两个网格上分别做后处理：
        denoise_pred 需要器官伪标签与预测同网格；
        画图用 1mm 网格更清晰，保存 NIfTI 则要回到原始分辨率。
    """
    image = item["image"][None].to(device)          # (1,1,X,Y,Z)
    organ = item["organ_pseudo"][0].cpu().numpy()   # (X,Y,Z)

    # 与 val_epoch 完全一致的滑窗参数：96³、overlap 0.75、gaussian 加权
    inferer = partial(sliding_window_inference, roi_size=(96, 96, 96),
                      sw_batch_size=1, predictor=model, overlap=0.75,
                      mode="gaussian")
    with torch.cuda.amp.autocast():
        logits = inferer(image)

    # 3 类 softmax -> argmax，得到 0/1/2 的硬分割
    pred_1mm = torch.argmax(torch.softmax(logits, 1), 1)[0].cpu().numpy().astype(np.uint8)

    # 1mm 网格上做后处理，用于画图（与 ct / gt 同网格）
    processed_1mm = denoise_pred(convert_to_one_hot(pred_1mm, 3), organ)

    # 原始分辨率上再做一次，用于保存 NIfTI（与 val_epoch 一致）
    pred_orig = resample(pred_1mm, orig_shape)
    organ_orig = resample(organ, orig_shape)
    processed_orig = denoise_pred(convert_to_one_hot(pred_orig, 3), organ_orig)
    return processed_1mm[2].astype(np.uint8), processed_orig[2].astype(np.uint8)


def pick_slices(gt, k=3):
    """挑出真值肿瘤面积最大的 k 个轴位层面。

    这样图上一定能看到肿瘤；如果病例本身没有肿瘤，返回空列表，
    调用方会退化为显示正中间那一层。
    """
    per_slice = gt.sum(axis=(0, 1))
    order = np.argsort(per_slice)[::-1]
    return [int(z) for z in order[:k] if per_slice[z] > 0]


def save_figure(ct, gt, preds, case, out_png):
    """rows = 层面，cols = real / fm / ddpm"""
    arms = list(preds.keys())
    slices = pick_slices(gt)
    if not slices:
        slices = [ct.shape[2] // 2]

    fig, axes = plt.subplots(len(slices), len(arms),
                             figsize=(4 * len(arms), 4 * len(slices)),
                             squeeze=False)
    for r, z in enumerate(slices):
        # np.rot90 只是为了让轴位图方向符合阅片习惯
        ct_sl = np.rot90(ct[:, :, z])
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            ax.imshow(ct_sl, cmap="gray", vmin=0.0, vmax=1.0)
            # 绿色 = 真值，红色 = 预测
            ax.contour(np.rot90(gt[:, :, z]), levels=[0.5],
                       colors="lime", linewidths=0.9)
            ax.contour(np.rot90(preds[arm][:, :, z]), levels=[0.5],
                       colors="red", linewidths=0.9)
            # 该层面的 2D Dice，仅用于快速目检
            inter = int((preds[arm][:, :, z] & gt[:, :, z]).sum())
            gsum = int(gt[:, :, z].sum())
            psum = int(preds[arm][:, :, z].sum())
            dice = 2 * inter / (gsum + psum) if (gsum + psum) else 0.0
            ax.set_title(f"{arm}  z={z}  dice={dice:.2f}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{case}  (green = GT, red = prediction)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="liver_26,liver_129")
    ap.add_argument("--arms", default="real,fm,ddpm")
    ap.add_argument("--val-txt",
                    default=str(STEP3_DIR / "cross_eval" / "liver_aug_data_fold" /
                                "real_tumor_val_0.txt"))
    ap.add_argument("--data-root", default="E:/thesisAndCode/data/Task03_Liver/")
    ap.add_argument("--pseudo-root",
                    default=str(STEP3_DIR / "organ_pseudo_swin_new" / "liver") + "/")
    ap.add_argument("--runs", default=str(HERE / "runs_author"))
    ap.add_argument("--out", default=str(HERE / "runs_author" / "preview"))
    args = ap.parse_args()

    cases = [c for c in args.cases.split(",") if c]
    arms = [a for a in args.arms.split(",") if a]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    case_dicts = build_case_dicts(cases, args)
    # 只取 val transform；训练 transform 里有随机裁剪，不适用于推理
    _, val_transform = _get_transform(argparse.Namespace(organ_type="liver", fg_thresh=30))

    for case, d in case_dicts.items():
        t0 = time.time()
        # Dataset(data=[d]) 会完整跑一遍 val transform：
        # 读图 -> RAS -> 1mm -> HU 归一化 -> padding
        item = data.Dataset(data=[d], transform=val_transform)[0]
        # 原始分辨率用于把预测 resample 回去
        orig_shape = list(nib.load(d["image"]).shape)
        # ct/gt 都在 1mm 网格上，供画图使用
        ct = item["image"][0].cpu().numpy()
        gt = (item["label"][0].cpu().numpy() == 2).astype(np.uint8)
        print(f"[{case}] shape={orig_shape} gt_tumor_vox={int(gt.sum())}", flush=True)

        preds = {}
        for arm in arms:
            ckpt = Path(args.runs) / arm / "model.pt"
            model, epoch = load_model(ckpt, device)
            t1 = time.time()
            pred_disp, pred_nii = infer_case(model, item, orig_shape, device)
            preds[arm] = pred_disp
            print(f"  [{arm}] epoch={epoch} pred_vox={int(pred_disp.sum())} "
                  f"time={time.time() - t1:.1f}s", flush=True)

            nib.save(nib.Nifti1Image(pred_nii, nib.load(d["image"]).affine),
                     str(out_dir / f"{case}_{arm}_tumor.nii.gz"))
            # 三个模型串行加载，用完立刻释放显存，避免 16GB 被占满
            del model
            torch.cuda.empty_cache()

        save_figure(ct, gt, preds, case, str(out_dir / f"{case}_compare.png"))
        print(f"[{case}] done in {time.time() - t0:.1f}s -> "
              f"{out_dir / f'{case}_compare.png'}", flush=True)


if __name__ == "__main__":
    main()
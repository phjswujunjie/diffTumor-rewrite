"""
把 fold-0 的训练病例预裁剪成 96³ 的 .npz 块，供 FM / DDPM 的小规模对照训练使用。

【为什么需要预裁剪】
    原始流程每个 batch 都要从 512×512×800 的完整体积里做 1mm 重采样 + 随机裁剪，
    单例就要十几秒；若再用 CacheDataset 缓存完整体积，内存会到 47 GB 量级。
    小规模 pilot（2k/10k 步）只需要少量 96³ 块，预先裁好可以让训练在几分钟内跑完。

【这个脚本做了什么】
    1. 读 fold-0 的真实肿瘤病例清单；
    2. 每个病例做一次确定性预处理（RAS + 1mm + HU→[0,1]）；
    3. 在肿瘤附近裁 N_CROPS 个 96³ 块（带随机偏移）；
    4. 保存为 .npz（image float32 / label uint8 / start 坐标 / case 名），
       并写一份 index.csv 记录每个块的来源和肿瘤体素数。

【重要】
    .npz 只用于 STEP2 的生成模型对照实验，不用于 STEP3 的分割训练。
    STEP3 走的是作者口径的在线合成，不读取这里的文件。
"""

import os

# 官方 VQGAN checkpoint 需要旧版 torch.load 行为
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# build_preprocess / crop_roi 与测量脚本共用，保证训练 crop 与评估 crop 的
# 预处理方式完全一致（RAS、1mm、HU[-175,250]→[0,1]）。
from run_measurement import build_preprocess, crop_roi
from seed_utils import set_seed

DATA_ROOT = Path(r"E:/thesisAndCode/data/Task03_Liver")
LABEL_ROOT = Path(r"E:/thesisAndCode/data/preprocessed_labels")
# early 版本清单：只保留肿瘤等效半径 <= 10mm 的病例
MANIFEST = ROOT / "cross_eval/liver_tumor_data_early_fold/real_tumor_train_0.txt"
OUT = ROOT / "flowmatching/texture_study/crops/train"
N_CROPS = 2   # 每个病例裁几个 96³ 块；增加它可提升数据多样性
SEED = 1234   # 与训练 seed 一致，保证 crop 可复现


def main():
    # 固定种子：crop 的随机偏移来自 crop_roi 内部的 RNG，
    # 固定后重复运行会得到完全相同的块，便于复现实验。
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    preprocess = build_preprocess()

    # 每行是 "imagesTr/xxx.nii.gz labelsTr/xxx.nii.gz"
    rows = [l.split() for l in MANIFEST.read_text().splitlines() if l.strip()]
    index = []

    # 每个病例只做一次重采样，然后裁 N_CROPS 个块；这是整个流程里最慢的部分。
    # 训练时直接读 .npz，不再重复做 Spacingd。
    for i, (img_rel, lbl_rel) in enumerate(rows):
        case = Path(lbl_rel).stem.replace(".nii", "")
        data = preprocess({
            "image": str(DATA_ROOT / img_rel),
            "label": str(LABEL_ROOT / lbl_rel),
        })
        # preprocess 返回带 channel 维的 (1,X,Y,Z)，取第 0 维变回 3D
        image = np.asarray(data["image"])[0].astype(np.float32)
        label = np.asarray(data["label"])[0]

        for k in range(N_CROPS):
            # 每个块用不同的种子，避免同一病例的两个 crop 完全重合
            cr = crop_roi(image, label, SEED + i * 10 + k)
            if cr is None:
                continue
            img_c, lbl_c, start = cr

            # 太小的肿瘤（<20 体素）多半是裁剪到了肿瘤边缘，
            # 这样的块对学习肿瘤纹理帮助不大，直接丢弃。
            tumor = (lbl_c == 2)
            n_tumor = int(tumor.sum())
            if n_tumor < 20:
                continue

            name = f"case{i:03d}_crop{k}.npz"
            # image 加回 channel 维 (1,96,96,96)，与后续网络的输入约定一致；
            # label 存 uint8 足够（只有 0/1/2）。
            np.savez(OUT / name,
                     image=img_c[None].astype(np.float32),
                     label=lbl_c[None].astype(np.uint8),
                     start=start.astype(np.int32),
                     case=case)
            index.append({"file": name, "case": case, "crop": k,
                          "n_tumor": n_tumor, "start": start.tolist()})

        print(f"[{i+1}/{len(rows)}] {case}: total crops={len(index)}", flush=True)

    # index.csv 用来追溯每个训练块来自哪个病例、裁剪起点在哪里
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "case", "crop", "n_tumor", "start"])
        w.writeheader()
        w.writerows(index)
    print(f"[done] {len(index)} crops -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
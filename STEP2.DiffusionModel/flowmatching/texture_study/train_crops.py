"""
在预裁剪的 96³ 训练块上训练 Flow Matching（2k / 10k 小规模对照实验）。

【用途】
    STEP2 的完整配置是 60k 步、读完整 CT 体积；为了先判断 FM 与 DDPM
    是否值得继续投入，这里用同一批 96³ 块做等步数、等 batch、等学习率、
    等 EMA、等种子的短程训练。三次 10k 实验（FM 4/8/16 步采样、DDPM 10k）
    就是用它产出的。

【和完整训练的关系】
    - 模型、VQGAN、数据管线、Trainer 完全一致；
    - 唯一区别是数据来自 .npz 预裁剪块，且训练步数更少；
    - 因此这里得到的结论只用于「方向判断」，正式结果仍应看完整训练。

运行：
    python -u flowmatching/texture_study/train_crops.py --steps 2001
    每步 loss 会立即打印，方便观察是否发散。
"""

import os

# 官方 VQGAN checkpoint 需要旧版 torch.load 行为
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmatching import Unet3D, FlowMatching, Trainer
from seed_utils import set_seed


class CropDataset(Dataset):
    """读取 prepare_crops.py 生成的 96^3 .npz 文件。

    每个样本返回：
        image : (1,96,96,96) float32，已归一化到 [0,1]
        label : (1,96,96,96) float32，0=背景 1=肝脏 2=肿瘤
        name  : 文件名，仅用于调试

    注意：这里的 label 不是训练目标，而是用于构造 inpainting 的 mask——
    Trainer 会把 image 和 mask 沿 batch 维拼起来送进模型。
    """

    def __init__(self, files):
        self.files = sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        return {
            "image": torch.from_numpy(d["image"]).float(),
            "label": torch.from_numpy(d["label"]).float(),
            "name": self.files[i].name,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(ROOT / "flowmatching/texture_study/crops/train"))
    ap.add_argument("--vqgan-ckpt", default=r"E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt")
    ap.add_argument("--out", default=str(ROOT / "flowmatching/texture_study/checkpoints/liver_crops_2k"))
    ap.add_argument("--steps", type=int, default=2001)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    # 固定种子；DDPM 脚本用同一个 seed，保证两边初始化一致
    seed = set_seed(args.seed)
    print(f"[seed] {seed}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    files = sorted(Path(args.crops).glob("*.npz"))
    print(f"[data] {len(files)} crops", flush=True)
    if not files:
        raise FileNotFoundError(f"no crops in {args.crops}")

    ds = CropDataset(files)
    # 独立的 generator 控制 shuffle，使数据顺序不受模型侧 RNG 消耗影响。
    # DDPM 脚本使用完全相同的 seed 和 batch_size，因此两个实验的 batch 顺序一致。
    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, generator=gen, drop_last=True)

    # 与 DDPM 完全相同的骨干：17 通道输入 = 8 噪声潜变量 + 8 条件潜变量 + 1 mask，
    # 8 通道输出 = 预测的潜变量。
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    flow = FlowMatching(net, image_size=24, num_frames=24, channels=17,
                        timesteps=4, sampling_steps=4,
                        # 连续时间 + 中点 Euler：与正式 FM 配置一致，
                        # 训练时 t~U[0,1)，推理时 4 步也能出结构。
                        time_sampling="continuous",
                        ode_solver="euler_midpoint",
                        vqgan_ckpt=args.vqgan_ckpt).to(device)

    # 复用 ddpm.diffusion.Trainer；它只调用 model(input_data) 取 loss，
    # 不接受 val_loader，cfg 也只在内部保存、未使用，因此传 None。
    trainer = Trainer(
        flow, None, dataset=dl,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.steps,
        # 梯度累积=1：96³、batch 2 在 16GB 显卡上显存充足
        gradient_accumulate_every=1,
        # EMA 0.995：与 DDPM 基线相同，推理只用 EMA 权重
        ema_decay=0.995,
        amp=False,
        save_and_sample_every=args.save_every,
        results_folder=args.out,
        num_sample_rows=1,
        num_workers=0,
    )
    print(f"[train] steps={args.steps} batch={args.batch_size} out={args.out}", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
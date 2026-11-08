"""
在预裁剪的 96³ 训练块上训练 DDPM 基线（与 FM 脚本严格配对）。

【为什么需要这个脚本】
    作者的 DDPM 权重是 60k 步、完整数据训练的；而我们的 FM 是小规模对照。
    直接拿两者比不公平，因此用**完全相同**的数据、batch、步数、学习率、
    EMA 和种子重训一个 DDPM，作为 FM 的同条件基线。

【与 train_crops.py 的差异】
    只有一行：GaussianDiffusion 取代 FlowMatching。
    其余参数（网络结构、timesteps=4、loss=l1、Trainer 配置、seed）逐项相同。

运行：
    python -u flowmatching/texture_study/train_crops_ddpm.py --steps 2001
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

# 注意：这里用的是 ddpm.diffusion 里的 Trainer / Unet3D，
# 与 FM 脚本引用的 Trainer 是同一个类（FM 只是换掉了扩散模型本身）。
from ddpm.diffusion import Unet3D, GaussianDiffusion, Trainer
from seed_utils import set_seed


class CropDataset(Dataset):
    """与 train_crops.py 完全相同的 crop 读取逻辑。

    两个脚本的 Dataset 必须一致，否则数据顺序/内容会产生隐性差异，
    对照实验就失去意义。
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
    ap.add_argument("--out", default=str(ROOT / "flowmatching/texture_study/checkpoints/liver_crops_2k_ddpm"))
    ap.add_argument("--steps", type=int, default=2001)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    # 与 FM 脚本同一个 seed → 相同的模型初始化
    seed = set_seed(args.seed)
    print(f"[seed] {seed}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    files = sorted(Path(args.crops).glob("*.npz"))
    print(f"[data] {len(files)} crops", flush=True)
    if not files:
        raise FileNotFoundError(f"no crops in {args.crops}")

    ds = CropDataset(files)
    # 相同的 generator seed 和 batch_size → 与 FM 实验逐个 batch 对齐
    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, generator=gen, drop_last=True)

    # Same backbone and same 4-step discrete diffusion as the author's DDPM.
    # timesteps=4 是作者对 early tumor 的设置，基线必须保持一致。
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    diffusion = GaussianDiffusion(
        net,
        vqgan_ckpt=args.vqgan_ckpt,
        image_size=24,
        num_frames=24,
        channels=17,
        timesteps=4,
        loss_type="l1",
    ).to(device)

    # Trainer is shared with the FM run; only the model class differs.
    # 其余超参与 train_crops.py 逐项一致，确保唯一变量是生成模型。
    trainer = Trainer(
        diffusion, None, dataset=dl,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.steps,
        gradient_accumulate_every=1,
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
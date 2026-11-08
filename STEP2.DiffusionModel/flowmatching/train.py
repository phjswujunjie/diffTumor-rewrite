"""
Flow Matching 训练入口，镜像 STEP2.DiffusionModel/train.py。

【运行方式】（cwd = STEP2.DiffusionModel）
    python flowmatching/train.py \
        model.vqgan_ckpt=../STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt \
        dataset.data_root_path=<...> \
        dataset.label_root_path=<...>

【与 ddpm 的 train.py 相比只改了】
    1. import 的生成模型类：GaussianDiffusion -> FlowMatching
    2. 额外的 FM 配置项：time_sampling / sampling_steps / ode_solver
    3. Hydra 配置入口：base_cfg_flowmatching（其余配置项同名同义）

【训练循环完全复用】
    Trainer 来自 ddpm.diffusion，它只要求 model(x) 返回一个 loss，
    因此 FlowMatching.forward 实现好之后，训练循环、EMA、AMP、
    梯度累积、checkpoint 格式都不用改——这是保证对照公平的关键。
"""

import os
import sys

# 官方 VQGAN checkpoint 需要旧版 torch.load 行为（见 flow.py 顶部说明）
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# 让 `python flowmatching/train.py` 也能 import STEP2 根目录下的包
# （dataset / ddpm / vq_gan_3d 都在 STEP2 根目录，不在 flowmatching/ 下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hydra
from omegaconf import DictConfig, open_dict
import torch

from flowmatching import Unet3D, FlowMatching, Trainer
from dataset.dataloader import get_loader
from seed_utils import set_seed


@hydra.main(config_path='../config', config_name='base_cfg_flowmatching', version_base=None)
def run(cfg: DictConfig):
    # 先固定所有随机源，保证 DDPM / FM 从相同的模型初始化开始。
    # 这里必须放在建模型之前，否则 Unet3D 的初始化已经消耗了 torch RNG。
    seed = set_seed(cfg.dataset.get('seed', 1234))
    print(f'[seed] {seed}')

    # 多卡时指定主卡；本实验只有一张 5080，gpus=0。
    torch.cuda.set_device(cfg.model.gpus)

    # results_folder 里再拼上数据集名和后缀，避免不同实验的 checkpoint 混在一起。
    # open_dict 是因为 hydra 默认把结构锁定（struct mode），不允许新增/修改键。
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    # 骨干网络与 DDPM 完全相同（同一份 Unet3D、同样的 dim/dim_mults/
    # channels/out_dim），只把生成过程换成 FlowMatching。
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            out_dim=cfg.model.out_dim
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    # FlowMatching 只在构造时读取 VQGAN 权重；训练过程中 VQGAN 全程冻结。
    flow = FlowMatching(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        # 以下三项是 FM 相对 DDPM 新增的配置：
        # continuous 表示 t 连续采样；sampling_steps 只影响推理；
        # euler_midpoint 让 8 步采样也能保持解剖结构。
        time_sampling=cfg.model.get('time_sampling', 'continuous'),
        sampling_steps=cfg.model.get('sampling_steps', cfg.model.timesteps),
        ode_solver=cfg.model.get('ode_solver', 'euler_midpoint'),
    ).cuda()

    # 模型初始化后再 seed 一次：让数据变换的 RNG 状态也与 DDPM 对齐。
    # 原因：模型初始化消耗了多少 torch/python RNG，取决于网络结构；
    # 虽然这里结构相同，但显式重置能避免以后改结构时出现隐性差异。
    # DataLoader 内部还有自己的 seeded generator，保证 shuffle 顺序不受模型影响。
    set_seed(seed)
    train_dataloader, _, _ = get_loader(cfg.dataset)

    # Trainer 直接复用 ddpm.diffusion.Trainer；它只依赖 model(input_data)
    # 返回 loss，因此不需要改任何训练循环代码。
    trainer = Trainer(
        flow,
        cfg=cfg,
        dataset=train_dataloader,
        # 注意：Trainer 不使用 train_batch_size；真实 batch 由 DataLoader 的
        # dataset.batch_size × dataset.num_samples 决定。
        train_batch_size=cfg.model.batch_size,
        # 每 save_and_sample_every 步保存一次 ckpt + 采样可视化
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        # 梯度累积；显存不足时用它模拟更大的 batch
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        # EMA 权重是推理时实际使用的权重（DDPM/FM 都只加载 ema）
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
    )

    # 从指定 step 恢复训练（本实验是从头训练，默认 None）
    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train()


if __name__ == '__main__':
    run()
import os

# 官方 VQGAN checkpoint 需要旧版 torch.load 行为（torch>=2.6）
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Unet3D / Trainer / Tester 直接复用 ddpm，确保对照实验的唯一变量是生成过程
from ddpm.diffusion import Unet3D, Trainer, Tester, EMA

from flowmatching.flow import FlowMatching

__all__ = [
    'Unet3D', 'Trainer', 'Tester', 'EMA',
    'FlowMatching',
]
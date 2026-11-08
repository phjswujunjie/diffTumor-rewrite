"""Flow Matching vs DDPM 对照实验的统一随机种子工具。

对照实验最怕的不是「结果差」，而是「两边数据顺序/初始化不同导致结果差」。
每次实验开始前调用一次 set_seed，把四类随机源全部固定：

    Python random  -> 数据增强里的随机决策（如是否注入肿瘤、几何 mask 尺寸）
    NumPy random   -> 部分 transform（如 RandCropByPosNegLabeld 的位置采样）
    torch CPU      -> 模型参数初始化、Dropout、torch.randn
    torch CUDA     -> GPU 上的随机数生成

注意：
    1. DataLoader 的 shuffle 顺序由 `dataset.dataloader` 里单独的
       `torch.Generator` 控制，它也需要手动 seed；否则仅固定全局 RNG
       并不能保证两个实验拿到相同的 batch 顺序。
    2. `cudnn.deterministic=True` + `benchmark=False` 会让卷积算法选择
       固定下来，牺牲一点速度换取可复现性。如果显存/速度紧张，可以
       关掉 deterministic，但同一实验的两次运行可能出现微小数值差异。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """固定 Python / NumPy / PyTorch（CPU + 全部 CUDA 设备）的随机种子。

    Parameters
    ----------
    seed : int
        任意整数；会同时用于 Python hash 种子，保证集合/字典迭代顺序一致。
    deterministic : bool
        是否强制 cuDNN 使用确定性算法。

    Returns
    -------
    int
        实际使用的种子，方便调用方打印日志。
    """
    seed = int(seed)

    # Python 的 hash 随机化会影响 set/dict 的迭代顺序，进而影响
    # collate 后字段的排列，设为固定值可消除这类隐性差异。
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 固定卷积算法选择；否则同一份代码两次运行可能得到不同结果。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
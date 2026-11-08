"""
FM / DDPM 肿瘤填充工具，供 STEP3 的分割训练做在线合成。

【在 STEP3 里的位置】
    STEP3 的作者流程会调用 TumorGeneration.utils.synthesize_early_tumor，
    它内部固定执行：
        tester.ema_model.eval()
        sample = tester.ema_model.sample(batch_size=..., cond=cond)
    本文件提供两个可直接替换 early 生成器的对象：
        FMTester -> FMSampler -> STEP2 的 FlowMatching
        Tester   -> 作者自带的 DDPM（直接复用）
    这样不需要改动作者的任何一行代码，就能把 early 合成器从 DDPM 换成 FM。

【设计原则】
    - 几何 mask 在外部生成一次，FM 和 DDPM 共用同一个 mask；
    - 只替换 STEP3 `synthesize_early_tumor` 里的采样器；
    - 所有新增代码放在 STEP3/flowmatching/，不改动原 STEP3 文件；
    - 载入的生成模型全程 eval + 冻结，不参与分割网络的梯度。

【用法】
    from flowmatching.fm_synth import load_fm, load_ddpm, load_vqgan
    fm_tester = load_fm(fm_ckpt, vqgan_ckpt, device, steps=8)
    out, mask = synthesize_early_tumor(ct, organ, "liver", vqgan, fm_tester)
"""

import os

# 官方 VQGAN checkpoint 需要旧版 torch.load 行为
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

STEP3_DIR = Path(__file__).resolve().parents[1]
STEP2_DIR = STEP3_DIR.parent / "STEP2.DiffusionModel"
for _p in (str(STEP3_DIR), str(STEP2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TumorGeneration.utils import get_fixed_geo  # noqa: E402
from TumorGeneration.ldm.vq_gan_3d.model.vqgan import VQGAN  # noqa: E402
from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion, Tester  # noqa: E402

# STEP2 的 flowmatching 包与 STEP3/flowmatching 同名，不能直接 `import flowmatching.flow`。
# 这里按文件路径加载 STEP2 的 FlowMatching，避免包名冲突。
# （STEP3 也被放在 sys.path 最前面，给的是 STEP3 的包。）
_spec = importlib.util.spec_from_file_location(
    "step2_flow_module", STEP2_DIR / "flowmatching" / "flow.py")
_step2_flow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step2_flow)
FlowMatching = _step2_flow.FlowMatching


class FMSampler:
    """让 FlowMatching 兼容 STEP3 的 `tester.ema_model.sample(...)` 调用。

    STEP3 的作者代码只会用到两个方法：
        .eval()
        .sample(batch_size=..., cond=...)
    这里做一个最小适配层，不改变 FlowMatching 本身。
    """

    def __init__(self, flow, steps=8):
        self.flow = flow
        # 推理步数：FM 用 8 步时纹理与解剖结构都比较稳；
        # 4 步会明显欠积分（outside SSIM 从 0.60 掉到 0.46）。
        self.steps = int(steps)

    def eval(self):
        self.flow.eval()
        return self

    def sample(self, cond=None, batch_size=1, **kwargs):
        # **kwargs 用来吞掉 STEP3 可能传入的其他参数（如 cond_scale），
        # 保证接口兼容。
        return self.flow.sample(cond=cond, batch_size=batch_size, steps=self.steps)


class FMTester:
    """最小 wrapper：只暴露 `ema_model` 属性，供 synthesize_* 使用。

    作者的 Tester 有 .ema_model；这里用同名属性挂 FMSampler，
    因此 `tester.ema_model.sample(...)` 的调用方式完全不变。
    """

    def __init__(self, flow, steps=8):
        self.ema_model = FMSampler(flow, steps=steps)


def load_vqgan(vqgan_ckpt, device):
    """加载并冻结 VQGAN。

    注意：这里用的是 STEP3 TumorGeneration 里的 VQGAN 实现，
    与 STEP2 训练 FM/DDPM 时使用的 VQGAN 是同一份权重、同一套编解码。
    """
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
    vqgan = vqgan.to(device).eval()
    return vqgan


def load_fm(fm_ckpt, vqgan_ckpt, device, steps=8):
    """加载 STEP2 训练好的 Flow Matching 权重。

    关键参数必须与训练时一致，否则 load_state_dict 会报错：
        dim=24, dim_mults=(1,2,4,8), channels=17, out_dim=8
        timesteps=4（决定时间嵌入的动态范围）
    """
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    flow = FlowMatching(
        net,
        image_size=24,
        num_frames=24,
        channels=17,
        timesteps=4,
        sampling_steps=steps,
        # 与 texture_study 的训练配置保持一致：
        # 连续时间 + 中点 Euler
        time_sampling="continuous",
        ode_solver="euler_midpoint",
        vqgan_ckpt=vqgan_ckpt,
    ).to(device)
    ck = torch.load(fm_ckpt, map_location="cpu", weights_only=False)
    # 推理只用 EMA 权重（与 DDPM 的口径一致）
    flow.load_state_dict(ck["ema"])
    flow.eval()
    print(f"[FM] loaded {fm_ckpt} (step={ck.get('step')}, steps={steps})", flush=True)
    return FMTester(flow, steps=steps)


def load_ddpm(ddpm_ckpt, vqgan_ckpt, device):
    """加载 STEP2 训练好的 DDPM 权重，返回作者原生的 Tester。

    timesteps=4 与训练一致；Tester.load 会同时恢复 model 和 ema_model，
    实际采样使用 ema_model。
    """
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    diffusion = GaussianDiffusion(
        net,
        vqgan_ckpt=vqgan_ckpt,
        image_size=24,
        num_frames=24,
        channels=17,
        timesteps=4,
        loss_type="l1",
        device=device,
    ).to(device)
    tester = Tester(diffusion)
    tester.load(ddpm_ckpt, map_location=device)
    print(f"[DDPM] loaded {ddpm_ckpt}", flush=True)
    return tester


@torch.no_grad()
def synthesize_with_mask(ct_volume, organ_mask, tumor_mask, organ_type,
                         vqgan, sampler, sigma=2.0):
    """用固定肿瘤 mask 做 inpainting，FM / DDPM 共用同一输入。

    这个函数是配对实验的核心：当 mask 完全相同时，
    两个模型生成结果的差异只来自生成器本身，而不是几何形状。

    参数
    ----
    ct_volume : (B,1,96,96,96) in [0,1]
    organ_mask: (B,1,96,96,96) in {0,1}
    tumor_mask: (B,1,96,96,96) in {0,1}
    sampler   : Tester 或 FMTester（都有 `.ema_model.sample`）
    sigma     : 高斯羽化核的宽度，用于把生成区域和周围组织平滑过渡

    返回
    ----
    final_volume_ : (B,1,96,96,96) in [0,1]
    organ_tumor_mask : organ_mask + tumor_mask（0/1/2）
    """
    device = ct_volume.device

    # [0,1] -> [-1,1]，与 STEP2 训练一致
    volume = ct_volume * 2.0 - 1.0
    mask = tumor_mask * 2.0 - 1.0
    mask_ = 1.0 - tumor_mask
    # 抹掉肿瘤区域，得到「健康肝脏」
    masked_volume = (volume * mask_).detach()

    # 轴顺序修正：VQGAN 期望 (B,C,D,H,W)
    volume = volume.permute(0, 1, -1, -3, -2)
    masked_volume = masked_volume.permute(0, 1, -1, -3, -2)
    mask = mask.permute(0, 1, -1, -3, -2)

    # 与训练时完全相同的条件构造：8 通道 masked latent + 1 通道 mask
    masked_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
    emin = vqgan.codebook.embeddings.min()
    emax = vqgan.codebook.embeddings.max()
    masked_feat = ((masked_feat - emin) / (emax - emin)) * 2.0 - 1.0
    cc = F.interpolate(mask, size=masked_feat.shape[-3:])
    cond = torch.cat((masked_feat, cc), dim=1)

    # 采样（FM 走 ODE 积分，DDPM 走 4 步离散去噪）
    sampler.ema_model.eval()
    sample = sampler.ema_model.sample(batch_size=volume.shape[0], cond=cond)

    if organ_type in ("liver", "kidney"):
        # 高斯羽化：只在肿瘤 mask 内部用生成值，边缘做平滑过渡，
        # 避免出现生硬的接缝。
        mask_01 = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
        mask_01_np_blur = gaussian_filter(
            mask_01.cpu().numpy(), sigma=[0, 0, sigma, sigma, sigma])
        volume_ = torch.clamp((volume + 1.0) / 2.0, min=0.0, max=1.0)
        sample_ = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)
        mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
        final_volume_ = (1 - mask_01_blur) * volume_ + mask_01_blur * sample_
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)
    else:
        # 胰腺没有器官羽化，直接用生成结果
        final_volume_ = (sample + 1.0) / 2.0

    # 换回 (B,C,X,Y,Z)，与 STEP3 的训练数据轴顺序一致
    final_volume_ = final_volume_.permute(0, 1, -2, -1, -3)
    # 器官 mask(1) + 肿瘤 mask(1) = 0/1/2，与分割标签约定一致
    organ_tumor_mask = organ_mask + tumor_mask
    return final_volume_, organ_tumor_mask
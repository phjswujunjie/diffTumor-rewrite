"""
Flow Matching 版本的 GaussianDiffusion。

【文件定位】
    STEP2 的 DDPM 实现位于 ddpm/diffusion.py，本文件是它的「同构替代品」：
    网络结构、条件构造、Trainer、EMA、checkpoint 格式全部不变，
    只把「扩散去噪」换成「连续时间流匹配」。

【对照实验原则】
    - Unet3D / Trainer / Tester / EMA 全部直接复用 ddpm.diffusion，不改动
    - 条件构造（VQGAN 编码 + 9 通道 cond）原样保留
    - 唯一差异：训练目标从「预测噪声 eps」换成「预测速度 v」，
      采样从「随机去噪链」换成「确定性 ODE 积分」

【符号约定】
    t = 0            -> 干净数据 x0（VQGAN 潜表示）
    t = 1            -> 纯高斯噪声 x1
    x_t              = (1 - t) * x0 + t * x1        线性插值路径
    v_target         = dx_t/dt = x1 - x0            网络要回归的速度场
    采样             -> 从 x1 出发，沿 -v 方向从 t=1 积到 t=0（Euler）

【为什么这样能和 DDPM 公平对比】
    两条路径都满足「t=0 是数据、t=1 是噪声」，训练/推理使用的 U-Net 输入
    (17 通道) 和输出 (8 通道潜变量) 完全一致，只是训练目标与采样器不同。
    因此相同的 ckpt 保存方式、相同的 VQGAN、相同的数据顺序都可以直接复用。
"""

import os

# 官方 VQGAN checkpoint 里带 omegaconf.DictConfig；torch>=2.6 默认
# weights_only=True 会拒绝加载。仅用于可信的官方权重。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch import nn
import torch.nn.functional as F

from ddpm.diffusion import default, normalize_img, unnormalize_img
from vq_gan_3d.model.vqgan import VQGAN


class FlowMatching(nn.Module):
    """
    流匹配模型。

    构造参数
    ----------
    denoise_fn : nn.Module
        与 DDPM 共用的 3D U-Net。前向签名必须是
        ``forward(x_t, t_emb, cond=cond, cond_scale=...)``。
    image_size / num_frames : int
        潜变量的空间边长与深度。VQGAN 下采样 4 倍，故 96³ 图像对应 24³ 潜变量。
    channels : int
        仅为了与 ddpm.GaussianDiffusion 的接口一致而保留。
        真正送进 U-Net 的通道数是 17 = 8(噪声潜变量) + 8(条件潜变量) + 1(mask)，
        ``sample()`` 里按 DDPM 的约定把潜通道数硬编码为 8。
    timesteps : int
        时间嵌入的动态范围。``_t_embed`` 把 [0,1] 映射到 [0, timesteps-1]，
        这样与 DDPM 的 ``t`` 取值范围一致，同一个 U-Net 的时间嵌入层可直接复用。
    loss_type : {'l1', 'l2'}
        速度场回归损失。作者 DDPM 用的是 l1，这里保持一致。
    vqgan_ckpt : str | None
        若给出，则加载冻结的 VQGAN 负责「图像 <-> 潜表示」的编解码。
    time_sampling : {'continuous', 'discrete'}
        - continuous: 每个样本独立采样 t ~ U[0,1)，更贴近连续流匹配的定义；
        - discrete  : t 只取 k/(T-1)，与 DDPM 的 T 个离散时间级严格对齐，
                      用于做「只换采样器、不换时间离散化」的消融。
    sampling_steps : int | None
        ODE 积分的步数；None 表示沿用 timesteps。
        注意：它只影响推理成本，不影响训练。
    ode_solver : {'euler', 'euler_midpoint'}
        - euler          : 在区间左端点求速度，采样步数少时误差较大；
        - euler_midpoint : 在区间中点求速度，避免在训练分布外的 t=1 端点
                           求值，8 步时明显更稳（实测 outside-SSIM 提升）。
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        vqgan_ckpt=None,
        time_sampling='continuous',
        sampling_steps=None,
        ode_solver='euler_midpoint',
    ):
        super().__init__()
        # 保留 channels 只为与 ddpm.GaussianDiffusion 的接口一致。
        # 真正送进 U-Net 的是 17 = 8(潜变量) + 9(条件)，sample() 里硬编码 8。
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        # 与 Trainer / Tester / 外部代码保持同名属性：
        # Trainer 里会根据 num_timesteps 构造时间嵌入、日志等，沿用可少改代码。
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # 'continuous': t ~ U[0,1)
        # 'discrete'  : t = k/(T-1)，与 DDPM 的 T 个时间级严格对齐
        assert time_sampling in ('continuous', 'discrete'), time_sampling
        self.time_sampling = time_sampling
        self.sampling_steps = (
            int(sampling_steps) if sampling_steps is not None else int(timesteps)
        )
        assert ode_solver in ('euler', 'euler_midpoint'), ode_solver
        self.ode_solver = ode_solver

        # VQGAN 全程冻结：只做 encode/decode，不参与梯度。
        # 若为 None，则退化成「直接在像素空间做流匹配」的数学路径（本实验不用）。
        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
        else:
            self.vqgan = None

    # ------------------------------------------------------------------
    # 路径 / 损失
    # ------------------------------------------------------------------
    def _t_embed(self, t):
        """把 t in [0,1] 映射到 [0, T-1]，与 DDPM 的时间嵌入动态范围一致。

        这一步很关键：U-Net 的时间嵌入层是在 DDPM 的 t∈{0..T-1} 上训练的，
        如果直接把 [0,1] 喂进去，激活值会落在训练分布之外。
        """
        if self.num_timesteps > 1:
            return t * (self.num_timesteps - 1)
        return t

    def interpolate_path(self, x_data, t, noise=None):
        """线性插值路径 x_t = (1-t) x0 + t x1。

        x_data : (B, C, D, H, W) 干净数据（潜表示）
        t      : (B,) 每个样本一个 [0,1] 浮点时间
        noise  : 可选，外部传入可复现同一组噪声（配对实验用）

        返回
        ----
        x_t    : 与 x_data 同形状
        noise  : 本次使用的噪声，供速度目标 v = noise - x_data 使用
        """
        noise = default(noise, lambda: torch.randn_like(x_data))
        # 把 (B,) 形状的 t 扩成 (B,1,1,1,1)，才能与 5 维张量广播。
        t_ = t.view(-1, *([1] * (x_data.ndim - 1)))
        x_t = (1.0 - t_) * x_data + t_ * noise
        return x_t, noise

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        """训练损失。

        x_start: 完整图像的潜表示 (B,8,24,24,24)
        t:       每个样本一个 [0,1] 的浮点时间；0=干净数据，1=纯噪声
        cond:    9 通道空间条件（抹洞后的潜表示 + mask）
                 8 通道是 masked latent，1 通道是降采样后的 mask
        """
        x_t, noise = self.interpolate_path(x_start, t, noise)

        # 速度场目标：dx/dt = x1 - x0。
        # 网络只需回归这个与 t 无关的常量向量，而不像 DDPM 那样要预测
        # 带噪程度相关的 eps，这是 FM 训练更稳、少步采样可用的原因。
        v_target = noise - x_start
        v_pred = self.denoise_fn(x_t, self._t_embed(t), cond=cond, **kwargs)

        if self.loss_type == 'l1':
            return F.l1_loss(v_pred, v_target)
        if self.loss_type == 'l2':
            return F.mse_loss(v_pred, v_target)
        raise NotImplementedError(f'loss_type={self.loss_type}')

    def forward(self, x, *args, **kwargs):
        """训练前向。

        x 是 Trainer 沿 batch 维拼起来的 [image; mask]：
            前一半 img  (B,1,96,96,96)  原始 CT，已归一化到 [0,1]
            后一半 mask (B,1,96,96,96)  1=tumor 区域，0=健康区域
        这里拆开、抹洞、过 VQGAN 构造 9 通道 cond，再算 FM 速度损失。

        为什么可以先抹洞再编码：
            VQGAN 编码器是卷积结构，抹洞后的图像（肿瘤位置置 0）编码出的
            潜表示就是「健康肝脏的样子」，作为 inpainting 的条件。
        """
        # 下面这段与 ddpm.diffusion.GaussianDiffusion.forward 完全一致，
        # 只把最后的 t 采样和 p_losses 换掉。
        bs = int(x.shape[0] / 2)
        img = x[:bs, ...]
        mask = x[bs:, ...]
        mask_ = (1 - mask).detach()
        masked_img = (img * mask_).detach()

        # 轴顺序修正：数据是 (B,C,X,Y,Z)，VQGAN 的 3D 卷积按 (B,C,D,H,W)
        # 理解，所以把最后三轴从 (X,Y,Z) 换成 (Z,X,Y)，与作者代码保持一致。
        # 解码后必须再做 permute(0,1,-2,-1,-3) 换回来。
        masked_img = masked_img.permute(0, 1, -1, -3, -2)
        img = img.permute(0, 1, -1, -3, -2)
        mask = mask.permute(0, 1, -1, -3, -2)

        if isinstance(self.vqgan, VQGAN):
            with torch.no_grad():
                # 干净图像 -> 潜表示（训练目标 x0）
                img = self.vqgan.encode(
                    img, quantize=False, include_embeddings=True)
                # 码本数值范围 -> [-1,1]，与 DDPM 的归一化方式完全一致
                img = ((img - self.vqgan.codebook.embeddings.min()) /
                       (self.vqgan.codebook.embeddings.max() -
                        self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0

                # 抹洞图像 -> 条件潜表示（不含肿瘤信息）
                masked_img = self.vqgan.encode(
                    masked_img, quantize=False, include_embeddings=True)
                masked_img = ((masked_img - self.vqgan.codebook.embeddings.min()) /
                              (self.vqgan.codebook.embeddings.max() -
                               self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            # 无 VQGAN 时的像素空间路径，仅用于数学单元测试。
            img = normalize_img(img)
            masked_img = normalize_img(masked_img)

        # mask 从 {0,1} 变到 {-1,1}，再降采样到潜空间尺寸 (24³)，
        # 与 masked latent 拼成 9 通道条件：
        #   [0:8]   masked latent
        #   [8]     mask 指示通道
        mask = mask * 2.0 - 1.0
        cc = torch.nn.functional.interpolate(mask, size=masked_img.shape[-3:])
        cond = torch.cat((masked_img, cc), dim=1)

        # 时间采样：每个样本一个 t。
        # discrete 模式复用 DDPM 的整数时间网格，continuous 模式用连续均匀分布。
        b, device = img.shape[0], img.device
        if self.time_sampling == 'discrete':
            idx = torch.randint(0, self.num_timesteps, (b,), device=device)
            t = idx.float() / max(self.num_timesteps - 1, 1)
        else:
            t = torch.rand((b,), device=device)

        return self.p_losses(img, t, cond=cond, *args, **kwargs)

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_latent(self, shape, cond=None, cond_scale=1., steps=None,
                      ode_solver=None):
        """从高斯噪声积分回数据潜表示，返回未解码的 latent。

        连续模式下从 t=1 积到 t=0；默认在区间中点求速度，避免在训练
        分布外的 t=1 端点求值。返回的 latent 仍在 permuted 轴顺序，
        调用方解码后需要再做 permute(0,1,-2,-1,-3) 才能对齐原图。

        ODE 形式：
            dx/dt = v(x_t, t, cond)
            x_{t-dt} = x_t - dt * v(x_t, t, cond)

        Euler-midpoint 的取法：
            第 k 步在 t_k - dt/2 处求速度，相当于用区间中点的速度
            近似整段积分，比左端点 Euler 更接近真实轨迹。
        """
        device = next(self.denoise_fn.parameters()).device
        # 起点 t=1 是纯高斯噪声
        x = torch.randn(shape, device=device)

        solver = ode_solver or self.ode_solver

        if self.time_sampling == 'discrete':
            # 训练时 t 只取 k/(T-1)，采样也落在同一组网格点上
            n = max(self.num_timesteps - 1, 1)
            t_values = [1.0 - k / n for k in range(n)]
        else:
            n = int(steps) if steps is not None else self.sampling_steps
            if solver == 'euler_midpoint':
                # 在区间中点求速度，避免在训练分布外的 t=1 端点求值
                t_values = [1.0 - (k + 0.5) / n for k in range(n)]
            else:
                t_values = [1.0 - k / n for k in range(n)]

        # 固定步长：每一步都往 t 减小的方向走 dt
        dt = 1.0 / n
        for t in t_values:
            # 整个 batch 共享同一个标量 t（采样阶段不需要逐样本时间）
            t_emb = torch.full((x.shape[0],), t, device=device, dtype=x.dtype)
            v = self.denoise_fn.forward_with_cond_scale(
                x, self._t_embed(t_emb), cond=cond, cond_scale=cond_scale)
            # 沿 -v 方向前进：t 从 1 降到 0，数据从噪声回到图像
            x = x - dt * v

        return x

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16, steps=None):
        """完整采样：噪声 -> 潜表示 -> VQGAN 解码 -> 图像。

        cond: 9 通道空间条件，由调用方按训练时同样的方式构造。
        batch_size: 一次生成多少个样本（每个样本对应一个 24³ 潜变量）。
        steps: ODE 步数；None 使用构造时的 sampling_steps。

        返回形状 (B,1,96,96,96)，数值范围 [0,1]，轴顺序为 permuted 后的
        (B,C,Z,X,Y)。调用方若要与原始 CT 对齐，还需 permute(0,1,-2,-1,-3)。
        """
        # cond 是训练时构造的 9 通道空间条件，不是文本 embedding。
        image_size = self.image_size
        channels = 8  # 与 ddpm 一致：潜通道数，不是 self.channels(17)
        num_frames = self.num_frames

        _sample = self.sample_latent(
            (batch_size, channels, num_frames, image_size, image_size),
            cond=cond,
            cond_scale=cond_scale,
            steps=steps,
        )

        if isinstance(self.vqgan, VQGAN):
            # 先反归一化回码本数值范围，再量化解码；与 ddpm.sample 一致。
            # 注意：忘记这一步会让解码出的肿瘤偏暗（曾踩过的坑）。
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()
            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            # 无 VQGAN 的纯数学路径；ddpm 原版这里丢掉了返回值，FM 版修正。
            _sample = unnormalize_img(_sample)

        return _sample
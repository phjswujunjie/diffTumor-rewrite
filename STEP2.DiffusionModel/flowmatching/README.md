# flowmatching —— STEP2 的 Flow Matching 对照实验

本目录把 `STEP2.DiffusionModel/ddpm` 的生成过程从 **DDPM** 换成 **Flow Matching**，
其余全部保持原样，用于和 DDPM 做对照实验。

## 唯一变量

| | `ddpm/` | `flowmatching/` |
|---|---|---|
| 骨干网络 | `Unet3D` | **直接复用** `ddpm.diffusion.Unet3D` |
| 条件构造 | VQGAN 编码 + 9ch cond | **完全一致** |
| 训练器 / EMA / 日志 / 存档 | `ddpm.diffusion.Trainer` | **直接复用** |
| 推理器 | `ddpm.diffusion.Tester` | **直接复用** |
| 训练目标 | 预测噪声 `eps` | 预测速度 `v = x1 - x0` |
| 采样 | 随机去噪链 T 步 | 确定性 ODE 积分 N 步 |

`Unet3D`、`Trainer`、`Tester`、`EMA` 都是从 `ddpm` import 的，
**没有复制、没有改动**，所以两边参数量、存档格式、日志字段完全一致。

## 数学约定

```
t = 0  -> 干净数据 x0
t = 1  -> 高斯噪声 x1

x_t    = (1 - t) * x0 + t * x1
v(x_t) = x1 - x0                      # 训练目标
```

时间嵌入输入为 `t * (T - 1)`，与 DDPM 的整数时间范围 `[0, T-1]` 对齐。

采样用 Euler 积分，从 t=1 积到 t=0：

```
x = randn(shape)
for k in range(N):
    t = 1 - (k + 0.5) / N
    v = denoise_fn(x, t * (T - 1), cond)
    x = x - (1 / N) * v
```

## 文件

| 文件 | 作用 |
|---|---|
| `flow.py` | `FlowMatching`：训练损失 + 普通 Euler ODE 采样 |
| `train.py` | 训练入口，镜像 `STEP2.DiffusionModel/train.py` |
| `__init__.py` | 导出 `Unet3D / Trainer / Tester / FlowMatching` |

对应的配置：

- `config/base_cfg_flowmatching.yaml`
- `config/model/flowmatching.yaml`

## 运行

```bash
cd STEP2.DiffusionModel
python flowmatching/train.py \
    model.vqgan_ckpt=../STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt \
    dataset.data_root_path=<...> \
    dataset.label_root_path=<...>
```

其余命令行参数与 `train.py` 完全一致，例如：

```bash
python flowmatching/train.py \
    dataset.name=liver_tumor_train \
    dataset.fold=0 \
    dataset.dataset_list=['liver_tumor_data_early_fold'] \
    model.results_folder_postfix=liver_early_fm \
    model.timesteps=4 \
    model.sampling_steps=4
```

checkpoint 默认写到 `checkpoints/flowmatching/...`，不会覆盖 DDPM 的结果。

## 配置开关

| 配置 | 默认 | 说明 |
|---|---|---|
| `time_sampling` | `continuous` | `continuous`: t~U[0,1)；`discrete`: t=k/(T-1)，与 DDPM 的 T 个时间级对齐 |
| `sampling_steps` | = `timesteps` | ODE 积分步数；默认与 DDPM 同等 NFE |
| `ode_solver` | `euler_midpoint` | `euler_midpoint`（推荐）或 `euler` |

> `time_sampling=discrete` 时，采样步数由 `T-1` 决定（每个网格点一次网络调用），
> 因此 NFE 与 `continuous` 不同。它只用于消融，不建议作为主实验。

## 已知差异 / 注意

1. **VQGAN 权重路径**：`config/model/flowmatching.yaml` 默认仍是
   `AutoencoderModel.ckpt`，实际运行时请用命令行覆盖到实际路径。
2. **`Tester.load` 同时加载 `model` 和 `ema`**：复用的 `ddpm.diffusion.Tester.load`
   会恢复两者，所以采样时直接用 `tester.ema_model` 即可（与 STEP3 的行为一致）。
3. **采样后不做 clamp**：与 DDPM 保持一致，避免引入额外差异。
4. **`sample()` 里 `channels = 8`**：沿用 DDPM 的硬编码（潜通道数，不是 U-Net 的 17）。
5. **STEP3 不能直接加载 FM 权重**：`STEP3.SegmentationModel/TumorGeneration/ldm/ddpm`
   是另一份分叉代码，期望 DDPM 的 `betas/alphas_cumprod` 与固定去噪接口。
   要做 STEP3 的 FM 对照，需要另写推理适配（本次不改）。
6. **采样只用 `FlowMatching.sample` / `sample_latent`**：普通 Euler ODE 积分，
   不依赖 DDIM，也不引入额外的 sampler 类。
7. **随机种子**：`train.py` 原本没有固定 seed，本入口也保持一致；
   若要做严格对照，两边需要同时固定 seed（会改动旧入口，未包含在本次改动中）。

## 冒烟测试

### A. 不加载 VQGAN 的纯数学路径

不走 VQGAN 时潜空间是 **1 通道**，条件（masked latent + mask）是 2 通道，
所以 UNet 必须是 `channels=3, out_dim=1`。真实训练不会用这个分支，
它只用来快速验证损失/梯度/积分。

```bash
cd STEP2.DiffusionModel
python - <<'PY'
import torch
from flowmatching import Unet3D, FlowMatching

net = Unet3D(dim=24, dim_mults=(1,2,4,8), channels=3, out_dim=1)
fm = FlowMatching(net, image_size=24, num_frames=24, channels=3,
                  timesteps=4, sampling_steps=4, vqgan_ckpt=None)

x = torch.randn(2, 1, 96, 96, 96)          # [image; mask] 沿 batch 维拼合
loss = fm(x)
loss.backward()
z = fm.sample_latent((1, 1, 24, 24, 24), cond=torch.randn(1, 2, 24, 24, 24))
print('loss', float(loss), 'latent', tuple(z.shape))
PY
```

### B. 真实 VQGAN 路径

```bash
cd STEP2.DiffusionModel
python - <<'PY'
import torch
from flowmatching import Unet3D, FlowMatching

ckpt = '../STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt'
net = Unet3D(dim=24, dim_mults=(1,2,4,8), channels=17, out_dim=8)
fm = FlowMatching(net, image_size=24, num_frames=24, channels=17,
                  timesteps=4, sampling_steps=4, vqgan_ckpt=ckpt).cuda().eval()

x = torch.randn(2, 1, 96, 96, 96).cuda()
loss = fm(x)                                # 训练态才会走 loss；eval 下同样可算
cond = torch.randn(1, 9, 24, 24, 24).cuda()
with torch.no_grad():
    s = fm.sample(cond=cond, batch_size=1)  # 走 VQGAN decode
print('loss', float(loss), 'sample', tuple(s.shape))
PY
```
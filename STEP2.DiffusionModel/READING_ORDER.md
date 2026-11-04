# STEP2 DiffusionModel 代码阅读指南

> 配套说明：本目录源码已加中文注释。本文档告诉你**按什么顺序读**、**每步该看懂什么**、
> **怎么自检**，以及**哪些文件不用读**。
>
> 关于行号：下文引用的行号是**当前带注释版本**的。你自己再增删注释就会偏移，
> 届时请以函数名/类名为准定位。

---

## 0. 先建立一句话认知

**STEP2 在一个压缩过的潜空间里做扩散，而这个扩散的任务是「条件修复」——
把图像里被抹掉的肿瘤区域，用扩散模型的先验"补"回来。**

两个关键限定词，缺一不可：

**（1）潜空间扩散（Latent Diffusion）**
```
96³ 的 CT ──VQGAN编码(÷4)──> 24³ 潜表示 ──扩散在这里做──> 24³ 潜表示 ──VQGAN解码──> 96³ CT
```
扩散模型**从不直接处理 96³ 的体素**。它看到的永远是 8×24×24×24。
这就是为什么所有配置里的尺寸都是 24 而不是 96。

**（2）条件修复（Inpainting）**
模型不是"从零生成一张 CT"，而是：
```
输入：被抹掉肿瘤的 CT + 告诉它洞在哪的 mask
输出：那个洞应该长什么样
```
用 UNet 的条件通道拼进去，见下面第 4 步。

**为什么这么设计**：论文的论点是"早期肿瘤跨器官相似"，所以模型只需要学会
**肿瘤的纹理和形状**，不需要学整个腹部的解剖结构。把任务限定成"填空"，
学习难度大幅下降 —— 这也是为什么早期肿瘤只需要 `timesteps=4`。

---

## 1. 文件地图

### 需要精读（4 个）

| 文件 | 行数 | 重要度 | 说明 |
|---|---|---|---|
| `config/model/ddpm.yaml` | 44 | ★★★★★ | 所有尺寸和超参，读代码时反复回查 |
| `train.py` | 86 | ★★★★ | 入口，很短，看模型怎么搭起来 |
| `ddpm/diffusion.py` | 1297 | ★★★★★ | **核心中的核心**，U-Net + 扩散过程 + 训练器全在这 |
| `ddpm/ddim.py` | 246 | ★★★★ | DDIM 采样器，中晚期肿瘤用 |

### 需要略读（3 个）

| 文件 | 怎么读 |
|---|---|
| `dataset/dataloader.py` | 只读 `get_loader`（第 245 行）的变换链和文件清单解析，其余是 MONAI 样板 |
| **`ddpm/util.py`** | **在路径上**——被 `ddim.py:21` 使用，提供 DDIM 的时间步挑选和系数计算。知道作用即可 |
| `ddpm/unet.py` | **被 import 但从未实例化**。`train.py:9` 有一句 `from ddpm.unet import UNet`，
但真正用的 `Unet3D` 来自 `ddpm/diffusion.py:407`。看一眼确认即可 |

### 不需要重读（STEP1 的拷贝）

| 目录 | 为什么 |
|---|---|
| `vq_gan_3d/` | **与 STEP1 的完全逐字节相同**（0 行差异，可自行 diff 验证）。中文详解看 STEP1 那份 |

### 直接跳过（死代码，已在文件头加标记）

| 文件 | 为什么 |
|---|---|
| `dataset/healthy_dataloader.py` | 全仓库无引用，是 STEP3 那条链路的配套代码被放错了地方 |
| `callbacks.py` | 全仓库无引用。是 PL 风格的回调，但 STEP2 训练完全手写，不走 PL |
| **`ddpm/time_embedding.py`** | **间接死代码**——只被 `unet.py` 引用，而 `unet.py` 本身不参与流程 |
| **`ddpm/text.py`** | 被 `diffusion.py:20` import，但 BERT 文本条件分支**永不触发**（条件恒为张量） |
| `dataset/dataloader.py` 里的 `Compose_Select`、`RandCropByLabelClassesd_select` | 多数据集混合训练的实验遗留，实际用的是普通 `Compose` |

---

## 2. 六步阅读路线

### 第 1 步 · 配置：建立"尺寸从哪来"（10 分钟）

**读**：`config/model/ddpm.yaml` → `config/dataset/synt_ct.yaml`

**核心是记住这张对照表**：

| 配置项 | 值 | 怎么来的 |
|---|---|---|
| `diffusion_img_size` | 24 | STEP1 的 `roi(96) / downsample(4)` |
| `diffusion_depth_size` | 24 | 同上（三个轴都是 24） |
| `out_dim` | 8 | STEP1 的 `embedding_dim`（潜通道数） |
| `diffusion_num_channels` | **17** | 8（噪声潜）+ 8（条件潜）+ 1（mask） |
| `timesteps` | **4** | 早期肿瘤只需 4 步 —— 论文论点在超参上的体现 |
| `dim_mults` | [1,2,4,8] | U-Net 通道倍率 |

> **自检**
> 1. 扩散模型处理的张量形状是什么？ → **8 × 24 × 24 × 24**（不是 96³）
> 2. 17 这个数字怎么来的？ → 8 + 8 + 1，U-Net 输入 = 噪声潜 + 条件潜 + mask
> 3. `batch_size: 2` 是真实 batch 吗？ → **不是**。Trainer 只拿它算 `len(dataloader)`，
>    真实 batch 来自 `dataset.batch_size × dataset.num_samples`
> 4. `objective: pred_x0` 说明模型预测 x0 吗？ → **不是**，那是无效字段。
>    代码实际预测**噪声 ε**（见 `p_losses`）
> 5. `sampling_timesteps: 250` 用在哪？ → **哪里都不用**，死字段

---

### 第 2 步 · 入口：`train.py`（15 分钟）

**读**：整个文件，只有 65 行。

**目标**：确认三件事
1. U-Net 的参数全是**潜空间**尺寸（`dim=24`, `channels=17`, `out_dim=8`）
2. `GaussianDiffusion` 把 VQGAN 载成了**自己的子模块**
3. 训练**不用 PyTorch-Lightning**，是手写的 `Trainer`

**第 2 点很重要**，它有两个后果：
- `Trainer` 里 `copy.deepcopy` 做 EMA 时，会连 VQGAN 一起复制 → **显存里住着 2 份**
- 优化器 `Adam(diffusion_model.parameters())` 会包含 VQGAN 参数（但拿不到梯度，实际不更新）

> **自检**
> 1. 为什么 `train.py` 里没有 `pl.Trainer`？ → STEP2 是手写训练循环，不是 PL
> 2. VQGAN 在这里是训练还是冻结的？ → **冻结**。载入后立刻 `.eval()`，
>    所有前向都在 `torch.no_grad()` 里

---

### 第 3 步 · 数据管道（25 分钟）

**读**：`dataset/dataloader.py` 的 `get_loader`（第 ~225 行起）

**与 STEP1 数据管道的两个关键区别**：

**区别一：数据来源不同**
```
STEP1: AbdomenAtlas，图像和标签在同一个目录下
       <case>/ct.nii.gz  +  <case>/segmentations/

STEP2: MSD-Liver，图像和标签在【两个不同的根目录】
       图像 → data_root_path  + "imagesTr/liver_95.nii.gz"
       标签 → label_root_path + "liver_early_tumor/liver_95.nii.gz"
```
清单文件 `cross_eval/<dataset_list>/real_tumor_train_<fold>.txt` 每行两列。

> **注意**：路径是**字符串拼接**不是 `os.path.join`，所以两个 root 都必须以 `/` 结尾。

**区别二：裁剪策略不同**
```python
# STEP2：按类别加权
RandCropByLabelClassesd(ratios=[0, 1, 1], num_classes=3, ...)
#                         背景 器官 肿瘤
# 背景权重为 0 → 完全不裁纯背景的块

# STEP1：按正负样本加权
RandCropByPosNegLabeld(pos=20, neg=1, ...)
#                       器官:背景 = 20:1
```
两者意图相同（只关心器官/肿瘤区域），但 STEP2 更激进。

> **自检**
> 1. `num_samples: 10` 对显存意味着什么？ → 每个病例裁 10 块 → batch 维变 10；
>    再乘 Trainer 的"拼合"（图像+mask）→ **每 step 有 20 个 96³ 体积过 VQGAN**
> 2. 为什么标签要重映射？ → 数据集标签是 {1=器官, 2=肿瘤}，
>    Trainer 里转成 {0=背景, 1=肿瘤}，让 mask 直接可用作"肿瘤指示器"

---

### 第 4 步 · ★ 条件构造：`forward()`（40 分钟，本步最重要）

**读**：`ddpm/diffusion.py` 第 948 行的 `GaussianDiffusion.forward`
（类 `GaussianDiffusion` 定义在第 671 行）

**这是 STEP2 最需要看懂的一个方法。** 它解释了"条件到底是什么"。

**输入**：一个沿 batch 维**拼了两份**的张量 `[图像们 ; mask们]`
（为什么这么拼，见第 5 步的 Trainer）

**五步流程**：

```
1. 拆开拼合的 batch
   bs = x.shape[0] / 2
   img  = x[:bs]      ← 前一半是真实 CT
   mask = x[bs:]      ← 后一半是标签

2. 用 mask 抹掉肿瘤区域
   mask_      = 1 - mask           ← 肿瘤处为 0
   masked_img = img * mask_        ← 肿瘤区域被置零

3. 两者都过 VQGAN 编码到潜空间，并用码本的 min/max 归一化到 [-1, 1]
   img        → 8×24³   （扩散的目标）
   masked_img → 8×24³   （条件的空间部分）

4. 构造条件
   mask → {0,1} 映射到 {-1,1}，再插值下采样到 24³
   cond = concat([masked_img(8ch), mask(1ch)])   ← 共 9 通道

5. 随机取时间步 t，交给 p_losses
```

**最终 U-Net 的输入 = 8(噪声潜) + 9(条件) = 17 通道**，与 `diffusion_num_channels: 17` 对上。

**所以模型的语义是**：
> "我知道周围组织长什么样（masked 潜表示），也知道洞在哪（mask），
>  请把洞里的内容补上。"

**这也是为什么不需要显式的灰度匹配**：条件来自这张 CT 自己的潜表示，
模型天然知道该用什么强度。

> **自检**
> 1. 为什么是 17 通道？ → 8 + 9（噪声 + 条件），条件是"通道拼接"不是 cross-attention
> 2. 条件里的 mask 为什么是 1 个通道而不是 8 个？ → mask 是二值的空间指示，
>    不需要 8 个通道；它告诉模型"洞在哪"，不携带内容信息
> 3. 潜表示归一化用的 min/max 是哪来的？ → `vqgan.codebook.embeddings` 的
>    全局最小/最大值。这是**固定常数**，采样时会用同样的值反归一化

---

### 第 5 步 · ★ 训练循环：`Trainer.train()`（35 分钟）

**读**：`ddpm/diffusion.py` 第 1031 行的 `class Trainer` 和第 1153 行的 `Trainer.train`

**这是第二个必须看懂的地方。**

#### 5.1 「拼合 batch」机制 —— 全流程最巧妙的一处

```python
image = data['image'].cuda()
mask  = data['label'].cuda()

# 标签重映射：{1=器官, 2=肿瘤} → {0=背景, 1=肿瘤}
mask[mask==1] = 0
mask[mask==2] = 1

# ★ 沿 batch 维拼起来
input_data = torch.cat([image, mask], dim=0)   # (B,...) + (B,...) → (2B,...)

loss = self.model(input_data, ...)             # 进 forward 后再拆开
```

**为什么要这么绕**？因为 DataLoader 一次只能返回一组张量，
而 `forward` 需要同时拿到「图像」和「mask」两样东西。
把它们沿 batch 维拼起来传进去，进去再拆开 —— 这是最省事的做法。

**理解了这个，就能理解 `forward` 开头那个 `bs = x.shape[0]/2`。**

#### 5.2 训练器是手写的

与 STEP1 对比：

| | STEP1 | STEP2 |
|---|---|---|
| 训练框架 | PyTorch-Lightning | **完全手写** |
| 优化器 | `configure_optimizers` 返回 | 自己 `Adam(...)` |
| 梯度累积 | PL 的 `accumulate_grad_batches` | 自己写 `for i in range(...)` 循环 |
| EMA | 无 | 自己实现 `EMA` 类 |
| 日志 | PL logger | `tensorboardX.SummaryWriter` |
| checkpoint | PL 的 ModelCheckpoint | 自己写 `save()` / `load()` |

#### 5.3 几个数值细节

| 参数 | 值 | 含义 |
|---|---|---|
| `train_num_steps` | 60000 | 总训练步数 |
| `gradient_accumulate_every` | 2 | 累积 2 步才更新一次 |
| `ema_decay` | 0.995 | EMA 衰减 |
| `step_start_ema` | 2000 | **前 2000 步不做 EMA**，只同步参数 |
| `save_and_sample_every` | 1000 | 每 1000 步存一个 milestone |
| `best_train_loss` | 0.05（初值） | 损失跌破这个值就存 `model_best` |

> **自检**
> 1. 模型 forward 的输入张量为什么是 2B 大小？ → 图像和 mask 沿 batch 维拼在一起了
> 2. 前 1000 步时，EMA 模型和当前模型一样吗？ → **一样**。`step_start_ema=2000`，
>    之前只做 `reset_parameters()` 直接同步
> 3. 采样时用哪个模型？ → **EMA 模型**（`tester.ema_model.sample(...)`）
> 4. 「最优模型」是按什么选的？ → 训练损失（不是验证损失），
>    因为整个流程没有验证集

---

### 第 6 步 · 采样：`sample()` 与 DDIM（30 分钟）

**读**：`ddpm/diffusion.py` 第 849 行的 `GaussianDiffusion.sample` → `ddpm/ddim.py`

#### 6.1 两种采样方式

| | 完整 DDPM | DDIM |
|---|---|---|
| 实现 | `GaussianDiffusion.sample` | `DDIMSampler.sample` |
| 步数 | `num_timesteps` 全走 | 可压缩到 S 步 |
| 用在哪 | **早期肿瘤**（只有 4 步，没必要压缩） | **中晚期肿瘤**（200 步压到 50） |

#### 6.2 采样链路

```
随机噪声 (8×24³)
   │  DDPM 4 步 或 DDIM 50 步，每步都用 cond 做条件
   ▼
潜表示 (8×24³)
   │  反归一化（用码本的 min/max 逆变换）
   │  vqgan.decode(quantize=True)
   ▼
CT 图像 (1×96³)
```

**注意 `sample` 里有一行硬编码**：
```python
channels = 8   # self.channels
```
必须写 8（潜通道数），**不能**用 `self.channels`（那是 17，U-Net 的输入通道）。

> **自检**
> 1. 早期肿瘤采样走哪条路径？ → 完整 DDPM，`p_sample_loop` 循环 4 次
> 2. DDIM 的 `S` 参数是什么？ → 目标采样步数（如 50），
>    会从完整 T 步里均匀挑出 S 个时间点
> 3. 解码前为什么要 `quantize=True`？ → VQGAN 是离散码本模型，
>    连续潜向量必须先量化到最近的码字才能解码

---

## 3. 三条主线

### 主线一：潜空间尺寸链

```
96³ CT ──encode(÷4, quantize=False)──> 8×24³ ──归一化到[-1,1]──> 扩散的目标
                                                                      │
                                                          加噪 / 去噪（t 步）
                                                                      │
96³ CT <──decode(quantize=True)── 8×24³ <──反归一化───────────────┘
```

### 主线二：17 通道的构成

```
U-Net 输入 (17)
├── 8  噪声潜表示        ← 待去噪的目标
└── 9  条件
    ├── 8  被抹掉的潜表示  ← 告诉模型周围长什么样
    └── 1  mask           ← 告诉模型洞在哪
```

### 主线三：数据流的两处"变形"

STEP2 的数据形状在流程中被改了两次，容易看晕：

```
① DataLoader 输出         (B, 1, 96, 96, 96)  image 和 label 各一份
② Trainer 拼合后          (2B, 1, 96, 96, 96) 沿 batch 维拼
③ forward 拆开            (B, 1, 96, 96, 96)  前一半 img / 后一半 mask
④ VQGAN 编码后            (B, 8, 24, 24, 24)  潜空间
⑤ 拼条件后                (B, 17, 24, 24, 24) 送进 U-Net
```

---

## 4. 最容易卡住的四个地方

**1. U-Net 的卷积是"2D"的**（第 4 步之前先看架构）

所有卷积的深度 kernel 都是 1：
```python
Block.proj    = nn.Conv3d(dim, dim_out, (1, 3, 3), ...)   # 深度 kernel = 1
Upsample/Downsample = ... (1, 4, 4), (1, 2, 2) ...        # 深度不下采样
```
**所以卷积部分本质上是逐切片的 2D 卷积，深度轴全程保持 24。**
唯一跨深度传递信息的地方是 `temporal_attn`（沿深度轴的注意力）。

这是从视频扩散模型改写来的痕迹 —— 原版把深度当"时间帧"，
于是"空间用卷积、时间用注意力"。术语 `num_frames`、`temporal_attn` 都是这么来的。

**2. 「拼合 batch」机制**（第 5 步）
`torch.cat([image, mask], dim=0)` 再在 `forward` 里拆开。
不看 Trainer 直接看 `forward`，会完全不明白 `bs = x.shape[0]/2` 是什么。

**3. 17 = 8 + 8 + 1**（第 4 步）
这个数字不是随便定的，是条件拼接的结果。想通这个，条件机制就通了。

**4. EMA 模型 vs 当前模型**（第 5 步）
训练时更新的是 `self.model`，但**采样用的是 `self.ema_model`**。
看 STEP3 的合成代码时，`tester.ema_model.sample(...)` 里的 `ema_model` 就是这个。

---

## 5. 已确认的坑与 bug 清单

| 位置 | 问题 | 影响 |
|---|---|---|
| `Trainer.__init__` | `copy.deepcopy(self.model)` 会把 VQGAN 一起复制 | 显存里 2 份 (VQGAN + U-Net)，无法回避的固定开销 |
| `Trainer.__init__` | `self.batch_size = train_batch_size` 从没被用到 | 真实 batch 来自 DataLoader，改这个没用 |
| `Trainer.train` | `best_train_loss = 0.05` 初值 + 只看训练损失 | 所谓 `model_best` 不是验证最优，只是个"损失创新低"的快照 |
| `GaussianDiffusion.sample` | `channels = 8` 硬编码（注释还写着 `# self.channels`） | 用 `self.channels`(17) 会出错，必须保持 8 |
| `Tester.load` | `milestone == -1` 分支引用未定义的 `self.results_folder` | 传 -1 必然 AttributeError，只能传明确路径 |
| `ddpm/ddim.py` `register_buffer` | 硬编码 `torch.device("cuda")` | 只能跑在 CUDA 上 |
| `ddpm/ddim.py` | 残留 `# breakpoint()`（已注释） | 无害，但是调试痕迹 |
| config `ddpm.yaml` | `objective: pred_x0` | **误导**，实际是 ε-prediction |
| config `ddpm.yaml` | `sampling_timesteps: 250` | **死字段**，没有任何代码读取 |
| config `ddpm.yaml` | `logger: wandb` | **没生效**，代码用的是 tensorboardX |

---

## 6. 速通版（25 分钟）

1. **`config/model/ddpm.yaml`**（3 分钟）—— 记住 24 / 17 / 8 / timesteps=4
2. **`train.py`**（5 分钟）—— 看 U-Net 参数和 VQGAN 被载成子模块
3. **`GaussianDiffusion.forward`**（10 分钟）—— ★ 条件构造的五步流程
4. **`Trainer.train`**（7 分钟）—— ★ 拼合 batch 机制

**25 分钟后你应该能回答**：
扩散在什么空间做？17 通道怎么来的？条件和噪声怎么拼进 U-Net？

---

## 7. 术语对照表

| 代码里的名字 | 含义 |
|---|---|
| `denoise_fn` | 去噪网络（U-Net），扩散模型的核心 |
| `q_sample` | 前向加噪：x_0 → x_t |
| `p_losses` | 训练损失：预测噪声的 L1 误差 |
| `p_sample` / `p_sample_loop` | 反向去噪采样 |
| `cond` | 条件，9 通道（被抹掉的潜表示 + mask） |
| `num_frames` | **实际是 CT 的深度维度**，视频模型遗留命名 |
| `temporal_attn` | **实际是沿深度轴的注意力**，同上 |
| `masked_img` | 肿瘤区域被置零的图像 |
| `input_data` | Trainer 里拼合的 `[image ; mask]` |
| `milestone` | 定期保存的 checkpoint，文件名是步数/1000 |
| `ema_model` | 参数的指数滑动平均副本，**采样用这个** |
| `extract` | 按时间步从系数表里取值并 reshape 的函数 |

---

## 8. 与上下游的衔接

### 上游：STEP1 给了什么

| STEP1 产物 | 在 STEP2 的用途 |
|---|---|
| `AutoencoderModel.ckpt` | `vqgan_ckpt`，做 encode/decode |
| 潜空间 `8×24³` | 扩散的目标空间 |
| `embedding_dim=8` | `out_dim: 8` |
| `roi=96 / downsample=4 → 24` | `diffusion_img_size: 24` |
| `codebook.embeddings` 的 min/max | 潜表示归一化的固定范围 |

### 下游：STEP2 给 STEP3 什么

| STEP2 产物 | 在 STEP3 的用途 |
|---|---|
| `liver_early.pt` / `liver_noearly.pt` | 合成早期 / 中晚期肿瘤的权重 |
| checkpoint 里的 `ema` 项 | STEP3 用的是 EMA 参数（`noearly_checkpoint['ema']`） |
| `tester.ema_model.sample(cond)` | 合成肿瘤的核心调用 |
| `timesteps=4`（早期）/ 200（中晚期） | 决定 STEP3 里是否要用 DDIM 加速 |

**下一步**：STEP3 会把 STEP2 的扩散模型当成一个"肿瘤生成器"来调用，
把生成的肿瘤贴到健康 CT 上，再去训练分割网络。详见 `../STEP3.SegmentationModel/READING_ORDER.md`。

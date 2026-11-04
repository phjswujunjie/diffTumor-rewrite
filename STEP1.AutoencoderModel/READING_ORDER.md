# STEP1 AutoencoderModel 代码阅读指南

> 配套说明：本目录的源码已经加了中文注释。本文档告诉你**按什么顺序读**、**每步该看懂什么**、
> **怎么自检**，以及**哪些文件不用读**。
>
> 建议配合 git 使用：`git log --oneline` 可看仓库改动，`git diff` 可看本目录注释加了哪些。
> 如果想撤销所有注释：`git checkout STEP1.AutoencoderModel/`

---

## 0. 先建立一句话认知

**STEP1 是一个 3D VQGAN，唯一使命是把 96³ 的 CT 体积压成 8×24³ 的离散潜表示。**

它不生成肿瘤、不做分割、不分类。它就是个**体积压缩器**，压缩结果供 STEP2 的扩散模型使用。

理解了这一点，很多事情就顺了：
- 它为什么只用器官区域的 crop 训练？—— 因为下游只用得上器官
- 它为什么用 L1 而不是 L2？—— L1 重建更锐利
- 它为什么还要 GAN？—— 纯重建损失会让结果偏模糊

---

## 1. 文件地图

### 需要精读（5 个）

| 文件 | 行数 | 重要度 | 说明 |
|---|---|---|---|
| `config/model/vq_gan_3d.yaml` | 37 | ★★★★★ | 所有超参集中地，读代码时反复回查 |
| `config/dataset/synt_ct.yaml` | 25 | ★★★★ | 数据侧配置 |
| `train.py` | 123 | ★★★★★ | 入口，调用骨架 |
| `vq_gan_3d/model/vqgan.py` | 718 | ★★★★★ | **核心**，模型 + 训练循环 |
| `vq_gan_3d/model/codebook.py` | 148 | ★★★★ | 向量量化，逻辑独立 |

> **关于行号**：下文中引用的行号是**当前带注释版本**的行号。
> 如果你增删了自己的注释，行号会再次偏移 —— 此时请以**函数名/类名**为准去定位。

### 需要略读（3 个）

| 文件 | 怎么读 |
|---|---|
| `dataset/dataloader.py` | 只读 `get_loader` 的变换链 + `label_transfer`，其余是 MONAI 样板 |
| `vq_gan_3d/utils.py` | 只读 `shift_dim` 和 `adopt_weight` 两个函数 |
| `vq_gan_3d/model/lpips.py` | 知道"用 VGG 特征算感知差异"即可，不必逐行 |

### vqgan.py 内部的组件（标准件，看尺寸即可）

`Encoder` / `Decoder` / `ResBlock` / `SamePadConv3d` / `SamePadConvTranspose3d` /
`NLayerDiscriminator` / `NLayerDiscriminator3D`

这些是常见网络组件，**重点是搞清楚张量形状怎么变**，而不是每个卷积怎么写。

### 直接跳过（死代码，已在文件头加标记）

| 文件 | 为什么跳过 |
|---|---|
| `dataset/dataloader_raw.py` | 全仓库无引用，是 `dataloader.py` 的早期版本 |
| `utils.py` | 全仓库无引用，内容是**目标检测**的残留（NMS、热力图），与 VQGAN 无关 |
| `cross_eval/transfer_txt.py` | 一次性数据清单转换脚本，不被 import |
| `evaluation/` 整个目录 | 整套指标系统无人调用；且 `fid.py` 依赖未安装的 `basicsr`，`niqe.py` 用了已被移除的 `scipy.ndimage.filters` |
| `callbacks.py` 里的 `VideoLogger` 类 | 未注册进 Trainer，且 VQGAN 没有 `log_videos` 方法，双重死代码 |
| `callbacks.py:55` | `torch.clamp` 返回值被丢弃，钳位不生效（无害但别被误导） |

---

## 2. 六步阅读路线

**总原则：按"执行顺序"读，不要按"文件顺序"读。** 代码本身就是 配置 → 入口 → 数据 → 模型 → 训练循环 的线性调用链。

---

### 第 1 步 · 配置：建立"数字从哪来"（10 分钟）

**读**：`config/base_cfg.yaml` → `config/model/vq_gan_3d.yaml` → `config/dataset/synt_ct.yaml`

**目标**：把下面这几个数记住，后面每处代码都能对上号。

| 参数 | 值 | 后面会在哪遇到 |
|---|---|---|
| `roi_x/y/z` | 96 | 数据裁剪尺寸 |
| `downsample` | [4,4,4] | 决定潜空间边长 96/4 = **24** |
| `embedding_dim` | 8 | 潜通道数 |
| `n_codes` | 16384 | 码本大小 |
| `num_samples` | 2 | 每个样本裁几块 |
| `discriminator_iter_start` | 10000 | GAN 何时启动 |
| `max_steps` | 80000 | 总训练步数 |

**注意**：`base_cfg.yaml` 只有 3 行，作用是告诉 hydra 把 dataset 和 model 两份 yaml 合并。

> **自检**
> 1. 模型的输入张量形状是什么？ → `1 × 96 × 96 × 96`（`image_channels: 1` + 3D ROI 96）
> 2. 潜空间边长是多少？为什么是这个数？ → **24**，因为 `96 / 4 = 24`（`downsample` 第一个元素）
> 3. `synt_ct.yaml` 里的 `dataset_list` 能直接用吗？ →
>    **不能**。写的是 `AbdomenAtals_1.0`（拼写错误），而 `cross_eval/recon/` 下实际是
>    `AbdomenAtlas1.0Mini` / `AbdomenAtlas_1.1` / `AbdomenAtlas_8k`，必须命令行覆盖。

---

### 第 2 步 · 入口：`train.py`（20 分钟）

**读**：整个文件，95 行，很短。

**目标**：确认三件事
1. `cfg` 怎么来的（hydra 装饰器自动合并）
2. `VQGAN(cfg)` —— **整个 cfg 对象塞进模型**，不是逐个传超参
3. 训练由 `pl.Trainer` 驱动，`trainer.fit(model, train_loader, val_loader)`

**读的时候重点看这四行**：

```python
val_dataloader = None                              # 第 37 行：验证集被置空
cfg.model.lr = 1 * (1/8.) * (2/4.) * base_lr       # 第 47 行：学习率被改写
model = VQGAN(cfg)                                 # 第 52 行：cfg 整体传入
model.load_from_checkpoint(...)                    # 第 97 行：这行是坏的
trainer.fit(model, train_dataloader, val_dataloader)  # 第 119 行：val 是 None
```

> **自检**
> 1. 总共训练多少步？按 step 还是 epoch？ → **80000 步**，按 step（`max_steps: 80000`，`max_epochs: -1`）
> 2. 学习率到底是多少？ → **1.875e-5**。yaml 写 `3e-4`，但第 47 行乘了 `(1/8)*(2/4) = 1/16`，**差了 16 倍**
> 3. 验证集跑不跑？ → **不跑**。`val_dataloader = None`（第 37 行），所以 `validation_step` 从不执行
> 4. 那 `ModelCheckpoint(monitor='val/recon_loss')` 怎么办？ → 那个指标根本不存在，该回调**实际不起作用**；
>    真正在存权重的是按步数触发的另外两个回调

---

### 第 3 步 · 数据管道：喂进去的到底是什么（30 分钟）

**读**：`dataset/dataloader.py` 的 `get_loader`（第 274 行起）+ `LoadImaged_BodyMap.label_transfer`（第 187 行起）

**目标**：搞清楚输入张量是怎么来的。

**变换链（9 步，按顺序）**：

```
1. LoadImaged_BodyMap  载 CT，并把拆分的器官标注合并成一张整数图
2. AddChanneld         加通道维
3. Orientationd(RAS)   统一朝向
4. Spacingd(1,1,1)     重采样到 1mm 各向同性
5. ScaleIntensityRanged  HU [-175, 250] → [-1, 1]，并截断
6. SpatialPadd(96³)    补齐到至少 96³
7. RandCropByPosNegLabeld  随机裁 96³ 块，pos=20 / neg=1   ★关键
8. RandRotate90d       概率 0.1 的 90 度旋转增强
9. ToTensord           转张量
```

**★ 关键理解点：label 在训练时根本没用。**

`vqgan.py` 的 `training_step` 只取 `batch['image']`。那 label 干嘛用？**只用来给第 7 步决定从哪裁**。

`pos=20, neg=1` 意味着**器官区域的块被采到的概率是背景的 20 倍**。

**为什么这么设计**：这个自编码器只为扩散模型服务，而扩散模型只在**器官区域**工作。
让 AE 把容量浪费在空气、骨头、体表上毫无意义。

所以准确说法是：**STEP1 训练的是一台"器官区域专用"的压缩器。**

**另外要理解 `label_transfer`**：AbdomenAtlas 的标注不是单张 label 文件，而是按器官拆开的：
```
<case>/ct.nii.gz
<case>/segmentations/liver.nii.gz, pancreas.nii.gz, kidney_left.nii.gz, ...
```
代码把它们叠加成一张整数标签图，编号 1~6（肝/胰/肾/肝瘤/胰瘤/肾瘤）。

> **自检**
> 1. `num_samples=2` 会怎样影响 batch 维度？ →
>    每个样本返回 2 个 96³ crop，collate 后 batch 维变成 `batch_size × num_samples`。
>    **这是显存消耗的隐藏来源**（STEP2 里同理，且是首要调优旋钮）
> 2. HU 窗口为什么要用 -175 到 250？ → 覆盖腹部软组织/器官的典型 HU 区间，超出部分截断
> 3. `label_transfer` 里有个复制粘贴 bug，在哪？ → 第 217 行判空写的是 `pancreas_tumor`，
>    但加载的是 `kidney_tumor`，意味着只有存在胰腺肿瘤时才可能读到肾肿瘤标注

---

### 第 4 步 · 模型骨架：`vqgan.py` 的 `__init__` + `encode`/`decode`（30 分钟）

**读**：`vqgan.py` 第 46~139 行（`VQGAN` 类定义 + `__init__` + `encode` + `decode`）+ 第 382 行起的 `Encoder` / 第 434 行起的 `Decoder`

**目标**：走通尺寸链。**这一步先不看损失**，只看数据怎么流动。

**完整尺寸链（已逐行核对过，可放心）**：

```
输入                        1 × 96 × 96 × 96
│
├─ Encoder
│    conv_first (k=3)       1 → 16            96³
│    block 0: strided conv  16 → 32           96³ → 48³
│             + ResBlock
│    block 1: strided conv  32 → 64           48³ → 24³
│             + ResBlock
│    final_block (Norm+SiLU)                   64
│
├─ pre_vq_conv (1×1×1)     64 → 8
│
├─ Codebook                 量化成离散 index
│
├─ post_vq_conv (1×1×1)     8 → 64
│
└─ Decoder
     final_block (Norm+SiLU)                     64      24³
     block 0: convT + 2×Res 64 → 64              24³ → 48³
     block 1: convT + 2×Res 64 → 32              48³ → 96³
     conv_last (k=3)        32 → 1
输出                        1 × 96 × 96 × 96

潜空间 = 8 × 24 × 24 × 24
```

**两个必须注意的不对称**：

1. **Decoder 不是 Encoder 的通道镜像**。Encoder 是 `1→16→32→64`，
   Decoder 是 `64→64→32→1`，两者**并不互逆**。容易看错。
2. **Encoder 每个下采样块只有 1 个 ResBlock，Decoder 有 2 个**（`res1` + `res2`）。

**`encode` / `decode` 的两个模式（重要）**：

```python
vqgan.encode(x, quantize=False)   # 返回【连续】潜表示，不查码本 ← 扩散模型用这个
vqgan.encode(x, quantize=True)    # 返回离散 index
vqgan.decode(latent, quantize=True)  # 先查码本再解码
```

`quantize=False` 保留连续信息，是给 STEP2 扩散模型当条件用的。这个区别后面很关键。

> **自检**
> 1. 编码器输出几通道？潜空间多大？ → **64 通道**，潜空间 **8 × 24 × 24 × 24**
> 2. `SamePadConv3d` 为什么不用 PyTorch 原生卷积？ →
>    原生 `Conv3d` 的 padding 只能补**零**，这里要 **replicate**（复制边缘）填充，避免边界伪影。
>    做法是先用 `F.pad(mode='replicate')` 补够，再让卷积 `padding=0`
> 3. 为什么 `roi` 必须是 4 的倍数？ → `SamePadConv3d` **假设输入尺寸能被 stride 整除**。
>    96 能被 4 整除所以没问题；改成 100 就会报形状错误

---

### 第 5 步 · 训练循环：★ 最难点（60 分钟）

**读**：`vqgan.py` 的 `forward`（第 140 行）→ `training_step`（第 293 行）→ `configure_optimizers`（第 324 行）

> 顺序上先看 `configure_optimizers`（知道有两个优化器）再看 `training_step`（知道怎么交替）也可以，
> 甚至更顺 —— 因为它解释了"为什么这个函数有 `optimizer_idx` 参数"。

**这是全文件唯一的真正难点。** 卡点是 PyTorch-Lightning 的"多优化器"机制。

#### 5.1 两个优化器怎么交替

```python
def configure_optimizers(self):
    ...
    return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
    #       ↑ 优化器0: 自编码器    ↑ 优化器1: 判别器
```

PL 看到返回**两个**优化器，就会**对同一个 batch 调用两次 `training_step`**，用 `optimizer_idx` 区分：

| 调用 | `optimizer_idx` | 训练谁 | 优化器 |
|---|---|---|---|
| 第 1 次 | 0 | 自编码器（生成器） | `opt_ae` |
| 第 2 次 | 1 | 判别器 | `opt_disc` |

**不理解这一点，看到代码里的 `optimizer_idx == 0 / 1` 会完全懵。**

#### 5.2 自编码器分支的五个损失（`optimizer_idx == 0`）

```
总损失 = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
```

| 损失 | 权重 | 是什么 | 实现位置 |
|---|---|---|---|
| `recon_loss` | `l1_weight: 4.0` | L1 重建，**主力** | `vqgan.py` forward |
| `commitment_loss` | 0.25 | 逼编码器输出靠近码本 | `codebook.py` |
| `perceptual_loss` | `perceptual_weight: 4.0` | LPIPS 感知相似度 | `lpips.py` |
| `aeloss` | `image/video_gan_weight: 1.0` | 对抗损失（骗判别器） | `vqgan.py` |
| `gan_feat_loss` | `gan_feat_weight: 4.0` | GAN 特征匹配，稳定训练 | `vqgan.py` |

#### 5.3 ★ 关键设计：前 10000 步没有 GAN

```python
disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
aeloss = disc_factor * g_loss
```

`adopt_weight` 在 `global_step < 10000` 时返回 **0**。

**后果非常关键**：生成器和判别器**两边都乘了这个系数**，所以
- 前 10000 步：**没有对抗损失**，是纯重建训练（L1 + 量化 + 感知）
- 前 10000 步：**判别器也完全不更新**，它的参数一直是随机初始化的

**为什么**：训练初期重建质量差，此时施加对抗梯度只会把模型带偏。

#### 5.4 2D / 3D 混合判别器

`forward` 里有段巧妙的处理。因为 LPIPS 和 2D 判别器都是 **2D** 的，所以要从 3D 体积里抽切片：

```python
frame_idx_T = torch.randint(0, D, [B]).cuda()
frames_T = torch.gather(x, 2, ...).squeeze(2)   # 沿 T 轴抽一层
# 同理沿 H 轴、W 轴各抽一层
```

于是得到三组 2D 切片，对它们用 **2D 判别器 + 2D LPIPS**；
而完整 3D 体积用 **3D 判别器**。两者损失加权求和——既有 2D 预训练 VGG 的强先验，又保证 3D 整体一致。

> **自检**
> 1. 一个 batch 会调用几次 `training_step`？ → **两次**，`optimizer_idx` 分别是 0 和 1
> 2. 第 5000 步时，判别器更新了吗？ → **没有**。`disc_factor = 0`，判别器损失被乘 0，参数保持随机初始化
> 3. 为什么要沿三个轴抽 2D 切片，而不是只抽一个方向？ →
>    三个方向各抽一次能得到互补的视图；只用 T 轴会丢失 H/W 方向的结构信息
> 4. `vqgan.py` 里哪几行硬编码了 `.cuda()`？ → 抽切片的 `torch.randint(...).cuda()` 三处。
>    纯 CPU 跑 `forward` 会直接崩

---

### 第 6 步 · 码本：独立的一块（30 分钟）

**读**：`codebook.py` 全文，108 行。

**为什么放最后**：它**不靠梯度更新**，用的是 EMA——跟常规神经网络训练是两套完全不同的逻辑，混在一起读容易乱。

**码本的作用**：把编码器输出的**连续**向量，替换成码本里离它最近的**离散**向量。
本仓库配置 16384 个码字，每个 8 维。

**三个 buffer**：
- `embeddings` (16384×8) —— 码本向量
- `N` (16384) —— 每个码字被分配到的次数
- `z_avg` (16384×8) —— 每个码字对应的特征均值

**`forward` 的五件事**：

1. **首次初始化**：从当前 batch 的真实特征里随机挑 16384 个当初始码本
   （比纯随机初始化好得多，随机码字离真实数据太远）
2. **找最近邻**：用展开的平方距离公式 `‖a‖² - 2a·b + ‖b‖²`（比广播相减省显存），`argmin` 得 index
3. **EMA 更新码本**（不靠梯度）：
   ```python
   self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
   self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)
   self.embeddings.data.copy_(self.z_avg / weights.unsqueeze(1))
   ```
4. **死码字重启**：使用次数低于 `restart_thres` 的码字，用随机真实特征替换，
   防止**码本塌缩**（大量码字永不被选中 → 有效码本越来越小 → 潜空间退化）
5. **直通估计器**（关键技巧）：
   ```python
   embeddings_st = (embeddings - z).detach() + z
   ```
   前向：结果等价于 `embeddings`（离散）
   反向：`(embeddings - z).detach()` 的梯度为 0，所以 `d(embeddings_st)/dz = 1`，梯度原样透传

**监控指标 `perplexity`**：码本使用分布的熵的指数，越接近 16384 说明用得越均匀。
这是判断 VQ 是否塌缩的**核心指标**，训练时被 log 成 `train/perplexity`。

> **自检**
> 1. 码本向量怎么更新？ → **EMA**（0.99 旧 + 0.01 新），不靠梯度
> 2. 直通估计器为什么能让离散量化也有梯度？ →
>    前向走离散值，反向因为 `.detach()` 让梯度等于 1，原样传给编码器
> 3. 训练时该盯哪个指标判断码本有没有塌缩？ → **`train/perplexity`**
> 4. `commitment_loss` 为什么不回传给码本？ → 它有 `.detach()`，**只逼编码器**靠近码本，
>    码本自己由 EMA 维护

---

## 3. 三条主线（贯穿始终，建议记在纸上）

### 主线一：尺寸链

```
96³ ──(Encoder ÷4)──> 24³ ──(pre_vq_conv)──> 8×24³ ──(Codebook)──> index
                                                          │
1×96³ <──(Decoder ×4)── 24³ <──(post_vq_conv)── 8×24³ <───┘
```

**记住 `8` 和 `24`**：它们会原封不动变成 STEP2 的 `out_dim: 8` 和 `diffusion_img_size: 24`。
整条流水线的尺寸就是这么传下去的。

### 主线二：损失表

| 阶段 | 自编码器（idx=0） | 判别器（idx=1） |
|---|---|---|
| step < 10000 | `recon×4 + commit + perceptual×4` | **不更新**（损失 ×0） |
| step ≥ 10000 | 上面 + `aeloss + gan_feat×4` | `d_image + d_video` |

### 主线三：优化器交替

```
DataLoader 出一个 batch
   └─> training_step(idx=0) → 更新 opt_ae
   └─> training_step(idx=1) → 更新 opt_disc
```

---

## 4. 最容易卡住的四个地方

按踩坑概率排序：

**1. `optimizer_idx` 的双分支**（第 5 步）
PL 的多优化器机制。没有写过 PL 的人会觉得"为什么同一个函数被调用两次"。**这是最大的卡点。**

**2. 直通估计器 `(embeddings - z).detach() + z`**（第 6 步）
"前向离散、反向连续"这个技巧。盯着看很久也想不通是正常的，
记住结论：**前向结果等于离散值，反向梯度等于 1**。

**3. 2D/3D 混合判别器**（第 5 步）
为什么要抽切片？因为 VGG（LPIPS）和 2D 判别器都是 2D 网络，吃不了 3D。

**4. Decoder 通道数不是 Encoder 的镜像**（第 4 步）
源码里写成了 `n_hiddens*2**(max_us-i)` 这种不易读的形式，展开后是 `64→64→32→1`。

---

## 5. 已确认的坑与 bug 清单

读代码时遇到这些地方**不要怀疑自己看错了，是代码本身有问题**：

| 位置 | 问题 | 影响 |
|---|---|---|
| `train.py:47` | `lr` 被静默改写为 `(1/8)*(2/4)*base_lr`，即 **1.875e-5** | 只读 yaml 会把学习率算错 16 倍 |
| `train.py:97` | `model.load_from_checkpoint(...)` **返回值被丢弃** | 用官方 ckpt 做"加载权重再微调"这条路是坏的。正确写法 `model = VQGAN.load_from_checkpoint(...)` |
| `train.py:37,61` | `val_dataloader=None`，但 checkpoint 回调 `monitor='val/recon_loss'` | 验证从不跑，该回调不起作用 |
| `callbacks.py:55` | `torch.clamp(...)` 返回值没赋回去 | 钳位不生效（因为范围本就是 [0,255]，通常无害；越界时 `astype(np.uint8)` 会溢出回绕） |
| `dataloader.py:217` | 判空用 `pancreas_tumor`，加载的是 `kidney_tumor` | 肾肿瘤标注加载条件错误 |
| `config/dataset/synt_ct.yaml:4` | `dataset_list: 'AbdomenAtals_1.0'` 拼写错误且文件不存在 | 必须命令行覆盖 |
| `lpips.py:111` | `if name is not "vgg_lpips"` 字符串用 `is not` 比较 | Python 3.12 会报 `SyntaxWarning`，靠字符串驻留凑巧能跑 |
| `lpips.py:85` | `vgg16(pretrained=True)` | **会联网下载** ImageNet VGG16，离线环境构造 `LPIPS()` 会失败 |
| `vq_gan_3d/utils.py` | `ForkedPdb` 打开 `/dev/stdin` | 仅限 Linux，Windows 不可用（该函数无人调用） |

---

## 6. 速通版（30 分钟）

如果时间紧，按这个最小路径读：

1. **`config/model/vq_gan_3d.yaml`**（3 分钟）—— 记住 `roi=96`、`downsample=4`、`discriminator_iter_start=10000`
2. **`train.py`**（7 分钟）—— 只看 34 / 38 / 91 三行
3. **`vqgan.py` 的 `__init__` + `encode`/`decode`**（8 分钟）—— 走通尺寸链
4. **`vqgan.py` 的 `training_step` + `configure_optimizers`**（8 分钟）—— 理解两个优化器
5. **`codebook.py` 的 `forward`**（4 分钟）—— 看 EMA 更新和直通估计器

**30 分钟后你应该能回答**：这个模型压缩后是什么形状？一个 batch 走几次前向？GAN 什么时候开始起作用？

---

## 7. 术语对照表

| 代码里的名字 | 含义 |
|---|---|
| `VQGAN` | Vector-Quantized GAN，本模块的主体 |
| `pre_vq_conv` / `post_vq_conv` | 量化**前**压到 embedding_dim / 量化**后**还原通道 |
| `Codebook` / `embeddings` | 码本 / 码本里的向量（16384 × 8） |
| `encodings` / `encoding_indices` | 量化后的离散下标 |
| `commitment_loss` | 承诺损失，逼编码器靠近码本 |
| `perplexity` | 码本使用均匀度，**判断是否塌缩的核心指标** |
| `straight-through` | 直通估计器，让离散量化可导 |
| `roi` | Region of Interest，这里指训练裁出的 96³ 块 |
| `pos` / `neg` | 正样本（器官）/ 负样本（背景）的采样比 |
| `adopt_weight` | "训练到某步才启用某损失"的开关 |
| `disc_factor` | 上述开关的返回值（0 或 1） |
| `n_hiddens` | 基础通道数 = 16 |

---

## 8. 与下游 STEP2 的衔接

读完 STEP1 后，你应该带着这些认知进入 STEP2：

| STEP1 里的东西 | 在 STEP2 里的对应 |
|---|---|
| 潜空间 `8 × 24 × 24 × 24` | `out_dim: 8` + `diffusion_img_size: 24` + `diffusion_depth_size: 24` |
| `encode(quantize=False)` 连续潜表示 | 扩散模型的**条件输入** |
| VQGAN 权重 `AutoencoderModel.ckpt` | STEP2 的 `model.vqgan_ckpt` |
| `96³` ROI | 扩散的条件图尺寸 |

**一句话**：STEP1 的产物是一条 `96³ 连续CT → 8×24³ 潜向量` 的通道，
STEP2 要在这个潜空间里做扩散。所以 STEP1 压得好不好，直接决定 STEP2 的上限。

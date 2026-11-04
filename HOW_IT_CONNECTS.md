# DiffTumor 三阶段是如何串通的

> 这份文档解释 STEP1 / STEP2 / STEP3 之间的**接口**：张量尺寸怎么传、
> 权重文件怎么传、哪些代码是复制品、每个阶段各自在训练什么。
>
> 各阶段的详细阅读指南：
> - [`STEP1.AutoencoderModel/READING_ORDER.md`](STEP1.AutoencoderModel/READING_ORDER.md)
> - [`STEP2.DiffusionModel/READING_ORDER.md`](STEP2.DiffusionModel/READING_ORDER.md)
> - [`STEP3.SegmentationModel/READING_ORDER.md`](STEP3.SegmentationModel/READING_ORDER.md)

---

## 0. 一句话总览

```
STEP1  学"怎么把 CT 压缩到潜空间"        →  产出一个 VQGAN
STEP2  在潜空间里学"早期肿瘤长什么样"     →  产出两套扩散模型
STEP3  用扩散模型造肿瘤，训分割网络        →  产出分割模型
```

论文的核心主张贯穿三步：**小于 2cm 的早期肿瘤在不同腹部器官里长得非常像**，
所以可以只在肝脏上学，然后合成到胰腺、肾脏上。

---

## 1. 全局数据流

```
                    ┌──────────────────────────────────────┐
                    │  STEP1  VQGAN（自编码器）              │
   健康CT 96³  ───►  │  1×96³ ──编码──► 8×24³ ──解码──► 1×96³ │
   (AbdomenAtlas)   │  训练目标：重建（L1+感知+GAN）          │
                    └──────────────┬───────────────────────┘
                                   │ 产出 AutoencoderModel.ckpt
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
   ┌─────────────────────────┐              ┌─────────────────────────┐
   │  STEP2  潜空间扩散       │              │  STEP3  分割            │
   │                         │              │                         │
   │  在 8×24³ 上学：         │              │  被当"肿瘤生成器"调用    │
   │  "抹掉肿瘤→补回来"       │──早期/中晚期─►│                         │
   │                         │   扩散权重    │  健康CT + 合成肿瘤       │
   │  训练数据：MSD-Liver     │              │       ↓                 │
   │  （真实肿瘤病例）         │              │  训练分割网络            │
   └─────────────────────────┘              │       ↓                 │
                                            │  在真实肿瘤上评测        │
                                            └─────────────────────────┘
```

**关键点**：STEP2 和 STEP3 是**并列**的消费者，都依赖 STEP1 的 VQGAN。
STEP2 产出扩散权重后再喂给 STEP3。

---

## 2. 接口一：尺寸链（最重要的一条线）

整条流水线的尺寸环环相扣，**改一处会连累多处**：

```
STEP1 的配置：
    roi_x/y/z          = 96          ← 训练块大小
    downsample         = [4,4,4]     ← 每轴下采样 4 倍
    embedding_dim      = 8           ← 潜通道数
                    ↓
    潜空间 = 8 × (96/4)³ = 8 × 24³
                    ↓
STEP2 的配置（必须对齐）：
    diffusion_img_size   = 24        ← = 96/4
    diffusion_depth_size = 24        ← 三个轴都是 24
    out_dim              = 8         ← = embedding_dim
    diffusion_num_channels = 17      ← = 8(噪声) + 8(条件潜) + 1(mask)
                    ↓
STEP3：
    分割网络 roi         = 96        ← 与 STEP1 相同（合成也在这个分辨率做）
    合成时仍走 VQGAN → 8×24³
```

**为什么必须一致**：VQGAN 的编码器把 96³ 降到 24³，如果 STEP2 的
`diffusion_img_size` 不是 24，U-Net 输出的形状就和 VQGAN 解码器期望的对不上。

### 17 这个数字的来历（两处出现，必须一致）

```
U-Net 输入 = 噪声潜 + 条件
           = 8      + 9
                      └─ 8(被抹掉的潜表示) + 1(mask)  = 9
```

这个构造在 **STEP2 的训练**（`ddpm/diffusion.py` 的 `forward`）
和 **STEP3 的合成**（`TumorGeneration/utils.py` 的 `synthesize_early_tumor`）
里是**完全一样**的 —— 这就是两个阶段之间真正的"接口协议"。

> 对照阅读这两处代码，你会发现构造 cond 的三行几乎逐字相同。
> 理解了这个，就理解了 STEP2 和 STEP3 是怎么对接的。

---

## 3. 接口二：权重文件的流转

```
STEP1 产出
  └─ AutoencoderModel.ckpt
       ├─► STEP2 的 model.vqgan_ckpt           （训练时编码/解码）
       └─► STEP3 的 TumorGeneration/model_weight/AutoencoderModel.ckpt
                                               （合成时编码/解码）

STEP2 产出
  ├─ {organ}_early.pt     （早期肿瘤，timesteps=4）
  └─ {organ}_noearly.pt   （中晚期肿瘤，timesteps=200）
       └─► STEP3 的 TumorGeneration/model_weight/
              └─ 由 synt_model_prepare() 加载

STEP3 产出
  └─ 分割模型权重（unet / nnunet / swinunetr 三种）
       └─ 供 validation.py 评测
```

**加载方式的三个细节**：

1. **VQGAN 的结构从 checkpoint 里重建**，不是从 yaml。
   `VQGAN.load_from_checkpoint(path)` 内部读的是 ckpt 里的 `hyper_parameters['cfg']`。
   所以即使某个 yaml 里的值对不上，也不影响 —— 但也意味着**yaml 不能用来推断结构**。

2. **扩散权重有两个版本，都要下**：
   - `liver_early.pt` → 生成 tiny/small 肿瘤
   - `liver_noearly.pt` → 生成 medium/large 肿瘤

3. **实际采样用的是 `ema` 参数**，不是 `model` 参数：
   - 早期：`Tester.load()` 会同时载 model 和 ema，采样用 `tester.ema_model`
   - 中晚期：直接 `load_state_dict(ckpt['ema'])`

---

## 4. 接口三：代码的三份拷贝（重要陷阱）

仓库里 `vq_gan_3d` 和 `ddpm` 被复制了多份，**彼此已经分叉**。
读代码时如果不知道这件事，会非常困惑。

### `vq_gan_3d` —— 三份

| 位置 | 与 STEP1 的关系 | 怎么处理 |
|---|---|---|
| `STEP1.AutoencoderModel/vq_gan_3d/` | **原版**（已加详细中文注释） | 读这份 |
| `STEP2.DiffusionModel/vq_gan_3d/` | **逐字节相同的拷贝**（0 行差异） | 跳过 |
| `STEP3.../TumorGeneration/ldm/vq_gan_3d/` | **分叉**（约 106 行差异） | 跳过，看文件头说明 |

**STEP3 那份的差异主要是三类**：
1. import 改成相对路径（因为是子包）
2. 通篇是**调试用的形状注释**（且数字是按作者的配置写的，与 STEP1 不同）
3. `codebook.py` 里有个**未注释掉的 `breakpoint()`**

### `ddpm` —— 两份

| 位置 | 定位 | 关键差异 |
|---|---|---|
| `STEP2.DiffusionModel/ddpm/` | **训练版** | 有完整的 `Trainer` 类（训练循环/EMA/存档） |
| `STEP3.../TumorGeneration/ldm/ddpm/` | **推理版** | **`Trainer` 被整个删掉**，只留 `Tester` 和采样 |

**还有一个必须注意的差异**：
STEP3 那份的 `GaussianDiffusion.__init__` **多了一个 `device` 参数**，
而 STEP2 那份**不接受**这个参数。

```python
# STEP3 的 synt_model_prepare 会这样调用：
GaussianDiffusion(..., device=device)     # ← STEP2 那份会 TypeError
```

**所以两份代码绝对不能混用。** 改 STEP2 时不要指望 STEP3 跟着变，反之亦然。

### 复制清单速查

```bash
# 验证 STEP2 的 vq_gan_3d 与 STEP1 完全一致
git diff --no-index --ignore-cr-at-eol \
  STEP1.AutoencoderModel/vq_gan_3d/ \
  STEP2.DiffusionModel/vq_gan_3d/
```

---

## 5. 三个阶段的"性格"完全不同

读代码前先知道这一点，能省很多困惑：

| | STEP1 | STEP2 | STEP3 |
|---|---|---|---|
| **在学什么** | 重建（压缩→还原） | 生成肿瘤纹理 | 分割肿瘤 |
| **网络** | VQGAN（编码器+解码器+判别器） | U-Net（去噪） | U-Net / nnU-Net / Swin-UNETR |
| **训练框架** | **PyTorch-Lightning** | **手写训练循环** | **手写训练循环** |
| **输入分辨率** | 96³ | 24³（潜空间） | 96³ |
| **数据来源** | AbdomenAtlas | MSD-Liver | MSD-Liver + HealthyCT |
| **强度归一化** | `[-175,250]→[-1,1]` | `[-175,250]→[-1,1]` | `[-175,250]→[0,1]` |
| **损失** | L1 + 感知 + GAN + 量化 | 噪声的 L1 | DiceCE |
| **优化器数** | **2 个**（AE + 判别器） | 1 个 | 1 个 |
| **有验证集吗** | 没有 | 没有 | **有**（`val_epoch`） |
| **训练时长参考** | 80000 步 | 60000 步 | 2000 epochs |

**三个最容易混淆的差异**：

1. **只有 STEP1 用 PyTorch-Lightning**。STEP2 和 STEP3 都是手写的。
   所以 STEP1 里 `optimizer_idx` 那种 PL 特有的双优化器机制，在另两步不存在。
2. **归一化终点不同**：STEP1/STEP2 到 `[-1,1]`，STEP3 到 `[0,1]`。
   （STEP3 的合成函数内部会自己 `*2-1` 转回去喂扩散模型。）
3. **只有 STEP3 有验证**。STEP1 和 STEP2 的 `val_dataloader` 都是 `None`。

---

## 6. 一次完整的数据旅程（把三步串起来看）

以"给一张健康肝脏 CT 合成肿瘤，然后训练分割"为例：

```
① 健康CT 读入
   STEP3 main.py → 重采样 1mm → HU[-175,250] → [0,1] → 裁 96³
   （裁剪时要求器官体素足够多，保证有地方放肿瘤）

② 选位置和形状                                    ← STEP3 几何阶段
   random_select 在器官中部找点（找不到就逐级放宽）
   get_ellipsoid 造椭球 → elasticdeform 形变
   → 得到肿瘤 mask（比如 small，半径 8）

③ 抹掉该区域，送进 VQGAN 编码                      ← 用的是 STEP1 的产物
   1×96³ → 8×24³（连续潜表示，quantize=False）

④ 构造条件
   cond = concat[被抹掉的潜表示(8ch), 下采样mask(1ch)] = 9ch

⑤ 扩散采样                                        ← 用的是 STEP2 的产物
   small 属于早期 → 用 liver_early.pt，走完整 DDPM 4 步
   噪声(8×24³) ──4 步去噪──► 潜表示(8×24³)

⑥ 解码回图像                                      ← 又回到 STEP1 的 VQGAN
   8×24³ → 1×96³

⑦ 高斯羽化混合
   final = (1-α)·原CT + α·合成CT    （α 是模糊后的 mask）

⑧ 得到训练样本
   图像 = 带合成肿瘤的 CT
   标签 = {0:背景, 1:器官, 2:肿瘤}

⑨ 和真实病例混在一起训练分割网络
```

**注意第 ⑤ 步**：早期肿瘤只走 **4 步**扩散。这正是论文"早期肿瘤简单"这个论点
在计算上的直接体现 —— 也是为什么这个方法的合成速度快到可以放进训练循环里在线做。

---

## 7. 跨阶段速查表

| 你想知道 | 去哪看 |
|---|---|
| 潜空间为什么是 24 | `STEP1/config/model/vq_gan_3d.yaml` 的 `downsample` |
| 17 通道怎么来的 | `STEP2/ddpm/diffusion.py` 的 `forward`（搜 `cond = torch.cat`） |
| 扩散在训练什么 | `STEP2/ddpm/diffusion.py` 的 `p_losses` |
| 为什么早期只要 4 步 | `STEP2/config/model/ddpm.yaml` 的 `timesteps` + STEP1 的 ROI |
| 肿瘤长什么样由谁决定 | `STEP3/TumorGeneration/utils.py` 的 `get_fixed_geo`（四档半径） |
| 肿瘤纹理由谁决定 | 同文件的 `synthesize_early_tumor`（第 5 步扩散采样） |
| 合成是什么时候做的 | `STEP3/monai_trainer.py` 的 `train_epoch`（每个 epoch 在线做） |
| 评测看哪个指标 | `STEP3/validation.py` 的 `tumor_dice` / `tumor_nsd` |
| 哪些代码是重复的 | 本文档第 4 节 |
| 哪些文件是死代码 | 各阶段 READING_ORDER.md 的"文件地图"一节 |

---

## 8. 三个阶段的共同"怪癖"（贯穿全仓库）

这几点在三个阶段都会遇到，理解了就不用每次都重新困惑：

**1. `permute(0,1,-1,-3,-2)` 的坐标系调整**
来自 VQGAN 原本处理视频的 `(B,C,T,H,W)` 约定。
它在 STEP1 的 `training_step`、STEP2 的 `forward`、STEP3 的 `synthesize_*` 里
都出现过，且**必须与 `permute(0,1,-2,-1,-3)` 成对出现**（两者互逆）。
本身没有物理意义，是历史包袱。

**2. 视频模型术语的残留**
`num_frames` 实际是 CT 的**深度维度**，`temporal_attn` 实际是**沿深度轴的注意力**，
`Upsample/Downsample` 的 kernel 是 `(1,4,4)` —— **深度方向根本不采样**。
这些都是从视频扩散（TATS / video-diffusion-pytorch）改写来的痕迹。

**3. 硬编码的 `.cuda()`**
多处的 `torch.randint(...).cuda()`、`torch.device("cuda")`
意味着**整条链路只能在 CUDA 上跑**，纯 CPU 环境会崩。

**4. 路径是字符串拼接而非 `os.path.join`**
所以配置里的 `data_root_path`、`label_root_path`、`healthy_data_root`
**都必须以 `/` 结尾**。少一个斜杠就静默找不到文件。

**5. `_label` 后缀是"健康CT"的标识**
这个约定贯穿 STEP3：文件名含 `liver_label` / `pancreas_label` / `kidney_label`
就表示这是健康CT（只带器官标注），需要注入合成肿瘤。
`main.py` 的数据配对、`monai_trainer.py` 的合成分支、
`validation.py` 的变体判断都靠它。

---

## 9. 一句话记住三者关系

> **STEP1 是压缩器，STEP2 是画肿瘤的手，STEP3 是用画出来的肿瘤教一个学生认肿瘤。**
>
> 压缩器只在 96³ ↔ 24³ 之间搬东西；
> 画肿瘤的手只认 24³ 的潜空间和 9 通道的条件；
> 学生（分割网络）只看 96³ 的最终图像和 {0,1,2} 的标签。

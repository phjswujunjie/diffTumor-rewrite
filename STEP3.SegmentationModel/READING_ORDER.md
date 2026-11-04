# STEP3 SegmentationModel 代码阅读指南

> 配套说明：本目录源码已加中文注释。本文档告诉你**按什么顺序读**、**每步该看懂什么**、
> **怎么自检**，以及**哪些文件不用读**。
>
> 行号是**当前带注释版本**的。你自己再增删注释就会偏移，届时以函数名/类名为准定位。
>
> 三阶段如何串通，见仓库根目录的 `../HOW_IT_CONNECTS.md`。

---

## 0. 先建立一句话认知

**STEP3 把 STEP1 的 VQGAN 和 STEP2 的扩散模型当成一台"肿瘤生成器"，
用它给健康 CT 造出带肿瘤的假样本，再拿这些假样本去训练分割网络。**

```
健康CT ──┐
         ├─► 合成肿瘤 ──► 带肿瘤的假CT ──┐
STEP2扩散 ┘                              ├─► 训练分割网络
                                         │
真实肿瘤病例(MSD-Liver) ─────────────────┘
```

**三个必须先破除的误解**：

**误解一：以为合成是"离线预生成一个数据集"。**
不是。合成是在**训练循环内部、每个 epoch 实时做的**（`monai_trainer.py:205` 的 `train_epoch`）。
所以每轮看到的合成肿瘤都不一样，等于无限数据增强。

**误解二：以为有一个"生成肿瘤"的独立脚本。**
没有。`TumorGeneration/` 是个**函数库**，不是可执行程序。
仓库里唯一能独立跑的推理入口是 `direct_inference/` 里的 Singularity 容器。

**误解三：以为整个训练集都是合成的。**
不是。训练集里**混着两类样本**：
- **真实肿瘤病例**（MSD-Liver）→ 原样使用，不做任何合成
- **健康 CT**（HealthyCT）→ 训练时注入一个合成肿瘤，约一半概率

---

## 1. 文件地图

### 需要精读（4 个）

| 文件 | 行数 | 重要度 | 说明 |
|---|---|---|---|
| `main.py` | 591 | ★★★★★ | 入口：数据配对、模型选择、变换链 |
| `monai_trainer.py` | 547 | ★★★★★ | **训练循环 + 在线合成**（第 205 行 `train_epoch`） |
| `TumorGeneration/utils.py` | 577 | ★★★★★ | **合成实现**：几何生成 + 扩散填充 |
| `validation.py` | 335 | ★★★★ | 评测：Dice 和 NSD |

### 需要略读（4 个）

| 文件 | 怎么读 |
|---|---|
| `monai_trainer.py` 的 `val_epoch` / `run_training` | 常规训练框架代码，知道流程即可 |
| **`networks/`** | 分割骨干的实现（UNETR / SwinUNETR 等）。**只需知道"是可选骨干"，不必逐层读**。详见 `networks/README.md` |
| **`optimizers/lr_scheduler.py`** | **在路径上**——`main.py:49` 使用。标准 warmup + cosine 调度器 |
| `dynunet_pipeline/create_network.py`、`task_params.py` | 只在 `--model_name nnunet` 时被用到，提供 DynUNet 的核尺寸/stride |

### 不需要重读（STEP1 / STEP2 的拷贝）

| 目录 | 情况 |
|---|---|
| `TumorGeneration/ldm/vq_gan_3d/` | STEP1 的**分叉拷贝**（约 106 行差异，主要是相对 import + 调试注释） |
| `TumorGeneration/ldm/ddpm/` | STEP2 的**推理精简版**（删掉了 Trainer，只留采样） |

两份都有文件头说明差异，中文详解看 STEP1/STEP2 的原版。
其中 `ldm/ddpm/util.py` **在路径上**（被 `ddim.py` 使用），其余是拷贝。

### 直接跳过（死代码，已加标记）

| 目录/文件 | 为什么 |
|---|---|
| **`networks2/`**（整个目录，8 个文件） | **完全无人引用**。看本目录的 `networks2/README.md`。实际用的是 `networks/` |
| **`networks/basicunetplusplus.py`** | 零引用（原打算做第四种骨干，没接进 main.py） |
| `TumorGeneration/diffusion_config/vq_gan_3d.yaml` | **从未被读取**。`synt_model_prepare` 只 compose 了 `ddpm.yaml` |
| `TumorGeneration/ldm/ddpm/unet.py` | 零引用，且其绝对导入 `from ddpm.time_embedding import ...` 在 STEP3 包结构下**本来就会 ImportError** |
| `TumorGeneration/ldm/ddpm/{time_embedding,text}.py` | 间接死代码（BERT 路径永不触发 / 只被 unet.py 引用） |
| `postprocess.py` | 零引用。其逻辑已被 `validation.py` 的 `denoise_pred` 等替代 |
| `dynunet_pipeline/`（其余部分） | 只在选 `--model_name nnunet` 时被 `main.py:391` 引用到两个函数，其余是 nnU-Net 的训练框架，本流程用不到（详见该目录 README.md 开头的阅读提示） |

---

## 2. 六步阅读路线

### 第 1 步 · 入口：`main.py` 的 `main_worker`（30 分钟）

**读**：`main.py` 第 319 行的 `main_worker`

**重点看数据配对那段**（搜 `train_txt =`）：

```python
if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
    train_img_real.append(healthy_data_root + ...)    # ← 健康CT
else:
    train_img_healthy.append(data_root + ...)         # ← 真实病例
```

**★ 这里有个极易踩的坑：变量名是反的。**

| 变量名 | 实际存的 | 来源 |
|---|---|---|
| `train_img_real` | **健康 CT** | `healthy_data_root` |
| `train_img_healthy` | **真实肿瘤病例** | `data_root` |

命名和内容完全相反。判断依据只有一个：**文件名里有没有 `<器官>_label`**。

> **自检**
> 1. 怎么区分一个样本是"健康CT"还是"真实病例"？ → 看 label 文件名里有没有 `liver_label` 之类
> 2. 为什么健康CT要单独处理？ → 它们只有器官标签，肿瘤要靠合成补上
> 3. 验证集里有合成样本吗？ → **没有**。验证集全是真实 MSD 病例，
>    这样才能评估"合成数据训出来的模型在真肿瘤上的表现"

---

### 第 2 步 · 变换链与自定义裁剪（25 分钟）

**读**：`main.py` 第 240 行的 `_get_transform` + 第 147 行的 `RandCropByPosNegLabeld_select`

**与 STEP1/STEP2 的关键差异**：

| | STEP1 / STEP2 | STEP3 |
|---|---|---|
| 强度归一化 | `[-175,250] → [-1,1]` | **`[-175,250] → [0,1]`** |
| 裁剪策略 | 固定比例采样 | **带"前景体素数下限"的重试** |

**归一化不同很重要**：分割网络习惯 [0,1]，而合成函数 `synthesize_*` 也假设
输入是 [0,1]（内部会自己 `*2-1` 转成 [-1,1] 喂给扩散模型）。

**自定义裁剪 `RandCropByPosNegLabeld_select` 在做什么**：

```python
if '..._label' in data_name:        # 只对健康CT生效
    while True:
        d_crop = super().__call__(d)
        pixel_num = (d_crop[0]['label']>0).sum()     # 器官体素数
        if pixel_num > self.R2voxel(self.fg_thresh):
            break                                     # 够了就接受
        if flag>5 and pixel_num > self.R2voxel(fg_thresh-5):
            break                                     # 否则逐步放宽
        ... 每 5 次放宽 5mm ...
        if flag>30: break                             # 兜底，防死循环
```

**为什么需要**：健康 CT 的标签只有器官。如果随便裁一块，可能裁到一片没器官的区域，
后面就没法在上面"放"合成肿瘤了。所以要求裁出的块里有足够的器官体素。

`fg_thresh` 是个**等效球半径**（mm），由器官决定：**肝 30 / 胰 15 / 肾 25**。

> **自检**
> 1. `fg_thresh` 的单位是什么？ → 毫米（等效球半径），
>    通过 `R2voxel()` 换算成体素数
> 2. 为什么要"逐步放宽"而不是直接要求？ → 器官小的切片可能永远达不到阈值，
>    逐步放宽 + 30 次上限可以保证不死循环

---

### 第 3 步 · ★ 在线合成：`train_epoch`（40 分钟，本步最重要）

**读**：`monai_trainer.py` 第 205 行的 `train_epoch`

**这是整个 STEP3 的核心。** 结构很简单，但要理解它在干什么：

```python
# ① 每个 epoch 开头加载一次合成模型
vqgan, early_sampler, noearly_sampler = synt_model_prepare(...)
# ★ 这三个模型会常驻显存直到该 epoch 结束

for idx, batch_data in enumerate(loader):
    data, target = ... cuda()

    # ② 逐个样本判断
    for bs in range(data.shape[0]):
        if 'liver_label' in data_name:            # 是健康CT
            if random.random() > 0.5:             # ~50% 概率注入肿瘤
                # ③ 随机选尺寸：80% 早期，中晚期各 10%
                tumor_probs = [0.8, 0.1, 0.1]

                if 'early':   synthesize_early_tumor(...)      # 完整 DDPM
                if 'medium':  synthesize_medium_tumor(...)     # DDIM
                if 'large':   synthesize_large_tumor(...)      # DDIM

                # ④ 原地改写 batch
                data[bs,...] = synt_data[0]
                target[bs,...] = synt_target[0]

    # ⑤ 用改写后的 batch 训练
    logits = model(data)
    loss = loss_func(logits, target)
```

**三个关键数字**：

| 参数 | 值 | 含义 |
|---|---|---|
| `sample_thresh` | 0.5 | 约一半健康CT会被注入合成肿瘤（另一半当负样本） |
| `tumor_probs` | `[0.8, 0.1, 0.1]` | **80% 是早期肿瘤** —— 与论文聚焦点一致 |
| `ddim_ts` | 50 | 中晚期肿瘤的 DDIM 采样步数 |

**★ 显存上的重要后果**：`synt_model_prepare` 加载的
**VQGAN + 两套扩散模型会在整个 epoch 常驻显存**，和分割网络、它的激活值挤在一起。
这是 STEP3 显存开销的大头，也是为什么官方要 `--noamp`（关掉混合精度）——
他们用 A100 80GB 不在乎，小显存卡上就要斟酌了。

> **自检**
> 1. 合成是一次性做完还是每轮重做？ → **每轮（epoch）重做**，所以每个 epoch 看到的都不同
> 2. 一个 batch 里有真实病例也有健康CT，怎么区分处理？ → 靠文件名，
>    健康CT走合成分支，真实病例原样保留
> 3. 为什么 80% 都用早期肿瘤？ → 论文的核心论点就是早期肿瘤跨器官相似、
>    标注稀缺、最需要合成弥补；中晚期本身在真实数据里就有
> 4. 显存里同时住着哪些模型？ → 分割网络 + VQGAN + 2 个扩散 U-Net（约等于 STEP2 的两套）

---

### 第 4 步 · ★ 合成实现：`TumorGeneration/utils.py`（50 分钟，本步最难）

**读**：第 129 行 `get_fixed_geo` → 第 279 行 `synt_model_prepare` → 第 359 行 `synthesize_early_tumor`

合成分**两个阶段**，要分清楚：

#### 阶段 A：几何（造出"肿瘤应该长在哪、什么形状"）

```
random_select  → 在器官内部随机挑一个放置点
get_ellipsoid  → 生成一个椭球
elasticdeform  → 随机形变，让形状不规则
get_fixed_geo  → 把上面几步串起来，输出肿瘤 mask
```

**四档肿瘤尺寸**（这是两个扩散模型分工的依据）：

| tumor_type | 基础半径 | 半径随机范围 | 形变强度 | 数量 | 归哪个模型 |
|---|---|---|---|---|---|
| `tiny` | 4 | 0.75~1.25× | 0.5~1 | 1~3 个 | **早期** |
| `small` | 8 | 0.75~1.25× | 1~2 | 1~3 个 | **早期** |
| `medium` | 16 | 0.75~1.25× | 3~6 | 1 个 | **中晚期** |
| `large` | 32 | 0.75~2.0× | 5~10 | 1 个 | **中晚期** |

半径单位是体素，因为已重采样到 1mm 各向同性，所以也约等于 mm。

`random_select` 里有个**逐级放宽的退避策略**（第 37 行）：先在器官中部 30%~70% 范围找，
找不到就放宽到 20%~80%、10%~90%，最后全范围。因为肿瘤偏向长在器官中部，
但也要兜底防止死循环。

#### 阶段 B：纹理（用扩散模型填充 mask 区域的内容）

`synthesize_early_tumor`（第 359 行）的七步：

```
1. 生成几何 mask（tiny / small 各 50%）
2. 归一化到 [-1,1]，把肿瘤区域【抹掉】
3. VQGAN 编码到潜空间 + 按码本 min/max 归一化
4. 条件 = 拼接[潜表示(8ch), 下采样mask(1ch)] = 9 通道
5. 扩散采样（早期：完整 DDPM 4 步）
6. 反归一化 + VQGAN 解码回图像
7. ★ 高斯羽化混合回原图
```

**第 7 步是理解的重点**（第 359 行函数内）：

```python
sigma = np.random.uniform(0, 4)                        # 随机模糊强度
mask_blur = gaussian_filter(mask, sigma=[0,0,sigma,sigma,sigma])
final = (1 - mask_blur) * volume + mask_blur * sample  # 线性 alpha 混合
```

**不是简单替换**，而是对边界做高斯模糊得到渐变的 alpha，再线性混合。
这样肿瘤边界不会有生硬接缝。`sigma` 每次随机取 0~4，让过渡的软硬程度也随机。

**三个器官的处理不一致，要注意**：

| 器官 | 处理方式 |
|---|---|
| 肝 / 肾 | 高斯羽化局部混合 |
| **胰腺** | **整片替换**（`final = (sample+1)/2`），没有任何混合 |

> **自检**
> 1. 合成分哪两个阶段？ → 几何（形状位置）+ 纹理（扩散填充）
> 2. 为什么边界要用高斯羽化而不是直接替换？ → 直接替换会有生硬接缝，
>    羽化让过渡自然
> 3. `synthesize_early_tumor` 和 `synthesize_medium_tumor` 的主要区别？ →
>    用的采样器不同（DDPM vs DDIM），几何尺寸不同（tiny/small vs medium）
> 4. 为什么不需要显式的灰度/HU 匹配？ → 条件来自这张 CT 自己的潜表示，
>    扩散模型天然知道该用什么强度；再加高斯羽化过渡
> 5. 哪个器官的处理方式和其他两个不同？ → **胰腺**，整片替换不做混合

---

### 第 5 步 · 模型加载：`synt_model_prepare`（25 分钟）

**读**：第 279 行的 `synt_model_prepare`

```python
vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)          # STEP1 的产物
early_tester.load(f'{organ}_early.pt')                  # STEP2 的早期模型
noearly_diffusion.load_state_dict(ckpt['ema'])          # STEP2 的中晚期模型
noearly_sampler = DDIMSampler(noearly_diffusion, ...)
return vqgan, early_tester, noearly_sampler
```

**两套扩散模型的分工**：

| | 早期 | 中晚期 |
|---|---|---|
| 权重文件 | `{organ}_early.pt` | `{organ}_noearly.pt` |
| timesteps | 4 | 200 |
| 采样方式 | 完整 DDPM（`Tester`） | DDIM（`DDIMSampler`，50 步） |
| 用在哪 | tiny / small | medium / large |

**两个细节**：

1. **加载方式不一致**：早期用 `Tester.load()`（会同时载 `model` 和 `ema`），
   中晚期只取 `ckpt['ema']`。
2. **配置只 compose 了 `ddpm.yaml`**：同目录的 `vq_gan_3d.yaml` 从未被读取。
   而且 VQGAN 的结构是从 **checkpoint 里的 `hyper_parameters`** 重建的，
   不是从这个 yaml —— 所以即使 yaml 里的值对不上也不影响。

> **自检**
> 1. 为什么需要两套扩散模型？ → 早期肿瘤纹理简单（4 步够），
>    中晚期异质性高（需要 200 步训练 + DDIM 加速采样）
> 2. 采样时用的是 `model` 还是 `ema_model`？ → **`ema_model`**
>    （`synthesize_*` 里写的是 `tester.ema_model.sample(...)`）
> 3. VQGAN 的结构由什么决定？ → checkpoint 里的 `hyper_parameters['cfg']`，
>    不是 `diffusion_config/` 下的 yaml

---

### 第 6 步 · 评测：`validation.py`（25 分钟）

**读**：`validation.py` 第 88 行 `cal_dice` → 第 93 行 `cal_dice_nsd` → 第 233 行 `main()`

（注意：`validation.py` 第 72 行也有一个 `denoise_pred`，
与 `monai_trainer.py:18` 的同名函数是**两份不同的实现**，
都做"强制肿瘤落在器官内"这件事。）

**两个指标**：

| 指标 | 含义 | 对小肿瘤的意义 |
|---|---|---|
| **Dice** | 体积重叠度 | 小目标体素数少，Dice 波动大 |
| **NSD** | 表面距离在 2mm 内的比例 | 对边界更敏感，小目标更可靠 |

**★ 一个容易困惑的地方**（`validation.py` 里搜 `val_outputs[1, ...] =`）：

```python
val_outputs[1, ...] = val_organ_pseudo[1, ...]   # 器官通道被【伪标注覆盖】
```

**器官那一列的指标不是模型预测的器官**，而是用已知的伪标注替换掉的。
意图是把问题简化成"只评肿瘤"，避免器官预测的误差传导。

**真正要看的是 `tumor_dice` 和 `tumor_nsd`。**

**后处理**（`denoise_pred`，`monai_trainer.py:18`）：
强制 `肿瘤 = 器官 ∧ 肿瘤`，把跑到器官外面的假阳性去掉。

**输出**：
```
<save_dir>/<model_name>/<val_overlap>/pred/<name>.nii.gz   # 预测，0/1/2
<save_dir>/<model_name>/<val_overlap>/metrics.csv          # 指标
```

> **自检**
> 1. 器官那一列的指标是怎么来的？ → 用伪器官标注覆盖后算的，不是模型输出
> 2. 为什么要强制"肿瘤必须在器官内"？ → 去掉器官外的假阳性
> 3. NSD 的 tolerance 是多少？ → 2mm

---

## 3. 三条主线

### 主线一：合成的两阶段

```
几何阶段（不涉及神经网络）           纹理阶段（用扩散模型）
─────────────────────────           ─────────────────────
random_select  选位置               抹掉肿瘤区域
get_ellipsoid  造椭球        ──►    VQGAN 编码
elasticdeform  形变                  扩散采样（DDPM / DDIM）
                 ↓                   VQGAN 解码
            肿瘤 mask          ──►   高斯羽化混合回原图
```

### 主线二：四档尺寸 ↔ 两套模型

```
tiny(4) ──┐
          ├─► 早期模型（4 步 DDPM）  ← 80% 的合成走这里
small(8) ─┘

medium(16) ──┐
             ├─► 中晚期模型（200 步训练 + 50 步 DDIM）
large(32)  ──┘
```

### 主线三：训练集的两类样本

```
真实肿瘤病例 (MSD-Liver)  →  原样使用
                              ↓
                          拼成同一个 DataLoader
                              ↑
健康 CT (HealthyCT)       →  ~50% 注入合成肿瘤（80% 早期）
                           →  ~50% 保持原样（当负样本）
```

---

## 4. 最容易卡住的四个地方

**1. `main.py` 里变量名是反的**
`train_img_real` 存的是健康CT，`train_img_healthy` 存的是真实病例。
**以路径来源为准，不要信变量名。**

**2. 合成是"在线"的，不是数据集**
`train_epoch` 里原地改写 batch（`data[bs,...] = synt_data[0]`）。
没有 `synthetic_dataset/` 这样的目录。

**3. 显存里同时住着三个模型**
分割网络 + VQGAN + 两套扩散 U-Net，且扩散模型在整个 epoch 常驻。
这是 STEP3 显存开销的结构性原因。

**4. 器官指标被伪标注覆盖**
`val_outputs[1,...] = val_organ_pseudo[1,...]`。
看指标时别把 `organ_dice` 当成模型的器官分割能力。

---

## 5. 已确认的坑与 bug 清单

| 位置 | 问题 | 影响 |
|---|---|---|
| `main.py` 数据配对 | `train_*_real` / `train_*_healthy` **命名与内容相反** | 纯阅读陷阱，逻辑本身正确 |
| `main.py:322` | `torch.multiprocessing.set_start_method('fork')` | **Windows 上不可用**（fork 不存在）。只在 `--distributed` 时触发，单卡不受影响 |
| `main.py` | `--noamp` 是官方推荐参数（AMP 关掉） | A100 上无所谓，小显存卡上会显著增加显存压力 |
| `main.py` | 大量路径是字符串拼接，要求 root 以 `/` 结尾 | 少一个斜杠就找不到文件 |
| `validation.py` | `val_outputs[1,...]` 被伪标注覆盖 | 器官指标不是模型能力 |
| `validation.py` | 路径用 `split('/')` 而非 `os.path.basename` | Windows 上可能出问题 |
| `TumorGeneration/ldm/.../codebook.py` | **`_init_embeddings` 里有未注释的 `breakpoint()`** | eval 模式下不触发；一旦切训练模式会卡进 pdb |
| `TumorGeneration/diffusion_config/vq_gan_3d.yaml` | 从未被读取，且 downsample 值与 STEP1 不同 | 死配置，别被误导 |
| `networks2/` | 整个目录无人引用 | 别在这里找网络实现 |
| `postprocess.py` | 路径硬编码、不被引用 | 独立脚本 |

---

## 6. 速通版（25 分钟）

1. **`main.py` 的数据配对那段**（5 分钟）—— 两类样本怎么区分（注意变量名是反的）
2. **`monai_trainer.py:205` `train_epoch`**（8 分钟）—— ★ 在线合成的调用逻辑
3. **`TumorGeneration/utils.py:129` `get_fixed_geo`**（5 分钟）—— 四档尺寸
4. **`TumorGeneration/utils.py:359` `synthesize_early_tumor`**（7 分钟）—— ★ 七步流程和羽化混合

**25 分钟后你应该能回答**：
合成是在什么时机做的？哪两个阶段？80% 的肿瘤是哪一档、用哪个模型？

---

## 7. 术语对照表

| 代码里的名字 | 含义 |
|---|---|
| `organ_type` | 目标器官（liver / pancreas / kidney），决定几何和混合方式 |
| `organ_model` | 用哪个器官的扩散权重（默认与 organ_type 相同） |
| `tumor_type` | **文件名拼接变量**，不是"真实vs合成"的开关 |
| `sample_thresh` | 健康CT被注入合成肿瘤的概率（0.5） |
| `fg_thresh` | 裁剪时要求的器官体素下限（等效球半径，mm） |
| `ddim_ts` | 中晚期肿瘤的 DDIM 采样步数（50） |
| `synt_data` / `synt_target` | 合成后的图像 / 标签 |
| `organ_pseudo` | 伪器官标注，评测时用来覆盖器官通道 |
| `denoise_pred` | 后处理：强制肿瘤落在器官内（tumor = organ ∧ tumor） |
| `tiny/small/medium/large` | 四档肿瘤尺寸（半径 4/8/16/32） |

---

## 8. 与上下游的衔接

### 上游：STEP1 + STEP2 给了什么

| 产物 | 来源 | 在 STEP3 的用途 |
|---|---|---|
| `AutoencoderModel.ckpt` | STEP1 | VQGAN，做潜空间编解码 |
| `liver_early.pt` | STEP2 | 生成 tiny/small 肿瘤 |
| `liver_noearly.pt` | STEP2 | 生成 medium/large 肿瘤 |
| checkpoint 的 `ema` 项 | STEP2 | **实际采样用的是 EMA 参数** |
| `organ_pseudo_swin_new/` | **仓库自带** | 评测时的伪器官标注（118 份肝脏 mask） |

### 本阶段新增的

- 在线合成逻辑（`monai_trainer.py` + `TumorGeneration/`）
- 数据配对与自定义裁剪（`main.py`）
- Dice / NSD 评测（`validation.py`）

**下一步**：三阶段如何串通、各阶段之间的张量尺寸怎么传递，
见仓库根目录的 **`../HOW_IT_CONNECTS.md`**。

# networks/ 目录说明

**这个目录是分割网络的实现。** 与 `networks2/`（死代码）不同，本目录是**在用的**。

## 目录内容与用途

| 文件 | 行数 | 用途 | 谁在用 | 状态 |
|---|---|---|---|---|
| `swin_transformer_3d.py` | 637 | Swin Transformer 的 3D 主干 | `swin3d_unetr.py`、`swin3d_unetrv2.py` | **在路径上** |
| **`unetr_block.py`** | 261 | UNETR 的基础块 / 上采样块 | **`swin3d_unetrv2.py`**、`unetr.py` | **在路径上**（v2 用它，别以为只有 unetr 用） |
| `swin3d_unetrv2.py` | 258 | **SwinUNETR v2** | `main.py:30`、`validation.py:14` | **在路径上（实际被构造的那个）** |
| `unetr.py` | 226 | UNETR | `main.py:28` | import 了但**从未实例化** |
| `swin3d_unetr.py` | 189 | SwinUNETR v1 | `main.py:29` | import 了但**从未实例化** |
| `basicunetplusplus.py` | 161 | BasicUNet++ | 无 | **死代码** |
| `patchembedding.py` | 132 | 图像切 patch + 线性投影 | `vit.py` | 间接未用（见下） |
| `vit.py` | 96 | 纯 ViT | `unetr.py` | 间接未用（见下） |
| `transformerblock.py` | 76 | Transformer 块 | `vit.py` | 间接未用（见下） |
| `selfattention.py` | 68 | 自注意力 | `transformerblock.py` | 间接未用（见下） |
| `mlp.py` | 51 | Transformer 里的 MLP | `transformerblock.py` | 间接未用（见下） |

**「间接未用」是什么意思**：`vit.py` 只被 `unetr.py` 引用，
而 `unetr.py` 本身从未被实例化，所以 `vit.py` → `transformerblock.py`
→ `selfattention.py` / `mlp.py` → `patchembedding.py` 这条链整体不可达。

**但 `unetr_block.py` 不属于这条链** —— 它同时被 `swin3d_unetrv2.py:34`
（实际被构造的那个模型）引用，所以**在路径上**。

## 实际被使用的三种分割骨干

`main.py` 的 `main_worker`（第 319 行起）里用 `--model_name` 选择：

| `--model_name` | 构造的网络 | 代码位置 |
|---|---|---|
| `unet`（**默认**） | MONAI 原生 `UNet`，channels `(16,32,64,128,256)` | `main.py:309` |
| `swinunetr` | `SwinUNETR_v2`（本目录的 `swin3d_unetrv2.py`） | `main.py:328` |
| `nnunet` | MONAI `DynUNet`（不是本目录，见 `dynunet_pipeline/`） | `main.py:347` |

**注意**：`main.py` 第 28、29 行 import 了 `UNETR` 和 `SwinUNETR`（v1），
但代码里**没有任何分支构造它们** —— 是未接线的 import。

## 与本项目核心的关系

**关系不大。** DiffTumor 的核心思路是"合成肿瘤做数据增强"，
与用哪个分割骨干是正交的。

作者用三种骨干（U-Net / nnU-Net / Swin-UNETR）各跑一遍，
是为了证明"合成数据带来的提升不依赖于特定网络结构"——这是消融实验的一部分。

所以读代码时：**这些文件只需要知道"是分割骨干"即可，不必逐层读**。
真正值得花时间的是 `main.py` 里模型选择的逻辑，和
`monai_trainer.py` 里怎么把合成肿瘤喂进去。

## 与 networks2/ 的关系

`networks2/` 是本目录的一份**近重复拷贝，且完全无人引用**。
详见 `../networks2/README.md`。

如果你在找某个网络实现，**只看本目录**，不要去 `networks2/`。

# 10k FM vs DDPM 指标对比

数据：19 例 fold-0 验证病例，配对 inpainting。
模型：FM 与 DDPM 均训练到 10k steps；DDPM 固定 4 步采样，FM 分别用 4/8/16 步。

## 1. 总体指标

| 指标 | FM 4步 | FM 8步 | FM 16步 | DDPM 10k | real | p (FM8 vs DDPM) |
|---|---:|---:|---:|---:|---:|---:|
| Wasserstein ↓ | 16.336±11.052 | 14.874±7.625 | 15.727±8.643 | 34.635±17.057 | — | 0.0012 |
| KS ↓ | 0.272±0.167 | 0.293±0.163 | 0.330±0.191 | 0.534±0.228 | — | 0.0039 |
| JS ↓ | 0.091±0.075 | 0.087±0.074 | 0.105±0.098 | 0.247±0.162 | — | 0.0024 |
| d_mean (HU) | -9.513±16.123 | -1.358±16.311 | -0.836±17.663 | 27.982±26.479 | — | 0.0004 |
| GLCM contrast | 5.229±2.406 | 3.975±1.929 | 3.624±2.003 | 4.716±2.470 | 3.426 | 0.0053 |
| GLCM homogeneity | 0.500±0.059 | 0.535±0.069 | 0.548±0.075 | 0.519±0.066 | 0.572 | 0.0323 |
| GLCM energy | 0.145±0.034 | 0.184±0.051 | 0.203±0.060 | 0.173±0.044 | 0.206 | 0.0728 |
| GLCM correlation | 0.548±0.082 | 0.421±0.062 | 0.371±0.062 | 0.402±0.100 | 0.508 | 0.3955 |
| GLCM dissimilarity | 1.697±0.423 | 1.464±0.404 | 1.385±0.425 | 1.581±0.447 | 1.300 | 0.0108 |
| outside SSIM ↑ | 0.458±0.096 | 0.600±0.117 | 0.655±0.123 | 0.555±0.144 | — | 0.0446 |
| outside PSNR ↑ | 21.220±2.418 | 23.879±2.835 | 24.736±3.031 | 19.929±3.380 | — | 0.0024 |

> 直方图/SSIM/PSNR 有明确方向；GLCM 没有单一“越大越好”，应与 real 列比较。

## 2. 逐病例胜率（FM 8步 vs DDPM）

| 指标 | FM 更好 | DDPM 更好 |
|---|---:|---:|
| Wasserstein ↓ | 16/19 | 3/19 |
| outside SSIM ↑ | 15/19 | 4/19 |
| outside PSNR ↑ | 17/19 | 2/19 |
| GLCM contrast（距 real） | 11/19 | 8/19 |
| GLCM homogeneity（距 real） | 13/19 | 6/19 |
| GLCM energy（距 real） | 13/19 | 6/19 |
| GLCM correlation（距 real） | 15/19 | 4/19 |
| GLCM dissimilarity（距 real） | 12/19 | 7/19 |

## 3. Diversity

| 方法 | mean pairwise distance |
|---|---:|
| real | 4.923 |
| DDPM 10k | 4.296 |
| FM 4步 | 3.392 |
| FM 8步 | 3.273 |
| FM 16步 | 3.526 |

## 4. 结论

- **FM 8 步是当前最佳配置**：直方图（W/KS/JS/d_mean）、outside SSIM/PSNR、以及多数 GLCM 指标都优于 DDPM 10k。
- **FM 4 步是欠积分**：SSIM 0.458 → 8 步 0.600，d_mean −9.5 → −1.4 HU；4 步只能作为“等 NFE”对照。
- **DDPM 10k 已不再崩溃**，但肿瘤强度整体偏亮（d_mean ≈ +28 HU），且多数病例的 Wasserstein 几乎等于均值偏移。
- **FM 的主要弱点是多样性**：real 4.92，DDPM 4.30，FM 仅 3.27–3.53；FM 在含空气/黑色背景的 crop 上仍有颗粒伪影。
- **建议**：下游 STEP3 用 FM 8 步；论文同时报告 FM 4 步 vs DDPM 4 步（等 NFE）和 FM 8/16 步（最佳质量）。

## 5. 文件

- 原始指标：`outputs_10k/texture_metrics/`、`outputs_10k_fm8/texture_metrics/`、`outputs_10k_fm16/texture_metrics/`
- 逐病例：各目录下 `texture_metrics/per_case.csv`
- 汇总 CSV：`comparison_10k.csv`
"""
在已保存的测量数组上补充计算纹理 / 解剖 / 多样性指标（不需要重跑模型）。

【输入】
    outputs_*/cases/*/arrays.npz
        real_hu      真实 96³ crop（HU）
        fm_hu        FM 生成结果（HU）
        ddpm_hu      DDPM 生成结果（HU）
        tumor        bool 肿瘤 mask
    outputs_*/metrics_per_case.csv
        已有的直方图指标，按 (case, method) 合并进来

【计算内容】
    1. 3D 带 mask 的 GLCM（灰度共生矩阵）
       - contrast      对比度，纹理越粗糙越大
       - homogeneity   同质性，局部越均匀越大
       - energy        能量，纹理越规则越大
       - correlation   相关性，方向性纹理的度量
       - dissimilarity 相异性
       GLCM 只在肿瘤体素上统计，量化 32 级、HU 范围 [-150,250]，
       6 个方向（3 个轴 + 3 个面对角），距离 1，并做对称化。

    2. 肿瘤外 SSIM / PSNR
       先把两份图像的肿瘤区域都替换成真实值，再比较——
       这样指标只反映「周围解剖结构是否被破坏」，不被肿瘤区差异污染。

    3. 多样性
       每个病例提取一组特征（一阶统计量 + GLCM），用真实样本的
       均值/标准差标准化后，计算病例两两之间的平均欧氏距离。
       距离越小说明模型生成的结果越同质化（mode collapse 倾向）。

    4. FM vs DDPM 的配对 Wilcoxon 检验
       同一个病例、同一个 crop、同一个 mask，只是生成器不同，
       因此可以用配对检验判断差异是否显著。

【解读注意】
    GLCM/SSIM/PSNR 没有统一的「越大越好」：例如 contrast 更接近真实值
    才好，而不是越大越好。应该把 FM/DDPM 与 real 列对比，
    而不是互相比较绝对值。

用法：
    python -u flowmatching/texture_study/texture_metrics.py \
        --arrays-dir flowmatching/texture_study/outputs_2k
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import skew, kurtosis, wilcoxon
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

A_MIN, A_MAX = -175.0, 250.0
GLCM_LO, GLCM_HI = -150.0, 250.0
GLCM_BINS = 32
OFFSETS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
           (1, 1, 0), (1, 0, 1), (0, 1, 1)]


def glcm_features(volume, mask):
    """3-D masked, symmetric GLCM features (distance 1, 6 directions).

    实现细节
    ----------
    - 先把 HU 线性量化到 32 个 bin：范围 [-150,250] 覆盖肝脏与肿瘤的
      主要密度区间，超出部分截断；
    - 只统计「相邻两个体素都在 mask 内」的体素对，
      避免把肿瘤边界外的正常组织算进纹理；
    - 每对 (a,b) 同时累加 (a,b) 和 (b,a)，得到对称 GLCM；
    - 最后归一化成概率分布 P，再算 5 个常用 Haralick 特征。
    """
    if mask.sum() < 2:
        return {k: np.nan for k in
                ("glcm_contrast", "glcm_homogeneity", "glcm_energy",
                 "glcm_correlation", "glcm_dissimilarity")}

    q = np.clip((volume - GLCM_LO) / (GLCM_HI - GLCM_LO) * (GLCM_BINS - 1),
                0, GLCM_BINS - 1).astype(np.int16)
    P = np.zeros((GLCM_BINS, GLCM_BINS), dtype=np.float64)

    for dx, dy, dz in OFFSETS:
        a = q[:q.shape[0] - dx, :q.shape[1] - dy, :q.shape[2] - dz]
        b = q[dx:, dy:, dz:]
        ma = mask[:mask.shape[0] - dx, :mask.shape[1] - dy, :mask.shape[2] - dz]
        mb = mask[dx:, dy:, dz:]
        m = ma & mb
        if not m.any():
            continue
        np.add.at(P, (a[m], b[m]), 1.0)
        np.add.at(P, (b[m], a[m]), 1.0)

    s = P.sum()
    if s == 0:
        return {k: np.nan for k in
                ("glcm_contrast", "glcm_homogeneity", "glcm_energy",
                 "glcm_correlation", "glcm_dissimilarity")}
    P /= s

    idx = np.arange(GLCM_BINS, dtype=np.float64)
    diff = idx[:, None] - idx[None, :]
    contrast = float((diff ** 2 * P).sum())
    dissimilarity = float((np.abs(diff) * P).sum())
    homogeneity = float((P / (1.0 + np.abs(diff))).sum())
    asm = float((P ** 2).sum())
    energy = float(np.sqrt(asm))

    pi = P.sum(axis=1)
    pj = P.sum(axis=0)
    mi = float((idx * pi).sum())
    mj = float((idx * pj).sum())
    si = float(np.sqrt(((idx - mi) ** 2 * pi).sum()))
    sj = float(np.sqrt(((idx - mj) ** 2 * pj).sum()))
    corr = (float(((idx[:, None] - mi) * (idx[None, :] - mj) * P).sum())
            / (si * sj)) if si > 0 and sj > 0 else np.nan

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": corr,
        "glcm_dissimilarity": dissimilarity,
    }


def outside_metrics(real, gen, tumor):
    """SSIM / PSNR outside the tumor (tumor replaced by real in both).

    为什么要替换肿瘤区：
        SSIM 在 96³ 上做全局计算，肿瘤区域的巨大差异会淹没
        周围组织的细微差异。把两边肿瘤都填成真实值后，
        指标只衡量「生成过程有没有破坏肿瘤外的解剖结构」。
    """
    r = real.copy()
    g = gen.copy()
    r[tumor] = real[tumor]
    g[tumor] = real[tumor]
    data_range = A_MAX - A_MIN
    try:
        ssim = float(structural_similarity(r, g, data_range=data_range))
    except Exception:
        ssim = np.nan
    try:
        psnr = float(peak_signal_noise_ratio(r, g, data_range=data_range))
    except Exception:
        psnr = np.nan
    return {"ssim_out": ssim, "psnr_out": psnr}


def first_order(v):
    v = np.asarray(v, dtype=np.float64)
    p, _ = np.histogram(v, bins=64, range=(A_MIN, A_MAX))
    p = p / max(p.sum(), 1)
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return [float(np.mean(v)), float(np.std(v)),
            float(skew(v, bias=False)) if len(v) > 2 else np.nan,
            float(kurtosis(v, fisher=True, bias=False)) if len(v) > 3 else np.nan,
            float(ent),
            float(np.percentile(v, 10)), float(np.percentile(v, 50)),
            float(np.percentile(v, 90))]


def pairwise_mean(feats):
    """Mean pairwise Euclidean distance in standardized feature space.

    输入是已经用真实样本统计量标准化过的特征矩阵。
    距离越大 = 病例之间差异越大 = 多样性越好；
    如果某个模型的生成结果都差不多，这个值会明显低于 real。
    """
    X = np.nan_to_num(np.asarray(feats, dtype=np.float64), nan=0.0)
    if len(X) < 2:
        return np.nan
    d = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d.append(np.linalg.norm(X[i] - X[j]))
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-dir", default=str(
        Path(__file__).resolve().parent / "outputs_2k"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.arrays_dir)
    out = Path(args.out) if args.out else src / "texture_metrics"
    out.mkdir(parents=True, exist_ok=True)

    # existing histogram metrics, keyed by (case, method)
    hist = {}
    hp = src / "metrics_per_case.csv"
    if hp.exists():
        for r in csv.DictReader(hp.open(encoding="utf-8-sig")):
            hist[(r["case"], r["method"])] = r

    rows = []
    feats = {"real": [], "fm": [], "ddpm": []}
    cases = []
    for cp in sorted((src / "cases").glob("*/arrays.npz")):
        z = np.load(cp)
        case = cp.parent.name
        real = z["real_hu"]
        tumor = z["tumor"].astype(bool)
        if tumor.sum() < 2:
            continue
        cases.append(case)

        # 真实样本的特征向量：一阶统计量(8) + GLCM(5) = 13 维，
        # 后面用于多样性的标准化基准。
        real_glcm = glcm_features(real, tumor)
        feats["real"].append(first_order(real[tumor]) +
                             list(real_glcm.values()))

        for method, key in (("fm", "fm_hu"), ("ddpm", "ddpm_hu")):
            gen = z[key]
            glcm = glcm_features(gen, tumor)
            anat = outside_metrics(real, gen, tumor)
            r = {
                "case": case,
                "n_tumor": int(tumor.sum()),
                "method": method,
                **glcm,
                **anat,
            }
            h = hist.get((case, method), {})
            for k in ("wasserstein", "ks", "js", "d_mean", "d_std", "d_entropy"):
                r[k] = float(h[k]) if k in h and h[k] != "" else np.nan
            rows.append(r)
            feats[method].append(first_order(gen[tumor]) +
                                 list(glcm.values()))

    if not rows:
        print("no rows")
        return

    fields = list(rows[0].keys())
    with (out / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # compute real GLCM metrics per case for the summary
    real_rows = []
    for cp in sorted((src / "cases").glob("*/arrays.npz")):
        z = np.load(cp)
        tumor = z["tumor"].astype(bool)
        if tumor.sum() < 2:
            continue
        g = glcm_features(z["real_hu"], tumor)
        real_rows.append({"case": cp.parent.name, "n_tumor": int(tumor.sum()),
                          "method": "real", **g,
                          "ssim_out": np.nan, "psnr_out": np.nan,
                          "wasserstein": np.nan, "ks": np.nan, "js": np.nan,
                          "d_mean": np.nan, "d_std": np.nan, "d_entropy": np.nan})

    metric_keys = ["wasserstein", "ks", "js", "d_mean",
                   "glcm_contrast", "glcm_homogeneity", "glcm_energy",
                   "glcm_correlation", "glcm_dissimilarity",
                   "ssim_out", "psnr_out"]

    summary = []
    for m in ("fm", "ddpm"):
        rs = [r for r in rows if r["method"] == m]
        row = {"method": m, "n": len(rs)}
        for k in metric_keys:
            v = np.array([r[k] for r in rs if np.isfinite(r[k])])
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_std"] = float(v.std()) if len(v) else np.nan
        summary.append(row)

    # 真实数据的 GLCM 作为参照列（real 没有「生成 vs 真实」的直方图指标，
    # 对应字段填 NaN）
    if real_rows:
        row = {"method": "real", "n": len(real_rows)}
        for k in metric_keys:
            v = np.array([r[k] for r in real_rows if np.isfinite(r[k])])
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_std"] = float(v.std()) if len(v) else np.nan
        summary.append(row)

    sfields = sorted({k for r in summary for k in r})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    # FM vs DDPM 的配对 Wilcoxon 检验：
    # 同一病例的两种生成结果天然配对，适合用配对非参数检验；
    # p < 0.05 表示该指标上两者的差异不太可能来自偶然。
    fm = {r["case"]: r for r in rows if r["method"] == "fm"}
    dd = {r["case"]: r for r in rows if r["method"] == "ddpm"}
    common = sorted(set(fm) & set(dd))
    pvals = {}
    for k in metric_keys:
        a = np.array([fm[c][k] for c in common])
        b = np.array([dd[c][k] for c in common])
        ok = np.isfinite(a) & np.isfinite(b)
        pvals[k] = float(wilcoxon(a[ok], b[ok]).pvalue) if ok.sum() >= 3 else np.nan

    # diversity
    # 用真实样本的均值/标准差做标准化，避免「HU 均值」这类大尺度特征
    # 主导距离计算；sd 为 0 的维度设为 1，避免除零。
    R = np.asarray(feats["real"], dtype=np.float64)
    mu = np.nanmean(R, axis=0)
    sd = np.nanstd(R, axis=0)
    sd[sd < 1e-8] = 1.0
    diversity = {}
    for m in ("real", "fm", "ddpm"):
        X = (np.asarray(feats[m], dtype=np.float64) - mu) / sd
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        diversity[m] = pairwise_mean(X)

    with (out / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Texture / anatomy / diversity metrics\n\n")
        f.write(f"cases: {len(cases)}, source: {src}\n\n")
        f.write("## FM vs DDPM\n\n")
        f.write("| metric | FM | DDPM | real | Wilcoxon p |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        s = {r["method"]: r for r in summary}
        for k in metric_keys:
            f.write(f"| {k} | "
                    f"{s.get('fm',{}).get(f'{k}_mean',np.nan):.3f} | "
                    f"{s.get('ddpm',{}).get(f'{k}_mean',np.nan):.3f} | "
                    f"{s.get('real',{}).get(f'{k}_mean',np.nan):.3f} | "
                    f"{pvals.get(k,np.nan):.4f} |\n")
        f.write("\n## Diversity (mean pairwise distance, lower = less diverse)\n\n")
        f.write("| method | mean pairwise distance |\n|---|---:|\n")
        for m in ("real", "fm", "ddpm"):
            f.write(f"| {m} | {diversity[m]:.3f} |\n")
        f.write("\n> GLCM / SSIM / PSNR are descriptive: compare FM and DDPM to "
                "the `real` column; there is no single 'higher is better' for "
                "all GLCM features.\n")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for group, name in [
        (["glcm_contrast", "glcm_homogeneity", "glcm_energy",
          "glcm_correlation"], "glcm_boxplots.png"),
        (["ssim_out", "psnr_out"], "anatomy_boxplots.png"),
    ]:
        fig, axes = plt.subplots(1, len(group), figsize=(4 * len(group), 4))
        for ax, k in zip(np.atleast_1d(axes), group):
            data = [[r[k] for r in rows if r["method"] == m and np.isfinite(r[k])]
                    for m in ("fm", "ddpm")]
            realv = [r[k] for r in real_rows if np.isfinite(r[k])]
            ax.boxplot(data + [realv], labels=["FM", "DDPM", "real"])
            ax.set_title(k)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["real", "fm", "ddpm"],
           [diversity[m] for m in ("real", "fm", "ddpm")],
           color=["gray", "tab:orange", "tab:green"])
    ax.set_ylabel("mean pairwise distance")
    ax.set_title("diversity")
    fig.tight_layout()
    fig.savefig(out / "diversity.png", dpi=150)
    plt.close(fig)

    print(f"[done] {out}")
    for k in metric_keys:
        print(f"{k:20s} FM={s['fm'][f'{k}_mean']:.3f} "
              f"DDPM={s['ddpm'][f'{k}_mean']:.3f} "
              f"real={s.get('real',{}).get(f'{k}_mean',np.nan):.3f} "
              f"p={pvals[k]:.4f}")
    print("diversity", {m: round(diversity[m], 3) for m in diversity})


if __name__ == "__main__":
    main()
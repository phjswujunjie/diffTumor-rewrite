"""
Flow Matching 与 DDPM 的配对 inpainting 纹理测量。

【核心思路：同输入、同 mask，只换生成器】
    对每个真实 early-tumor 病例，先把真实肿瘤擦掉得到「健康肝脏」，
    再分别用 FM 和 DDPM 填补同一个空洞。因为条件完全相同，
    生成结果在肿瘤区域的差异只来自生成模型本身。

【单个病例的完整流程】
    1. 预处理与 STEP2 一致：RAS + 1mm + HU[-175,250] → [-1,1]
    2. 以肿瘤质心为中心、加随机抖动裁 96³
    3. 抹掉肿瘤，构造 9 通道条件：
         8 通道 = VQGAN 编码的「抹洞后潜表示」（码本范围归一化到 [-1,1]）
         1 通道 = 降采样到 24³ 的 mask（-1/1）
    4. FM 用 Euler 积分（默认 4 步，可 --fm-steps 改 8/16）
       DDPM 用作者 liver_early.pt 的 4 步采样
    5. 只在肿瘤体素上比较生成值与真实值：
         - 直方图距离：Wasserstein / KS / JS
         - 一阶统计量：均值、标准差、偏度、峰度、熵、分位数
    6. 两个参照：
         - fm_match / ddpm_match：把生成值做分位数映射到真实分布，
           这是「后处理直方图匹配」的上界，用来区分「分布形状」与
           「整体偏移」两类误差；
         - real_ref：把真实肿瘤体素随机分两半互相比，
           这是同一分布下的噪声底，任何模型都不可能低于它。
    7. 额外算肿瘤外区域的相关性，检查生成是否破坏了周围解剖结构。

【输出】
    flowmatching/texture_study/outputs/
        metrics_per_case.csv  每个病例 × 每种方法的全部指标
        summary.csv / summary.md  按方法汇总的 mean±std
        histogram_overlay.png 四种方法的平均 HU 直方图
        boxplots.png          Wasserstein/KS/JS 的分布箱线图
        cases/<case>/         真实/生成 NIfTI、mask、预览图
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis

ROOT = Path(__file__).resolve().parents[2]  # STEP2.DiffusionModel/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Orientationd, Spacingd,
    ScaleIntensityRanged,
)
from flowmatching import Unet3D, FlowMatching
from ddpm.diffusion import GaussianDiffusion, Tester
from seed_utils import set_seed

A_MIN, A_MAX = -175.0, 250.0
ROI = 96


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def build_preprocess():
    """与 STEP2 训练完全一致的确定性预处理。

    注意 b_min/b_max 是 [-1,1] 而不是 [0,1]：STEP2 的 DDPM/FM 训练
    就是在 [-1,1] 空间里做的，这里必须对齐，否则 VQGAN 编码会偏。
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=A_MIN, a_max=A_MAX,
                             b_min=-1.0, b_max=1.0, clip=True),
    ])


def crop_roi(image, label, seed):
    """以肿瘤质心为中心裁 96³，并加 ±12 体素的随机抖动。

    抖动的作用：
        - 避免所有样本都严格居中，减少「中心先验」；
        - FM 和 DDPM 用同一个 seed 时得到完全相同的 crop，
          保证配对比较的输入一致。

    返回 (image_crop, label_crop, start)，start 是裁剪起点，
    用于把 crop 的仿射矩阵还原到原始图像坐标系。
    """
    tumor = (label == 2)
    if tumor.sum() == 0:
        return None

    # 体积小于 96³ 时先补零，保证能裁出固定大小
    spatial = np.array(image.shape)
    if np.any(spatial < ROI):
        pad = [(0, max(0, ROI - int(s))) for s in spatial]
        image = np.pad(image, pad, mode="constant")
        label = np.pad(label, pad, mode="constant")
        spatial = np.array(image.shape)
        tumor = (label == 2)

    # 肿瘤质心（所有肿瘤体素坐标的均值）
    center = np.argwhere(tumor).mean(axis=0).astype(int)
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-12, 13, size=3)
    # clip 保证裁剪窗口不越界
    start = np.clip(center - ROI // 2 + jitter, 0, spatial - ROI).astype(int)
    sl = tuple(slice(int(start[d]), int(start[d]) + ROI) for d in range(3))
    return image[sl].copy(), label[sl].copy(), start


def to_hu(x):
    """归一化 [-1,1] -> HU，并截断到 [-175,250] 与预处理一致。"""
    return np.clip(x * (A_MAX - A_MIN) / 2.0 + (A_MAX + A_MIN) / 2.0,
                   A_MIN, A_MAX)


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def load_fm(vq_ckpt, fm_ckpt, device):
    """加载 FM 权重。

    只加载 ckpt 里的 "ema"：与训练代码一致，推理用 EMA 权重，
    它比原始权重更平滑、采样更稳定。
    """
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    fm = FlowMatching(net, image_size=24, num_frames=24, channels=17,
                      timesteps=4, sampling_steps=4,
                      vqgan_ckpt=vq_ckpt).to(device).eval()
    ck = torch.load(fm_ckpt, map_location="cpu", weights_only=False)
    fm.load_state_dict(ck["ema"])
    print(f"[FM] loaded {fm_ckpt} (step={ck.get('step')})")
    return fm


def load_ddpm(vq_ckpt, ddpm_ckpt, device):
    """加载 DDPM 权重。

    Tester 是作者代码里的封装：内部 deepcopy 出 ema_model，
    load() 同时恢复 model 和 ema_model；这里只使用 ema_model 采样。
    """
    net = Unet3D(dim=24, dim_mults=(1, 2, 4, 8), channels=17, out_dim=8)
    diff = GaussianDiffusion(
        net, vqgan_ckpt=vq_ckpt, image_size=24, num_frames=24,
        channels=17, timesteps=4, loss_type="l1").to(device)
    tester = Tester(diff)
    tester.load(ddpm_ckpt, map_location=device)
    tester.ema_model.eval()
    print(f"[DDPM] loaded {ddpm_ckpt} (step={tester.step})")
    return tester


@torch.no_grad()
def build_cond(vqgan, image, tumor, device):
    """构造与训练时完全一致的 9 通道条件。

    返回 cond = [masked_latent(8ch), downsampled_mask(1ch)]。
    注意 VQGAN 使用 permute(0,1,-1,-3,-2) 的轴顺序。
    """
    # image 是 [-1,1] 的 96³，tumor 是 bool 96³；
    # 加两个维度变成 (1,1,X,Y,Z)，与训练时 batch=1 的输入一致。
    img = torch.from_numpy(image[None, None]).to(device).float()
    msk = torch.from_numpy(tumor[None, None].astype(np.float32)).to(device)
    # 抹洞：肿瘤位置置 0（对应 [-1,1] 里的 0，接近肝脏平均密度）
    masked = img * (1 - msk)

    # VQGAN 期望 (B,C,D,H,W)，所以把 (X,Y,Z) 换成 (Z,X,Y)
    ip = img.permute(0, 1, -1, -3, -2)
    mp = masked.permute(0, 1, -1, -3, -2)
    tp = msk.permute(0, 1, -1, -3, -2)

    emin = vqgan.codebook.embeddings.min()
    emax = vqgan.codebook.embeddings.max()
    feat = vqgan.encode(ip, quantize=False, include_embeddings=True)
    feat = ((feat - emin) / (emax - emin)) * 2.0 - 1.0
    mfeat = vqgan.encode(mp, quantize=False, include_embeddings=True)
    mfeat = ((mfeat - emin) / (emax - emin)) * 2.0 - 1.0
    cc = torch.nn.functional.interpolate(tp * 2.0 - 1.0, size=mfeat.shape[-3:])
    return torch.cat((mfeat, cc), dim=1)


@torch.no_grad()
def sample_fm(fm, cond, steps=None):
    out = fm.sample(cond=cond, batch_size=1, steps=steps)
    # sample() returns the decoded image in the permuted axis order used by
    # the VQGAN; invert it before comparing with the original crop.
    out = out.permute(0, 1, -2, -1, -3)
    return out[0, 0].detach().float().cpu().numpy()


@torch.no_grad()
def sample_ddpm(tester, cond):
    out = tester.ema_model.sample(cond=cond, batch_size=1)
    out = out.permute(0, 1, -2, -1, -3)
    return out[0, 0].detach().float().cpu().numpy()


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def quantile_match(src, ref):
    """把 src 的数值按分位数映射到 ref 的经验分布。

    这是「后处理直方图匹配」的 oracle 版本：
        - 它使用了真实肿瘤的分布信息，现实中拿不到；
        - 作用是给出一个上界——如果模型 + 直方图匹配后仍然差，
          说明问题不在整体亮度偏移，而在空间纹理/结构。
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    if len(ref) == 0 or len(src) == 0:
        return src
    ranks = np.searchsorted(np.sort(src), src, side="right") - 1
    ranks = np.clip(ranks, 0, len(src) - 1)
    pos = ranks / max(len(src) - 1, 1) * (len(ref) - 1)
    return np.interp(pos, np.arange(len(ref)), ref)


def js_divergence(a, b, bins=64, value_range=(A_MIN, A_MAX)):
    """Jensen-Shannon divergence（自然对数，越小越接近）。

    与 KS 的区别：
        KS 只看两条累积分布的最大垂直距离，对局部形状不敏感；
        JS 对整条分布做对称 KL，能反映「分布整体形状」的差异。
        两者一起看，避免单一指标误判。
    """
    pa, _ = np.histogram(a, bins=bins, range=value_range)
    pb, _ = np.histogram(b, bins=bins, range=value_range)
    pa = pa / max(pa.sum(), 1) + 1e-12
    pb = pb / max(pb.sum(), 1) + 1e-12
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) +
                 0.5 * np.sum(pb * np.log(pb / m)))


def first_order(v):
    """一阶/直方图特征：均值、标准差、偏度、峰度、熵、分位数。"""
    v = np.asarray(v, dtype=np.float64)
    bins = np.linspace(A_MIN, A_MAX, 65)
    p, _ = np.histogram(v, bins=bins)
    p = p / max(p.sum(), 1)
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "skew": float(skew(v, bias=False)) if len(v) > 2 else np.nan,
        "kurt": float(kurtosis(v, fisher=True, bias=False)) if len(v) > 3 else np.nan,
        "entropy": float(ent),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
    }


def compare(real, gen):
    """比较真实肿瘤体素和生成肿瘤体素的分布。

    只比较「肿瘤区域内」的体素（调用方已用 tumor mask 取过），
    因为肿瘤外的解剖结构由条件 latent 直接决定，不能反映生成模型的纹理能力。

    返回的 d_* 字段是「生成 - 真实」的差值：
        d_mean > 0  生成偏亮
        d_std  < 0  生成过于均匀（纹理缺失）
        d_entropy < 0 生成灰度层次单一
    """
    s_real = first_order(real)
    s_gen = first_order(gen)
    out = {
        "wasserstein": float(wasserstein_distance(real, gen)),
        "ks": float(ks_2samp(real, gen).statistic),
        "ks_p": float(ks_2samp(real, gen).pvalue),
        "js": js_divergence(real, gen),
    }
    for k in s_real:
        out[f"{k}_real"] = s_real[k]
        out[f"{k}_gen"] = s_gen[k]
        out[f"d_{k}"] = s_gen[k] - s_real[k]
    return out


def split_half_reference(real, rng, repeats=10):
    """噪声底：把真实肿瘤体素随机分成两半，比较两半之间的分布距离。

    这是「同一个真实分布内部」的天然差异，因此是任何生成模型
    理论上能达到的下限（再低就说明它比真实数据更像真实数据，不合理）。
    重复 10 次取平均以降低抽样波动。
    """
    vals = []
    n = len(real)
    if n < 4:
        return None
    half = n // 2
    for _ in range(repeats):
        idx = rng.permutation(n)
        vals.append(compare(real[idx[:half]], real[idx[half:2 * half]]))
    keys = vals[0].keys()
    return {k: float(np.mean([v[k] for v in vals])) for k in keys}


# ----------------------------------------------------------------------
# plotting / saving
# ----------------------------------------------------------------------
def save_case_outputs(case_out, real_hu, fm_hu, ddpm_hu, fm_match_hu,
                      tumor, start, affine):
    """保存每个病例的 NIfTI/npz、预览图和 crop affine。

    裁剪后的数组只是原始体积的一块，直接保存会丢失空间位置。
    这里把「裁剪起点」写进仿射矩阵的平移项，得到 crop_affine，
    这样生成的 NIfTI 在 3D Slicer 里能对回原始 CT 的位置。
    """
    case_out.mkdir(parents=True, exist_ok=True)
    translation = np.eye(4)
    translation[:3, 3] = start
    crop_affine = affine @ translation

    def save_nii(arr, name, dtype=np.float32):
        nib.save(nib.Nifti1Image(arr.astype(dtype), crop_affine),
                 str(case_out / name))

    save_nii(real_hu, "real_hu.nii.gz")
    save_nii(fm_hu, "fm_hu.nii.gz")
    save_nii(ddpm_hu, "ddpm_hu.nii.gz")
    save_nii(fm_match_hu, "fm_match_hu.nii.gz")
    save_nii(tumor.astype(np.uint8), "tumor_mask.nii.gz", np.uint8)
    np.savez_compressed(case_out / "arrays.npz",
                        real_hu=real_hu, fm_hu=fm_hu, ddpm_hu=ddpm_hu,
                        fm_match_hu=fm_match_hu, tumor=tumor, start=start)

    coords = np.argwhere(tumor)
    z = int(np.median(coords[:, 2])) if len(coords) else ROI // 2
    panels = [
        (real_hu[:, :, z].T, "Real"),
        (fm_hu[:, :, z].T, "FM"),
        (ddpm_hu[:, :, z].T, "DDPM"),
        (fm_match_hu[:, :, z].T, "FM + hist match"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", origin="lower", vmin=A_MIN, vmax=A_MAX)
        ax.contour(tumor[:, :, z].T.astype(np.uint8), levels=[0.5],
                   colors=["red"], linewidths=1.0)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(case_out / "preview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:/thesisAndCode/data/Task03_Liver")
    ap.add_argument("--label-root", default=r"E:/thesisAndCode/data/preprocessed_labels")
    ap.add_argument("--manifest", default=str(
        ROOT / "cross_eval/liver_tumor_data_early_fold/real_tumor_val_0.txt"))
    ap.add_argument("--vqgan-ckpt", default=r"E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/TumorGeneration/model_weight/AutoencoderModel.ckpt")
    ap.add_argument("--ddpm-ckpt", default=r"E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/TumorGeneration/model_weight/liver_early.pt")
    ap.add_argument("--fm-ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "flowmatching/texture_study/outputs"))
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fm-steps", type=int, default=4,
                    help="FM Euler steps at inference (default 4)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fm = load_fm(args.vqgan_ckpt, args.fm_ckpt, device)
    tester = load_ddpm(args.vqgan_ckpt, args.ddpm_ckpt, device)
    preprocess = build_preprocess()

    rows = [l.split() for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    rows = rows[:args.max_cases]
    print(f"[cases] {len(rows)}")

    all_rows = []
    case_meta = []
    for i, (img_rel, lbl_rel) in enumerate(rows):
        case = Path(lbl_rel).stem.replace(".nii", "")
        try:
            # 1) 确定性预处理：RAS、1mm、HU[-175,250] -> [-1,1]
            data = preprocess({
                "image": str(Path(args.data_root) / img_rel),
                "label": str(Path(args.label_root) / lbl_rel),
            })
            image = np.asarray(data["image"])[0].astype(np.float32)
            label = np.asarray(data["label"])[0]
            tumor_all = (label == 2)
            if tumor_all.sum() == 0:
                print(f"[{i+1}/{len(rows)}] {case}: no tumor, skip")
                continue

            cropped = crop_roi(image, label, args.seed + i)
            if cropped is None:
                print(f"[{i+1}/{len(rows)}] {case}: crop failed, skip")
                continue
            img_c, lbl_c, start = cropped
            tumor = (lbl_c == 2)
            n_tumor = int(tumor.sum())
            if n_tumor < 20:
                print(f"[{i+1}/{len(rows)}] {case}: tumor too small ({n_tumor}), skip")
                continue

            meta = data.get("image_meta_dict", {}) or {}
            affine = meta.get("affine", np.eye(4))
            if torch.is_tensor(affine):
                affine = affine.detach().cpu().numpy()
            affine = np.asarray(affine, dtype=np.float64).squeeze()
            if affine.shape != (4, 4):
                affine = np.eye(4)

            # 2) 同一个条件（抹洞后的潜表示 + mask）分别喂给两个生成器。
            #    条件只构造一次，确保 FM/DDPM 的输入逐位相同。
            cond = build_cond(fm.vqgan, img_c, tumor, device)
            fm_norm = sample_fm(fm, cond, args.fm_steps)
            ddpm_norm = sample_ddpm(tester, cond)

            # 3) [-1,1] -> HU，只在肿瘤体素上取分布
            real_hu_all = to_hu(img_c)
            real_hu = real_hu_all[tumor]
            fm_hu_all = to_hu(fm_norm)
            ddpm_hu_all = to_hu(ddpm_norm)
            fm_hu = fm_hu_all[tumor]
            ddpm_hu = ddpm_hu_all[tumor]

            # 4) 肿瘤外相关性：检查生成有没有破坏周围解剖结构。
            #    这里只是普通相关系数，正式的 SSIM/PSNR 在 texture_metrics.py 里。
            outside = ~tumor
            corr_fm = float(np.corrcoef(real_hu_all[outside], fm_hu_all[outside])[0, 1])
            corr_ddpm = float(np.corrcoef(real_hu_all[outside], ddpm_hu_all[outside])[0, 1])
            case_meta.append({"case": case, "n_tumor": n_tumor,
                              "corr_outside_fm": corr_fm,
                              "corr_outside_ddpm": corr_ddpm})

            fm_match = quantile_match(fm_hu, real_hu)
            ddpm_match = quantile_match(ddpm_hu, real_hu)
            fm_match_all = fm_hu_all.copy()
            ddpm_match_all = ddpm_hu_all.copy()
            fm_match_all[tumor] = fm_match
            ddpm_match_all[tumor] = ddpm_match

            rng = np.random.default_rng(args.seed + 10000 + i)
            ref = split_half_reference(real_hu, rng)

            for method, gen in [
                ("fm", fm_hu),
                ("ddpm", ddpm_hu),
                ("fm_match", fm_match),
                ("ddpm_match", ddpm_match),
            ]:
                r = compare(real_hu, gen)
                r.update({"case": case, "n_tumor": n_tumor, "method": method})
                all_rows.append(r)

            if ref is not None:
                ref.update({"case": case, "n_tumor": n_tumor, "method": "real_ref"})
                all_rows.append(ref)

            # save full crops (not only tumor voxels)
            save_case_outputs(out / "cases" / case, to_hu(img_c),
                              fm_hu_all, ddpm_hu_all, fm_match_all,
                              tumor, start, affine)

            print(f"[{i+1}/{len(rows)}] {case}: n_tumor={n_tumor} "
                  f"FM W={compare(real_hu, fm_hu)['wasserstein']:.1f} "
                  f"DDPM W={compare(real_hu, ddpm_hu)['wasserstein']:.1f} "
                  f"corr_out FM={corr_fm:.3f} DDPM={corr_ddpm:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(rows)}] {case}: ERROR {e!r}")

    if case_meta:
        with (out / "case_meta.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(case_meta[0].keys()))
            w.writeheader()
            w.writerows(case_meta)

    if not all_rows:
        print("no results")
        return

    # per-case CSV
    fields = sorted({k for r in all_rows for k in r})
    with (out / "metrics_per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # summary
    methods = ["fm", "ddpm", "fm_match", "ddpm_match", "real_ref"]
    metric_keys = ["wasserstein", "ks", "js",
                   "d_mean", "d_std", "d_skew", "d_kurt", "d_entropy"]
    summary = []
    for m in methods:
        rs = [r for r in all_rows if r["method"] == m]
        if not rs:
            continue
        row = {"method": m, "n": len(rs)}
        for k in metric_keys:
            vals = np.array([r[k] for r in rs if k in r and np.isfinite(r[k])])
            row[f"{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{k}_std"] = float(vals.std()) if len(vals) else np.nan
        summary.append(row)

    sfields = sorted({k for r in summary for k in r})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    with (out / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Texture measurement summary\n\n")
        f.write(f"cases: {len({r['case'] for r in all_rows})}, seed: {args.seed}\n\n")
        f.write("| method | n | Wasserstein | KS | JS | d_mean | d_std | d_skew | d_kurt | d_entropy |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            f.write(f"| {r['method']} | {r['n']} | "
                    f"{r['wasserstein_mean']:.2f}±{r['wasserstein_std']:.2f} | "
                    f"{r['ks_mean']:.3f}±{r['ks_std']:.3f} | "
                    f"{r['js_mean']:.4f}±{r['js_std']:.4f} | "
                    f"{r['d_mean_mean']:.2f}±{r['d_mean_std']:.2f} | "
                    f"{r['d_std_mean']:.2f}±{r['d_std_std']:.2f} | "
                    f"{r['d_skew_mean']:.3f}±{r['d_skew_std']:.3f} | "
                    f"{r['d_kurt_mean']:.3f}±{r['d_kurt_std']:.3f} | "
                    f"{r['d_entropy_mean']:.3f}±{r['d_entropy_std']:.3f} |\n")

    # aggregate histogram plot
    bins = np.linspace(A_MIN, A_MAX, 65)
    centers = 0.5 * (bins[:-1] + bins[1:])
    curves = {}
    # build from saved case arrays
    for m in ("real", "fm", "ddpm", "fm_match"):
        hs = []
        for cp in sorted((out / "cases").glob("*/arrays.npz")):
            z = np.load(cp)
            tumor = z["tumor"]
            arr = z["real_hu" if m == "real" else f"{m}_hu"]
            v = arr[tumor]
            p, _ = np.histogram(v, bins=bins)
            p = p / max(p.sum(), 1)
            hs.append(p)
        if hs:
            curves[m] = np.mean(hs, axis=0)
    if curves:
        plt.figure(figsize=(7, 4))
        for m, c in curves.items():
            plt.plot(centers, c, label=m)
        plt.xlabel("HU")
        plt.ylabel("mean normalized frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "histogram_overlay.png", dpi=150)
        plt.close()

    # boxplots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ["wasserstein", "ks", "js"]):
        data = [[r[key] for r in all_rows if r["method"] == m and key in r]
                for m in methods if any(r["method"] == m for r in all_rows)]
        labels = [m for m in methods if any(r["method"] == m for r in all_rows)]
        ax.boxplot(data, labels=labels)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "boxplots.png", dpi=150)
    plt.close()

    print(f"\n[done] {out}")
    print("files: metrics_per_case.csv, summary.csv, summary.md, "
          "histogram_overlay.png, boxplots.png, cases/*")


if __name__ == "__main__":
    main()
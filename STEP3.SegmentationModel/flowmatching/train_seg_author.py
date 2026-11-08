"""
参考 STEP3/main.py + monai_trainer.py 重写的分割训练入口。

保留作者的核心操作：
- `RandCropByPosNegLabeld_select`（健康 CT 的器官前景重试裁剪）
- 完整 train/val transform 链
- 3D U-Net (16,32,64,128,256) + DiceCELoss
- AdamW + LinearWarmupCosineAnnealingLR（warmup_epochs=100）
- 在线合成：健康 CT 以 50% 概率注入肿瘤，80% early / 10% medium / 10% large
- sliding window validation + organ pseudo + denoise_pred + Dice
- model.pt / model_final.pt checkpoint

与作者代码的差异（为了 FM vs DDPM 对照）：
1. early 合成器可切换：FM 8 步 或 DDPM 4 步；
2. medium/large 两组建模器固定使用作者的 liver_noearly.pt（两组相同）；
3. 合成器在训练开始时加载一次，不每个 epoch 重新加载；
4. DataLoader 使用独立 seeded generator，保证两组数据顺序一致；
5. val_epoch 额外计算 tumor NSD(2mm) 和 tumor-wise sensitivity；
6. 支持 --arm real 时只使用真实肿瘤病例（过滤掉健康 CT 行）。

用法：
  python flowmatching/train_seg_author.py --arm fm --synt_model fm \
      --max_epochs 2000 --val_every 200 --save_checkpoint \
      --logdir flowmatching/runs_author/fm
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import csv
import math
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai import transforms, data
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

HERE = Path(__file__).resolve().parent
STEP3_DIR = HERE.parent
STEP2_DIR = STEP3_DIR.parent / "STEP2.DiffusionModel"
for _p in (str(STEP3_DIR), str(STEP2_DIR),
           str(STEP3_DIR / "external" / "surface-distance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# STEP2 也有一个名为 flowmatching 的包；确保 STEP3 的包优先解析。
sys.path.insert(0, str(STEP3_DIR))

from monai_trainer import (  # noqa: E402
    AverageMeter, denoise_pred, convert_to_one_hot, dice, resample,
    save_checkpoint)
from TumorGeneration.utils import (  # noqa: E402
    synthesize_early_tumor, synthesize_medium_tumor, synthesize_large_tumor)
from TumorGeneration.ldm.ddpm.ddim import DDIMSampler  # noqa: E402
from TumorGeneration.ldm.ddpm import Unet3D, GaussianDiffusion  # noqa: E402
from flowmatching.fm_synth import load_vqgan, load_fm, load_ddpm  # noqa: E402
from surface_distance import (  # noqa: E402
    compute_surface_distances, compute_surface_dice_at_tolerance)


# ----------------------------------------------------------------------
# 与 STEP3/main.py 一致的 transforms
# ----------------------------------------------------------------------
class RandCropByPosNegLabeld_select(transforms.RandCropByPosNegLabeld):
    """作者的自定义随机裁剪：健康 CT 必须裁到足够多的器官前景。

    为什么需要它：
        健康 CT 没有肿瘤标签，普通的 pos/neg 随机裁剪可能裁到纯背景，
        这样的样本对「在有器官的地方注入肿瘤」没有意义。
        这里对健康 CT 反复重裁，直到器官前景体素数超过一个阈值，
        阈值随重试次数逐步放宽，避免极小器官导致死循环。

    注意两件事：
        1. `name` 键是一个字符串，MONAI 的裁剪算子会对所有 keys 取 .shape，
           所以必须先 pop 掉、裁完再放回去（作者原实现就是这样）。
        2. num_samples=1 会让 DataLoader 返回 [dict] 而不是 dict，
           调用方需要取 batch_data[0]。
    """

    def __init__(self, keys, label_key, spatial_size,
                 pos=1.0, neg=1.0, num_samples=1,
                 image_key=None, image_threshold=0.0, allow_missing_keys=True,
                 fg_thresh=0):
        super().__init__(keys=keys, label_key=label_key, spatial_size=spatial_size,
                         pos=pos, neg=neg, num_samples=num_samples,
                         image_key=image_key, image_threshold=image_threshold,
                         allow_missing_keys=allow_missing_keys)
        self.fg_thresh = fg_thresh

    def R2voxel(self, R):
        """把「等效半径 R mm」换算成球体体素数（1mm 各向同性）。"""
        return (4 / 3 * np.pi) * (R) ** 3

    def __call__(self, data):
        d = dict(data)
        data_name = d["name"]
        # 先移除非张量字段，否则父类的 key_iterator 会对字符串取 shape 报错
        d.pop("name")
        if ("kidney_label" in data_name or "liver_label" in data_name
                or "pancreas_label" in data_name):
            flag = 0
            while 1:
                flag += 1
                d_crop = super().__call__(d)
                pixel_num = (d_crop[0]["label"] > 0).sum()
                if pixel_num > self.R2voxel(self.fg_thresh):
                    break
                if flag > 5 and pixel_num > self.R2voxel(max(self.fg_thresh - 5, 5)):
                    break
                if flag > 10 and pixel_num > self.R2voxel(max(self.fg_thresh - 10, 5)):
                    break
                if flag > 15 and pixel_num > self.R2voxel(max(self.fg_thresh - 15, 5)):
                    break
                if flag > 20 and pixel_num > self.R2voxel(max(self.fg_thresh - 20, 5)):
                    break
                if flag > 25 and pixel_num > self.R2voxel(max(self.fg_thresh - 25, 5)):
                    break
                if flag > 30:
                    break
        else:
            d_crop = super().__call__(d)
        d_crop[0]["name"] = data_name
        return d_crop


class LoadImage_train(transforms.MapTransform):
    """训练集读图：按器官类型做标签重映射。

    liver 分支不需要特殊处理（标签 0/1/2 直接可用）；
    kidney 分支把肿瘤标签 3 归并到 1，并把健康 CT 的所有前景统一成 1。
    本实验 organ_type="liver"，走的是最后的 else 分支。
    """

    def __init__(self, organ_type):
        self.reader1 = transforms.LoadImaged(keys=["image", "label"])
        self.organ_type = organ_type

    def __call__(self, data):
        d = dict(data)
        data_name = d["name"]
        if (not "kidney_label" in data_name) and self.organ_type == "kidney":
            d = self.reader1.__call__(d)
            d["label"][d["label"] == 3] = 1
        elif ("kidney_label" in data_name) and self.organ_type == "kidney":
            d = self.reader1.__call__(d)
            d["label"][d["label"] > 0] = 1
        else:
            d = self.reader1.__call__(d)
        return d


class LoadImage_val(transforms.LoadImaged):
    """验证集读图：把肿瘤标签 3 归并到 1。

    验证集标签来自 Task03_Liver 的预处理标签：
        0=背景, 1=肝脏, 2=肿瘤, 3=其他
    这里只保留 0/1/2，与训练时的类别定义一致。
    """

    def __init__(self, keys, *args, **kwargs):
        # 故意忽略 *args/**kwargs：作者原代码只传 keys，
        # 多传的参数（如 dtype）不会被使用，保证与作者行为一致。
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        d = super().__call__(d)
        d["label"][d["label"] == 3] = 1
        return d


def _seed_worker(worker_id):
    """给每个 DataLoader worker 设置可复现的随机种子。

    torch 只负责给 worker 一个基础种子（initial_seed），
    而 MONAI 的随机裁剪用的是 numpy，Python 侧的几何/注入决策用的是
    random 模块，所以两者都要手动 seed。

    seed + worker_id：同一基础的多个 worker 之间种子不同，
    但又完全由 DataLoader 的 generator 种子决定，
    因此 FM 和 DDPM 两组在每个 epoch 会拿到相同的裁剪。
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _get_transform(args):
    train_transform = transforms.Compose([
        LoadImage_train(args.organ_type),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"],
                               spatial_size=[96, 96, 96]),
        # 必须放在 Spacingd 之后：Spacingd 会把 label 重新转成 float32。
        # 标签只有 0/1/2，缓存里用 uint8 存是无损的，载入后再转回 float32。
        transforms.CastToTyped(keys=["label"], dtype=np.uint8),
        RandCropByPosNegLabeld_select(
            keys=["image", "label", "name"], label_key="label",
            spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=1,
            image_key="image", image_threshold=0, fg_thresh=args.fg_thresh),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        transforms.ToTensord(keys=["image", "label"]),
        transforms.CastToTyped(keys=["label"], dtype=np.float32),
    ])
    val_transform = transforms.Compose([
        LoadImage_val(keys=["image", "label", "organ_pseudo"]),
        transforms.AddChanneld(keys=["image", "label", "organ_pseudo"]),
        transforms.Orientationd(keys=["image", "label", "organ_pseudo"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label", "organ_pseudo"],
                            pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        transforms.SpatialPadd(keys=["image", "label", "organ_pseudo"],
                               mode=["minimum", "constant", "constant"],
                               spatial_size=[96, 96, 96]),
        transforms.ToTensord(keys=["image", "label", "organ_pseudo"]),
    ])
    return train_transform, val_transform


# ----------------------------------------------------------------------
# 合成器加载
# ----------------------------------------------------------------------
def prepare_synth(args, device):
    """加载合成所需的三样东西，返回 (vqgan, early, noearly_sampler)。

    vqgan           : 冻结的 VQGAN，负责图像 <-> 潜表示编解码
    early           : early 肿瘤合成器
                      - --synt_model fm   -> 我们训练的 FM 10k（8 步）
                      - --synt_model ddpm -> 我们训练的 DDPM 10k（4 步）
    noearly_sampler : 作者提供的 liver_noearly.pt + DDIM 采样器，
                      专门用于 medium / large 肿瘤

    【为什么 medium/large 固定用作者的权重】
        对照实验只关心 early 合成器的差异。如果 medium/large 也各训一个，
        变量就不止一个；固定成同一份权重后，FM 组和 DDPM 组的差别
        被严格限制在 early 分支上。

    【为什么只加载一次】
        作者原版每个 epoch 都会重新调用 synt_model_prepare（重新读盘、
        重新建 hydra 配置），在 2000 epoch 下浪费大量时间。
        这些模型全程 eval、不更新，加载一次即可。
    """
    vqgan = load_vqgan(args.vqgan_ckpt, device)
    if args.synt_model == "fm":
        early = load_fm(args.fm_ckpt, args.vqgan_ckpt, device, steps=args.fm_steps)
    else:
        early = load_ddpm(args.ddpm_ckpt, args.vqgan_ckpt, device)

    # hydra 的 config_path 是相对「调用 initialize 的文件」解析的，
    # 本文件在 flowmatching/ 下，所以必须用绝对目录。
    from hydra import initialize_config_dir, compose
    cfg_dir = str(STEP3_DIR / "TumorGeneration" / "diffusion_config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="ddpm.yaml")

    # noearly 用的是 200 步扩散 + DDIM 50 步加速，
    # 与 early（4 步）不同：中/大肿瘤需要更多去噪步数才能保持结构。
    net = Unet3D(dim=cfg.diffusion_img_size, dim_mults=cfg.dim_mults,
                 channels=cfg.diffusion_num_channels, out_dim=cfg.out_dim).to(device)
    noearly = GaussianDiffusion(
        net, vqgan_ckpt=args.vqgan_ckpt,
        image_size=cfg.diffusion_img_size,
        num_frames=cfg.diffusion_depth_size,
        channels=cfg.diffusion_num_channels,
        timesteps=200, loss_type=cfg.loss_type, device=device).to(device)
    ck = torch.load(args.noearly_ckpt, map_location=device)
    noearly.load_state_dict(ck["ema"])
    noearly_sampler = DDIMSampler(noearly, schedule="cosine")
    print(f"[synth] early={args.synt_model}, noearly=author liver_noearly.pt", flush=True)
    return vqgan, early, noearly_sampler


# ----------------------------------------------------------------------
# train_epoch：与 monai_trainer.py 一致，合成器只加载一次
# ----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args,
                synth_models, max_iters=0):
    """训练一个 epoch。

    核心逻辑（与作者 monai_trainer.train_epoch 一致）：

        for 每个 batch:
            对 batch 里的每个样本：
                如果它是健康 CT，且 random.random() > 0.5：
                    按 [0.8, 0.1, 0.1] 抽 early / medium / large
                    用对应的生成器合成肿瘤，覆盖掉原图和标签
            前向 -> DiceCE loss -> 反向

    为什么是 50% 注入：
        健康 CT 本身没有肿瘤，如果每例都注入，真实肿瘤样本的比例会被稀释；
        50% 让每个 epoch 里「真实肿瘤 : 合成肿瘤」大致相当。

    synth_models=None 表示 real-only 组：此时一旦遇到健康 CT 就报错，
    说明数据配对逻辑出错了（正常情况下 --arm real 已把健康行过滤掉）。
    """
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    sample_thresh = 0.5
    vqgan, early_sampler, noearly_sampler = (synth_models if synth_models is not None
                                             else (None, None, None))

    for idx, batch_data in enumerate(loader):
        if max_iters and idx >= max_iters:
            break
        # RandCropByPosNegLabeld(num_samples=1) 会把一个 batch 包成 [dict]，
        # 取第 0 个元素才是真正的 batch dict。
        if isinstance(batch_data, list):
            batch_data = batch_data[0]
        data, target, data_names = (batch_data["image"], batch_data["label"],
                                    batch_data["name"])
        data, target = data.cuda(), target.cuda()

        # 逐样本判断是否需要注入肿瘤。
        # 注意 random.random() / np.random.choice 的顺序在 FM 和 DDPM 两组
        # 完全一致（训练开始前重置过种子），所以两组抽到的肿瘤类型、
        # 几何形状、羽化 sigma 都是配对的——唯一区别是 early 分支的生成器。
        for bs in range(data.shape[0]):
            data_name = data_names[bs]
            if ("kidney_label" in data_name or "liver_label" in data_name
                    or "pancreas_label" in data_name):
                if synth_models is None:
                    raise RuntimeError("healthy CT found in real-only arm")
                if random.random() > sample_thresh:
                    # 保持 batch 维，因为合成函数按 (B,1,96,96,96) 工作
                    healthy_data = data[bs][None, ...]
                    healthy_target = target[bs][None, ...]
                    # 80% early / 10% medium / 10% large，与作者一致
                    tumor_types = ["early", "medium", "large"]
                    tumor_probs = np.array([0.8, 0.1, 0.1])
                    synthetic_tumor_type = np.random.choice(
                        tumor_types, p=tumor_probs.ravel())
                    if synthetic_tumor_type == "early":
                        synt_data, synt_target = synthesize_early_tumor(
                            healthy_data, healthy_target, args.organ_type,
                            vqgan, early_sampler)
                    elif synthetic_tumor_type == "medium":
                        synt_data, synt_target = synthesize_medium_tumor(
                            healthy_data, healthy_target, args.organ_type,
                            vqgan, noearly_sampler, ddim_ts=args.ddim_ts)
                    else:
                        synt_data, synt_target = synthesize_large_tumor(
                            healthy_data, healthy_target, args.organ_type,
                            vqgan, noearly_sampler, ddim_ts=args.ddim_ts)
                    data[bs, ...] = synt_data[0]
                    target[bs, ...] = synt_target[0]

        # detach：合成过程在 no_grad 里完成，这里确保 data/target 不带计算图。
        # 手动清 grad 而不是 optimizer.zero_grad()，与作者写法一致
        # （效果相同，但更轻量）。
        data = data.detach()
        target = target.detach()
        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)
        print("Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
              "loss: {:.4f}".format(run_loss.avg),
              "time {:.2f}s".format(time.time() - start_time), flush=True)
        start_time = time.time()

    for param in model.parameters():
        param.grad = None
    return run_loss.avg


# ----------------------------------------------------------------------
# val_epoch：作者 Dice + 额外 NSD(2mm) / sensitivity
# ----------------------------------------------------------------------
def val_epoch(model, loader, val_shape_dict, epoch, loss_func, args,
              model_inferer=None, max_iters=0):
    """验证一个 epoch，返回 (loss, [肝脏Dice, 肿瘤Dice], tumor_loss, NSD, 检出率)。

    计算分两层：
        1mm 网格（滑窗推理的原始输出）
            - tumor-wise sensitivity：只要预测出任意肿瘤体素就算检出
            - NSD@2mm：归一化表面距离，衡量边界吻合度
        原始分辨率（与作者 val_epoch 完全一致）
            - 先 resample 回原始尺寸，再用器官伪标签做 denoise_pred
            - 肝脏 Dice 用原始预测，肿瘤 Dice 用去噪后的预测
              （作者原逻辑如此，保持可比性）

    model_inferer 是 sliding_window_inference 的 partial；
    若为 None 则直接前向（本脚本始终传 inferer）。
    """
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    val_tumor_loss = AverageMeter()
    nsd_list, det_list = [], []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if max_iters and idx >= max_iters:
                break
            if isinstance(batch_data, list):
                batch_data = batch_data[0]
            data, target, organ_pseudo = (batch_data["image"],
                                          batch_data["label"],
                                          batch_data["organ_pseudo"])

            data, target = data.cuda(), target.cuda()
            organ_pseudo = organ_pseudo.cuda()

            with autocast(enabled=args.amp):
                logits = model_inferer(data) if model_inferer is not None else model(data)

            loss = loss_func(logits, target)
            tumor_logits = logits.detach()
            tumor_loss = loss_func(
                torch.stack([tumor_logits[:, 0], tumor_logits[:, 2]], dim=1),
                target == 2)

            # 1mm 空间上算 NSD / detection。
            # 这里的 logits 覆盖整个（padding 后的）体积，不是 96³ crop。
            pred_1mm = np.argmax(torch.softmax(logits, 1).cpu().numpy(), axis=1).astype(np.uint8)
            y_1mm = target.cpu().numpy()[:, 0]
            organ_1mm = organ_pseudo.cpu().numpy()[:, 0]
            pred_oh = convert_to_one_hot(pred_1mm[0], 3)
            processed_1mm = denoise_pred(pred_oh, organ_1mm[0])
            tumor_pred = processed_1mm[2].astype(bool)
            tumor_gt = (y_1mm[0] == 2)
            inter = int((tumor_pred & tumor_gt).sum())
            det_list.append(int(inter > 0))
            try:
                sd = compute_surface_distances(tumor_gt, tumor_pred,
                                               spacing_mm=(1.0, 1.0, 1.0))
                nsd_list.append(float(compute_surface_dice_at_tolerance(sd, 2.0)))
            except Exception:
                nsd_list.append(0.0)

            # 作者原始逻辑：resample 回原始尺寸后算 Dice
            name = Path(batch_data["image_meta_dict"]["filename_or_obj"][0]).name
            val_shape = val_shape_dict[name]
            pred = resample(pred_1mm[0], val_shape)
            y = resample(y_1mm[0], val_shape)
            organ_pseudo_np = resample(organ_1mm[0], val_shape)
            pred = convert_to_one_hot(pred, 3)
            processed_pred = denoise_pred(pred, organ_pseudo_np)

            dice_list_sub = []
            for i in range(1, args.num_classes):
                if i == 1:
                    organ_dice = dice(pred[i, ...], y == i)
                else:
                    organ_dice = dice(processed_pred[i, ...], y == i)
                dice_list_sub.append(organ_dice)

            avg_classes = np.array(dice_list_sub)
            run_acc.update(avg_classes, n=args.batch_size)
            run_loss.update(loss.item(), n=args.batch_size)
            val_tumor_loss.update(tumor_loss.item(), n=args.batch_size)

            print("Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                  "loss: {:.4f}".format(run_loss.avg),
                  "tumor_loss {:.4f}".format(val_tumor_loss.avg),
                  "acc", run_acc.avg,
                  "acc_avg: {:.4f}".format(np.mean(run_acc.avg)),
                  "time {:.2f}s".format(time.time() - start_time), flush=True)
            start_time = time.time()

            # 与作者 monai_trainer.val_epoch 一致：每个病例后释放大数组，防止显存碎片
            del pred, y, data, target, organ_pseudo, processed_pred
            del pred_1mm, y_1mm, organ_1mm, pred_oh, processed_1mm
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    return (run_loss.avg, run_acc.avg, val_tumor_loss.avg,
            float(np.mean(nsd_list)), float(np.mean(det_list)))


# ----------------------------------------------------------------------
# run_training：与 monai_trainer.py 一致，增加 NSD/sensitivity 记录
# ----------------------------------------------------------------------
def run_training(model, train_loader, val_loader, optimizer, loss_func, args,
                 val_shape_dict, model_inferer=None, scheduler=None,
                 start_epoch=0, val_channel_names=None, val_acc_max=0.0,
                 synth_models=None):
    """完整训练循环（与作者 monai_trainer.run_training 对齐）。

    checkpoint 约定：
        model.pt        验证集肿瘤 Dice 最高时保存的那一份（用于最终评估）
        model_final.pt  每个 epoch 都覆盖，保存最新状态（用于续训/排查）
    """
    writer = SummaryWriter(log_dir=args.logdir) if args.logdir and args.rank == 0 else None
    scaler = GradScaler() if args.amp else None
    metric_rows = []

    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch, flush=True)
        epoch_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch,
                                 loss_func, args, synth_models,
                                 max_iters=args.max_iters)
        print("Final training {}/{} loss: {:.4f} time {:.2f}s".format(
            epoch, args.max_epochs - 1, train_loss, time.time() - epoch_time), flush=True)
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        # 只有到验证间隔才跑滑窗推理；验证很贵（23 例 × 完整体积）。
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            val_loss, val_acc, val_tumor_loss, val_nsd, val_det = val_epoch(
                model, val_loader, val_shape_dict, epoch=epoch,
                loss_func=loss_func, model_inferer=model_inferer, args=args,
                max_iters=args.max_val_iters)
            print("Final validation {}/{} loss: {:.4f} acc {} nsd {:.4f} det {:.4f}".format(
                epoch, args.max_epochs - 1, val_loss, val_acc, val_nsd, val_det), flush=True)
            metric_rows.append({"epoch": epoch, "organ_dice": float(val_acc[0]),
                                "tumor_dice": float(val_acc[1]),
                                "tumor_nsd": val_nsd, "sensitivity": val_det})
            if args.logdir:
                with (Path(args.logdir) / "val_metrics.csv").open(
                        "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
                    w.writeheader(); w.writerows(metric_rows)
            if writer is not None:
                writer.add_scalar("val_loss", val_loss, epoch)
                writer.add_scalar("val_tumor_loss", val_tumor_loss, epoch)
                writer.add_scalar("val_mean_dice", np.mean(val_acc), epoch)
                writer.add_scalar("val_tumor_dice", val_acc[1], epoch)
                writer.add_scalar("val_tumor_nsd", val_nsd, epoch)
                writer.add_scalar("val_sensitivity", val_det, epoch)
                if val_channel_names is not None:
                    for i, name in enumerate(val_channel_names):
                        if i < val_acc.size:
                            writer.add_scalar(name, val_acc[i], epoch)

            # 以「肿瘤 Dice」而不是平均 Dice 作为最佳模型判据：
            # 肝脏分割很容易到 0.9+，会把肿瘤的差异淹没。
            if val_acc[1] > val_acc_max:
                print("new best ({:.6f} --> {:.6f}).".format(val_acc_max, val_acc[1]), flush=True)
                val_acc_max = val_acc[1]
                b_new_best = True
                if args.save_checkpoint:
                    save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                                    optimizer=optimizer, scheduler=scheduler)

        if args.logdir and args.save_checkpoint:
            # 同时保存 optimizer / scheduler，供 --resume 精确恢复。
            # 文件会从 ~19 MB 增到 ~57 MB，但换来真正的断点续训。
            save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                            filename="model_final.pt",
                            optimizer=optimizer, scheduler=scheduler)
            if b_new_best:
                import shutil
                shutil.copyfile(Path(args.logdir) / "model_final.pt",
                                Path(args.logdir) / "model.pt")

        if scheduler is not None:
            scheduler.step()

    print("monai_trainer DONE, best acc", val_acc_max, flush=True)
    return val_acc_max


# ----------------------------------------------------------------------
# 断点续训
# ----------------------------------------------------------------------
def resume_from_checkpoint(model, optimizer, scheduler, ckpt_path, device):
    """从 checkpoint 恢复模型 / 优化器 / 学习率调度器。

    返回 (start_epoch, best_acc)。

    两种情况：
        1. checkpoint 里有 optimizer / scheduler（本次改动之后保存的）：
           精确恢复，续训曲线与不中断基本一致；
        2. 旧格式只有 state_dict（例如当前 fm 的 epoch 96）：
           AdamW 的一阶/二阶动量会重置，学习率通过逐步 step 推演到正确值，
           续训开头会有几轮轻微扰动。
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])

    start_epoch = int(ck.get("epoch", -1)) + 1
    best_acc = float(ck.get("best_acc", 0.0))

    if "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
        print("[resume] optimizer state restored", flush=True)
    else:
        print("[resume] no optimizer state -> AdamW moments reset", flush=True)

    if scheduler is not None:
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
            print("[resume] scheduler state restored", flush=True)
        else:
            # 旧 checkpoint：spawn 一个 scheduler 需要逐步 step 才能对到 epoch
            for _ in range(start_epoch):
                scheduler.step()
            print(f"[resume] scheduler fast-forwarded to epoch {start_epoch}",
                  flush=True)

    print(f"[resume] {ckpt_path} -> start_epoch={start_epoch}, "
          f"best={best_acc:.4f}", flush=True)
    return start_epoch, best_acc


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="real", choices=["real", "fm", "ddpm"])
    parser.add_argument("--synt_model", default="ddpm", choices=["ddpm", "fm"])
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--optim_lr", default=2e-4, type=float)
    parser.add_argument("--optim_name", default="adamw", type=str)
    parser.add_argument("--reg_weight", default=1e-5, type=float)
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str)
    parser.add_argument("--warmup_epochs", default=100, type=int)
    parser.add_argument("--val_every", default=200, type=int)
    parser.add_argument("--val_overlap", default=0.75, type=float)
    parser.add_argument("--cache_rate", default=0.5, type=float)
    parser.add_argument("--cache_mode", default="disk",
                        choices=["disk", "memory", "none"],
                        help="disk=PersistentDataset（Windows 可用，跨 arm 共享）；"
                             "memory=CacheDataset（Linux/fork 才建议）；none=不缓存")
    parser.add_argument("--cache_dir", default=str(HERE / "cache_persistent"))
    parser.add_argument("--cache_fill_workers", default=8, type=int,
                        help="仅 memory 模式：并行填充内存缓存的线程数，"
                             "与训练时的 --workers 无关")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--logdir", default=None, type=str)
    parser.add_argument("--max_iters", default=0, type=int,
                        help="only for smoke; 0 = full epoch")
    parser.add_argument("--max_val_iters", default=0, type=int,
                        help="only for smoke; 0 = full validation set")
    parser.add_argument("--resume", default=None, type=str,
                        help="'auto' 表示 <logdir>/model_final.pt；"
                             "也可传具体 checkpoint 路径")
    parser.add_argument("--seed", default=1234, type=int)

    parser.add_argument("--data_root", default="E:/thesisAndCode/data/Task03_Liver/")
    parser.add_argument("--healthy_data_root",
                        default="E:/thesisAndCode/data/HealthyCT/extracted/healthy_ct/")
    parser.add_argument("--datafold_dir",
                        default=str(STEP3_DIR / "cross_eval" / "liver_aug_data_fold"))
    parser.add_argument("--model_name", default="unet", choices=["unet", "swinunetr"])
    parser.add_argument("--swin_type", default="base", type=str)
    parser.add_argument("--tumor_type", default="tumor", type=str)
    parser.add_argument("--organ_type", default="liver", type=str)
    parser.add_argument("--organ_model", default="liver", type=str)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--fg_thresh", default=30, type=int)

    parser.add_argument("--vqgan_ckpt",
                        default=str(STEP3_DIR / "TumorGeneration" / "model_weight" / "AutoencoderModel.ckpt"))
    parser.add_argument("--fm_ckpt",
                        default=str(STEP2_DIR / "flowmatching" / "texture_study" / "checkpoints" / "liver_crops_10k" / "5.pt"))
    parser.add_argument("--ddpm_ckpt",
                        default=str(STEP2_DIR / "flowmatching" / "texture_study" / "checkpoints" / "liver_crops_10k_ddpm" / "5.pt"))
    parser.add_argument("--noearly_ckpt",
                        default=str(STEP3_DIR / "TumorGeneration" / "model_weight" / "liver_noearly.pt"))
    parser.add_argument("--fm_steps", default=8, type=int)
    parser.add_argument("--ddim_ts", default=50, type=int)

    args = parser.parse_args()
    os.chdir(STEP3_DIR)
    if args.logdir is None:
        args.logdir = str(HERE / "runs_author" / args.arm)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    args.amp = not args.noamp
    args.rank = 0
    args.distributed = False

    # 统一随机源：模型初始化、合成几何、注入决策都由这里决定。
    # FM 和 DDPM 两组使用同一个 seed，因此除 early 生成器外的一切都配对。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 作者原代码设置 benchmark=True；保持一致。
    # 这会牺牲一点位级确定性，但不影响数据顺序与合成几何的配对。
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} [arm] {args.arm} [synt] {args.synt_model}", flush=True)

    train_transform, val_transform = _get_transform(args)
    datafold_dir = Path(args.datafold_dir)
    train_txt = datafold_dir / f"real_{args.tumor_type}_train_{args.fold}.txt"
    val_txt = datafold_dir / f"real_{args.tumor_type}_val_{args.fold}.txt"

    real_img, real_lbl, real_name = [], [], []
    healthy_img, healthy_lbl, healthy_name = [], [], []
    # 数据配对：
    #   - 名字里带 liver_label 的是健康 CT（器官伪标签），图像在 HealthyCT 下；
    #   - 其余是真实肿瘤病例，图像在 Task03_Liver 下。
    # real 组只保留真实肿瘤，健康 CT 整行跳过。
    for line in train_txt.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name = parts[1].split(".")[0]
        if ("kidney_label" in name or "liver_label" in name or "pancreas_label" in name):
            if args.arm == "real":
                continue
            healthy_img.append(args.healthy_data_root + parts[0])
            healthy_lbl.append(args.healthy_data_root + parts[1])
            healthy_name.append(name)
        else:
            real_img.append(args.data_root + parts[0])
            real_lbl.append(args.data_root + parts[1])
            real_name.append(name)

    train_img = healthy_img + real_img
    train_lbl = healthy_lbl + real_lbl
    train_name = healthy_name + real_name
    data_dicts_train = [{"image": i, "label": l, "name": n}
                        for i, l, n in zip(train_img, train_lbl, train_name)]
    print(f"[train] real={len(real_img)} healthy={len(healthy_img)} "
          f"total={len(data_dicts_train)}", flush=True)

    # 验证集：23 例真实肿瘤 + 对应的器官伪标签（用于 denoise_pred）。
    # 伪标签文件名与真实标签同名，放在 organ_pseudo_swin_new/liver/ 下。
    val_img, val_lbl, val_name, val_pseudo = [], [], [], []
    for line in val_txt.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name = parts[1].split(".")[0]
        val_img.append(args.data_root + parts[0])
        val_lbl.append(args.data_root + parts[1])
        val_pseudo.append("organ_pseudo_swin_new/{}/".format(args.organ_type)
                          + Path(parts[1]).name)
        val_name.append(name)
    data_dicts_val = [{"image": i, "label": l, "organ_pseudo": p, "name": n}
                      for i, l, p, n in zip(val_img, val_lbl, val_pseudo, val_name)]
    print(f"[val] {len(data_dicts_val)}", flush=True)

    # 原始分辨率：验证时要把 1mm 预测 resample 回去再算 Dice。
    # 用文件名做 key，val_epoch 里通过 image_meta_dict 的 filename 查回来。
    val_shape_dict = {}
    for d in data_dicts_val:
        img = nib.load(d["image"])
        val_shape_dict[Path(d["image"]).name] = list(img.shape)

    if args.cache_mode == "disk":
        # 只把「第一个随机变换之前」的确定性预处理写到磁盘；
        # 随机裁剪/翻转/旋转/强度扰动每个 epoch 照常重新执行。
        train_ds = data.PersistentDataset(data=data_dicts_train,
                                          transform=train_transform,
                                          cache_dir=args.cache_dir)
        print(f"[cache] PersistentDataset dir={args.cache_dir}", flush=True)
    elif args.cache_mode == "memory":
        # 作者的原始做法：确定性预处理结果缓存在内存。
        # Windows 的限制：DataLoader 用 spawn，会把整个 Dataset 序列化后
        # 通过管道发给每个 worker；缓存几十 GB 时这里会失败
        # （OSError 22 / pickle truncated），与虚拟内存无关。
        # 因此 Windows 上建议配合 --workers 0 使用（缓存留在主进程）。
        if os.name == "nt" and args.workers > 0:
            print("[warn] Windows + workers>0：CacheDataset 的缓存会被序列化"
                  "给每个 worker，很可能崩溃；建议 --workers 0", flush=True)
        train_ds = data.CacheDataset(
            data=data_dicts_train, transform=train_transform,
            cache_rate=args.cache_rate,
            # MONAI 的这个 num_workers 用线程池并行填充缓存（启动阶段），
            # 与训练时的 DataLoader workers 无关；即使训练用 --workers 0，
            # 这里也可以开大以缩短缓存预热时间。
            num_workers=max(args.cache_fill_workers, 1))
        print(f"[cache] CacheDataset rate={args.cache_rate} "
              f"fill_workers={max(args.cache_fill_workers, 1)}", flush=True)
    else:
        train_ds = data.Dataset(data=data_dicts_train, transform=train_transform)
        print("[cache] disabled", flush=True)
    # 独立的 generator 控制 shuffle：
    # 它不受模型侧 torch RNG 消耗的影响，因此 FM/DDPM 两个进程
    # 每个 epoch 的样本顺序完全一致；worker 内再用 _seed_worker 派生种子。
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              generator=gen, worker_init_fn=_seed_worker)
    val_ds = data.Dataset(data=data_dicts_val, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(args.workers // 2, 0), pin_memory=True)

    # 网络与作者一致：3D U-Net，输入 1 通道 CT，输出 3 类（背景/肝脏/肿瘤）。
    # 也没有用任何预训练权重，三组都从同一随机初始化开始。
    if args.model_name == "unet":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=3,
                     channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                     num_res_units=2)
    else:
        from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
        feature_size = {"tiny": 12, "small": 24, "base": 48}[args.swin_type]
        model = SwinUNETR_v2(in_channels=1, out_channels=3, img_size=(96, 96, 96),
                             feature_size=feature_size, patch_size=2,
                             depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                             window_size=[7, 7, 7])
    model = model.to(device)

    # 作者使用的损失：Dice + CrossEntropy，squared_pred 让大器官的
    # Dice 项更稳定；smooth_dr=1e-6 防止空 mask 除零。
    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                           smooth_nr=0, smooth_dr=1e-6)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    else:
        raise ValueError(args.optim_name)

    # 作者的学习率策略：前 100 epoch 线性 warmup，之后余弦退火到 0。
    # warmup 对 3D 分割很重要，直接从 2e-4 开始容易在前几个 epoch 发散。
    if args.lrschedule == "warmup_cosine":
        from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    # 验证用的滑窗：96³、overlap 0.75、高斯加权。
    # sw_batch_size=1 是因为完整体积很大，同时跑多个窗口容易爆显存。
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(sliding_window_inference, roi_size=inf_size,
                            sw_batch_size=1, predictor=model,
                            overlap=args.val_overlap, mode="gaussian")

    synth_models = None
    if args.arm != "real":
        synth_models = prepare_synth(args, device)

    # 断点续训：必须在 optimizer / scheduler 建好之后、训练循环之前。
    # 注意 resume 不恢复 DataLoader 的 epoch 计数，shuffle 顺序会从第 0 轮重来；
    # 对同一 arm 自身的持续训练没有影响，只是与「从未中断」的曲线不完全逐位一致。
    start_epoch, val_acc_max = 0, 0.0
    if args.resume:
        ckpt = (Path(args.logdir) / "model_final.pt"
                if args.resume == "auto" else Path(args.resume))
        if not ckpt.exists():
            if args.resume == "auto":
                # 链式启动时，某个 arm 可能是全新的（例如 ddpm），自动从头开始
                print(f"[resume] {ckpt} not found -> start from epoch 0", flush=True)
            else:
                raise SystemExit(f"[resume] checkpoint not found: {ckpt}")
        else:
            start_epoch, val_acc_max = resume_from_checkpoint(
                model, optimizer, scheduler, ckpt, device)
            if start_epoch >= args.max_epochs:
                print(f"[resume] already finished ({start_epoch} >= {args.max_epochs})",
                      flush=True)
                return

    # CacheDataset 的随机 crop 已经消费过 RNG；在训练前重置，
    # 保证 FM / DDPM 两组的几何 mask、注入决策、羽化 sigma 从同一状态开始。
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_training(model=model, train_loader=train_loader, val_loader=val_loader,
                 optimizer=optimizer, loss_func=dice_loss, args=args,
                 model_inferer=model_inferer, scheduler=scheduler,
                 val_shape_dict=val_shape_dict,
                 val_channel_names=["val_liver_dice", "val_tumor_dice"],
                 synth_models=synth_models,
                 start_epoch=start_epoch, val_acc_max=val_acc_max)


if __name__ == "__main__":
    main()
"""
依次训练三组作者口径的分割模型：real / fm / ddpm。

【三组的关系】
    real : 95 例真实肿瘤，不加载任何生成模型（baseline）
    fm   : 95 例真实 + 116 例健康 CT，健康 CT 以 50% 概率在线注入
           FM 合成肿瘤（early，8 步）
    ddpm : 同样 95 + 116，early 合成器换成 STEP2 训练的 DDPM（4 步）

    medium / large 肿瘤两组都用作者的 liver_noearly.pt，完全相同；
    因此 fm 与 ddpm 的唯一差别就是 early 合成器。

【为什么串行】
    - 三个 arm 都要读同一份磁盘缓存，串行可以避免缓存并发写；
    - 显存足够并行，但磁盘 I/O 会互相争抢，收益不确定；
    - 串行时每个 arm 的耗时更容易解释。

【用法】
  python flowmatching/run_author.py --arm all --dry-run
  python flowmatching/run_author.py --arm real --smoke
  python flowmatching/run_author.py --arm all --max-epochs 300 --val-every 100
  python flowmatching/run_author.py --arm all --max-epochs 2000 --val-every 200
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP3_DIR = HERE.parent
# 保证 STEP3 的 flowmatching 包优先于 STEP2 的同名包被解析
if str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

TRAIN = HERE / "train_seg_author.py"
RUNS = HERE / "runs_author"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["real", "fm", "ddpm", "all"])
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    # 验证很贵：每次要跑 23 例完整体积的滑窗推理（约 5–8 分钟）
    ap.add_argument("--val-every", type=int, default=200)
    # 仅 memory 模式使用；disk 模式下忽略
    ap.add_argument("--cache-rate", type=float, default=0.5)
    ap.add_argument("--cache-mode", default="disk", choices=["disk", "memory", "none"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--cache-fill-workers", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    # 仅 smoke 用：限制每个 epoch 的 iter 数和验证病例数
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--max-val-iters", type=int, default=0)
    # 'auto' 表示各 arm 自己的 <logdir>/model_final.pt
    ap.add_argument("--resume", default=None)
    ap.add_argument("--noamp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # 冒烟测试：只验证链路能跑通，不产出有效结果。
        # cache_mode=none 避免第一次就花十几分钟建缓存。
        args.max_epochs = 1
        args.val_every = 1
        args.cache_rate = 0.0
        args.cache_mode = "none"
        args.max_iters = 5
        args.max_val_iters = 2

    # all 时按 real -> fm -> ddpm 的顺序串行
    arms = ["real", "fm", "ddpm"] if args.arm == "all" else [args.arm]
    RUNS.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        # 注意：real 组也传 --synt_model ddpm，但 --arm real 时不会加载生成模型
        cmd = [sys.executable, "-u", str(TRAIN),
               "--arm", arm,
               "--synt_model", "fm" if arm == "fm" else "ddpm",
               "--max_epochs", str(args.max_epochs),
               "--batch_size", str(args.batch_size),
               "--optim_lr", str(args.lr),
               "--val_every", str(args.val_every),
               "--cache_rate", str(args.cache_rate),
               "--cache_mode", args.cache_mode,
               "--workers", str(args.workers),
               "--save_checkpoint",
               "--max_iters", str(args.max_iters),
               "--max_val_iters", str(args.max_val_iters),
               "--logdir", str(RUNS / arm)]
        if args.cache_dir:
            cmd += ["--cache_dir", args.cache_dir]
        cmd += ["--cache_fill_workers", str(args.cache_fill_workers)]
        if arm != "real":
            # FM 用 8 步；DDPM 的 4 步是模型内部固定，不会用到这个参数
            cmd += ["--fm_steps", "8"]
        if args.resume:
            cmd += ["--resume", args.resume]
        if args.noamp:
            cmd += ["--noamp"]

        print(f"[arm] {arm}\n  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        # 每个 arm 一个日志文件；训练入口自身也会打印到 stdout
        log = RUNS / f"{arm}_train.log"
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(STEP3_DIR), stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            # fail-fast：一个 arm 失败就停止，避免在错误状态下继续烧算力
            print(f"[error] {arm} exit {p.returncode}, see {log}", flush=True)
            sys.exit(p.returncode)
        print(f"[done] {arm}, log={log}", flush=True)


if __name__ == "__main__":
    main()
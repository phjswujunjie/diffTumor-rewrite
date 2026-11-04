# 警告：本目录是死代码

**`networks2/` 整个目录在全仓库中没有任何地方被引用。**

验证方式：

```bash
cd STEP3.SegmentationModel
grep -rn "networks2" --include=*.py . | grep -v __pycache__
# 无输出
```

真正被使用的是**同级的 `networks/` 目录**：

| 文件 | 谁在用 |
|---|---|
| `networks/unetr.py` | `main.py:28` |
| `networks/swin3d_unetr.py` | `main.py:29` |
| `networks/swin3d_unetrv2.py` | `main.py:30`、`validation.py:14` |

## 两个目录的关系

`networks2/` 是 `networks/` 的一份**近重复拷贝**，两份文件几乎一样：

| 文件 | 差异行数 |
|---|---|
| `mlp.py` | 0（完全相同） |
| `selfattention.py` | 0 |
| `patchembedding.py` | 0 |
| `unetr_block.py` | 0 |
| `vit.py` | 4 |
| `transformerblock.py` | 24 |
| `unetr.py` | 26 |

看起来是某个阶段重构时的中间产物（改到一半的版本），最终 `networks/` 胜出，
`networks2/` 被遗留下来没有删除。

## 读代码时的建议

**直接跳过整个目录。**

如果你在找某个网络结构的实现（例如 SwinUNETR 或 UNETR），
去 `networks/` 找，不要在这里找 —— 这里的内容可能和实际跑的不是同一版。

注意：`swin3d_unetr.py` 只存在于 `networks/`，`networks2/` 里没有，
也可以作为"`networks/` 才是正式的"的一个佐证。

# Global Branch Level Innovation Models

## 核心创新点

### 谱图流形变换 (Spectral Manifold Transform)

替代原始 FFT 的固定频率基，使用**自适应相似度图**和 **Chebyshev 多项式近似**。

### 数学原理

```
1. 构建相似度图: A_ij = exp(-||x_i - x_j||² / T)
2. 归一化拉普拉斯: L = I - D^{-1/2} A D^{-1/2}
3. Chebyshev 多项式滤波: g(L) ≈ Σ_{k=0}^{K-1} c_k · T_k(L)
4. 三分支融合: filtered = filtered_low + filtered_mid + filtered_high
5. 超球面投影: x → x / ||x||
```

### 与原 FFT 的区别

| 方面 | 原 FFT | 新 SMT |
|------|--------|--------|
| 邻居关系 | 固定循环邻接 | **自适应相似度** |
| 频率基 | 固定 exp 基 | **可学习的图基** |
| 论文贡献 | 无 | **理论+方法双贡献** |

## 文件结构

```
global_level_innovation/
├── __init__.py           # 包入口
├── bir_lstm.py           # 核心创新：SpectralManifoldBlock
├── global_branch.py      # GlobalBranch 模型
├── object_encoder.py     # 对象编码器
├── language_encoder.py   # 语言编码器
├── pointnet2.py          # 点云特征提取
├── msg_encoder.py        # 消息编码器
└── mamba_model.py        # Mamba 模型
```

## 使用方法

```python
from models.global_level_innovation import GlobalBranch

model = GlobalBranch(known_classes, known_colors, args)
```

## 训练命令

```bash
python -m training.train_branches \
  --batch_size 64 \
  --coarse_embed_dim 256 \
  --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features class color position num \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 32 \
  --learning_rate 0.0001 \
  --ranking_loss CCL \
  --hungging_model t5-large \
  --folder_name exp_coarse_staged \
  --branch global \
  --epochs_per_branch 32 \
  --cpus 0
```

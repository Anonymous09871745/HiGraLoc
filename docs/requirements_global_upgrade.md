# MNCL Global分支升级需求文档

## 项目背景

MNCL是一个三分支对齐的神经网络项目，在coarse阶段包含三个对齐分支：
- **Global Branch（全局特征分支）** - 提取全局语义特征
- **Instance/Object Branch（实例特征分支）** - 提取实例级特征
- **Relation Branch（关系特征分支）** - 提取关系级特征

## 需求目标

将Global分支升级为包含数学创新点（谱图流形变换 Spectral Manifold Transform）的新版本，同时**保持Instance和Relation分支完全不变**。

## 技术方案

### 创新点说明

原Global分支使用FFT（快速傅里叶变换）进行频域特征提取，新版本使用**谱图流形变换（Spectral Manifold Transform）**替代：

1. **构建自相似度图**：基于高斯核计算样本间相似度
2. **归一化拉普拉斯**：使用图拉普拉斯算子描述流形结构
3. **Chebyshev多项式滤波**：数值稳定的谱域滤波（替代显式特征分解）
4. **超球面投影**：保持特征的几何性质

### 替换文件清单

| 文件路径 | 操作 | 说明 |
|---------|------|------|
| `models/bir_lstm.py` | **替换** | 核心创新文件，新增`SpectralManifoldBlock`类 |
| `models/pointnet2.py` | **新增** | PointNet++实现（如果原项目不存在） |

### 保持不变清单

| 文件路径 | 说明 |
|---------|------|
| `models/branches/global_branch.py` | Global分支封装（保持引用接口不变） |
| `models/branches/object_branch.py` | Instance分支（**完全不变**） |
| `models/branches/relation_branch.py` | Relation分支（**完全不变**） |
| `models/branches/__init__.py` | 分支模块入口 |
| `models/object_encoder.py` | 对象编码器 |
| `models/language_encoder.py` | 语言编码器 |
| `models/mamba_model.py` | Mamba模型 |
| `models/msg_encoder.py` | 消息编码器 |
| 其他所有文件 | 项目原有结构 |

## 兼容性要求

1. 替换后的代码必须与原有训练/评估流程完全兼容
2. `ConvBlock`类的接口必须与原来保持一致
3. 新增的`SpectralManifoldBlock`作为独立模块，不影响原有功能
4. 分支加载和权重管理逻辑不变

## 验证标准

- [x] ✅ Global分支能正常加载和训练（`ConvBlock` + `SpectralManifoldBlock`）
- [x] ✅ Instance和Relation分支保持不变
- [x] ✅ 代码语法检查通过
- [x] ✅ 模块导入测试通过

## 实施记录

| 操作 | 文件 | 时间 |
|------|------|------|
| 替换 | `models/bir_lstm.py` | 已完成 |
| 新增 | `models/pointnet2.py` | 已完成 |
| 新增 | `models/global_level_innovation/` | 已完成 |
| 保持 | `models/branches/object_branch.py` | 未修改 |
| 保持 | `models/branches/relation_branch.py` | 未修改 |
| 保持 | `models/branches/global_branch.py` | 未修改 |

## 创新模块使用方式

```python
from models.bir_lstm import ConvBlock, SpectralManifoldBlock

# ConvBlock 中已集成 SpectralManifoldBlock
model = ConvBlock()
# 内部自动使用谱图流形变换替代FFT
```

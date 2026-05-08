# 分支权重消融实验（Ablation Study）需求文档

## 1. 背景与目标

当前三分支融合模型的最优权重配置为：

| 分支 | 参数名 | 最优值 |
|------|--------|--------|
| Global 分支 | `weight_global` | 0.3 |
| Object 分支 | `weight_object` | 1.0 |
| Relation 分支 | `weight_relation` | 0.8 |

为探究各分支对最终性能的贡献程度，设计 **消融实验（Ablation Study）**，系统性地将各分支权重置零（相当于"关闭"该分支），观察性能变化。

## 2. 实验设计

### 2.1 实验组（共 7 组）

以最优配置为 Baseline（组 0），其余 6 组覆盖所有"任一分支权重为 0"的组合，以及 1 组"三个分支全为 0"的极限定界。

| 组号 | 实验名称 | weight_global | weight_object | weight_relation | 说明 |
|------|----------|:---:|:---:|:---:|------|
| 0 | **Baseline（最优）** | 0.3 | 1.0 | 0.8 | 基准参照 |
| 1 | Ablate-Global | **0.0** | 1.0 | 0.8 | 关闭 Global 分支 |
| 2 | Ablate-Object | 0.3 | **0.0** | 0.8 | 关闭 Object 分支 |
| 3 | Ablate-Relation | 0.3 | 1.0 | **0.0** | 关闭 Relation 分支 |
| 4 | Ablate-Global-Object | **0.0** | **0.0** | 0.8 | 关闭 Global + Object |
| 5 | Ablate-Global-Relation | **0.0** | 1.0 | **0.0** | 关闭 Global + Relation |
| 6 | Ablate-Object-Relation | 0.3 | **0.0** | **0.0** | 关闭 Object + Relation |
| 7 | All-Zero | **0.0** | **0.0** | **0.0** | 三分支全关（极限下限） |

### 2.2 权重组合逻辑（共 8 组 = 1 个基准 + 7 种消融）

所有组合由 `weight_global`、`weight_object`、`weight_relation` 三个维度取 {最优值, 0} 的排列构成，其中：
- 每个维度固定取最优值或置 0；
- 三者同时为最优值 → Baseline；
- 其余 7 种组合 → 消融实验。

## 3. 评估指标

与原始实验保持一致，使用以下指标：

| 指标 | 说明 |
|------|------|
| `hit@1` | Top-1 命中率（主要指标） |
| `hit@3` | Top-3 命中率 |
| `hit@5` | Top-5 命中率 |
| `close@1` | Top-1 接近率 |
| `close@3/5/10` | 多档接近率 |

## 4. 实验脚本修改要求

在 `pipeline_three_branch_with_fine.py` 中，将权重参数化，支持通过命令行或配置注入 7 组消融值。建议修改点：

```python
# 待修改位置（示例）
weight_global  = args.weight_global   # 或从配置文件读取
weight_object  = args.weight_object
weight_relation = args.weight_relation
```

## 5. 实验预期与分析方向

| 分析维度 | 预期观察 |
|---------|---------|
| 单分支关闭的贡献度 | hit@1 下降越多 → 该分支越重要 |
| 双分支组合关闭的交互效应 | 1+1 > 2（协同增益）或 1+1 ≈ 2（独立贡献）|
| 全关闭基准 | 确定无融合时的性能下限 |
| 各指标变化趋势 | hit@1 敏感度高，但 close@k 是否同步变化？|

## 6. 输出要求

每组实验独立保存结果，命名格式：

```
weight_search_test_step0.1_ablate_<name>_<timestamp>.csv
```

最终汇总表格（建议）：

| 实验组 | hit@1 | hit@3 | hit@5 | close@1 | Δ hit@1 vs Baseline |
|-------|-------|-------|-------|--------|---------------------|
| Baseline | xxx | xxx | xxx | xxx | — |
| Ablate-Global | xxx | xxx | xxx | xxx | Δ |
| ... | ... | ... | ... | ... | ... |

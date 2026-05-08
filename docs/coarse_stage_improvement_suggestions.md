# Coarse 阶段改进意见（围绕准确度与 Loss）

本文档基于当前实现，从 **准确度与 loss 强相关** 的角度，给出 coarse 阶段可落地的改进建议。每条都尽量对应「问题 → 对 loss/acc 的影响 → 改法」。

---

## 一、Loss 设计与权重

### 1.1 五项损失固定等权 (1.0)

**现状**  
`loss = 1.0*L_global + 1.0*L_object + 1.0*L_relation + 1.0*L_anchor_self + 1.0*L_global_self`，五项标量直接相加。

**问题**  
- 不同分支的梯度量纲、收敛速度不一致，容易某一项主导或某一项几乎不学。  
- 自监督两项（anchor–anchor、F_Global–F_Global）与检索目标并非直接对应，等权可能压制真正的 text–cell 对齐。

**对准确度的影响**  
检索主要看 Global + Object + Relation 三路；若自监督过强或某一分支过强，val acc 会震荡或上限变低。

**改进建议**  
1. **可学习/可调权重**  
   - 引入可训练标量或固定超参：  
     `loss = w_g*L_global + w_o*L_object + w_r*L_relation + w_a*L_anchor_self + w_s*L_global_self`  
   - 先用网格搜索或验证集调 `w_*`（例如先设 `w_a=w_s=0.3`，再调 w_g/w_o/w_r），再考虑用 Uncertainty Weighting（多任务学习）或 GradNorm 自动平衡。  
2. **阶段性权重**  
   - 前期加大 Global（收敛快），中后期加大 Object/Relation（细粒度对齐），例如按 epoch 线性调整 w_g 与 w_o、w_r。  
3. **与 eval 一致**  
   - 若验证发现某分支对 top-k 贡献更大，可适当提高该分支的 loss 权重，使训练目标更贴近「最终打分」。

---

### 1.2 自监督项的必要性与形式

**现状**  
`criterion(anchor, anchor)` 与 `criterion(F_Global, F_Global)` 在 batch 内做「文本–文本」「场景–场景」的对比，注释称带来约 0.004 提升。

**问题**  
- 正样本是自身（对角线），负样本是 batch 内其它样本；若 batch 内有多句相似描述或多个相似 cell，会被当成负样本推远，与检索语义略有冲突。  
- 两项都使用与 Global 相同的 CCL，梯度会叠加在 language encoder 和 scene encoder 上，可能过度强调「batch 内区分」而非「跨 batch 的泛化排序」。

**改进建议**  
1. **降权或仅保留一项**  
   - 例如 `w_a=0.3`, `w_s=0.3`，或只保留一项（如仅 F_Global–F_Global）做正则。  
2. **换更温和的正则**  
   - 用 variance–covariance 正则（如 Barlow Twins 风格）或只做 alignment 不做 uniformity，减轻对「batch 内负样本」的依赖。  
3. **做消融**  
   - 分别跑「无自监督 / 仅 anchor / 仅 F_Global / 两者」对比 val top-1/top-3/top-5 与 loss 曲线，确认对准确度的真实贡献。

---

## 二、Object / Relation 的 score 与 dist

### 2.1 Object/Relation score 的聚合方式 (max → mean)

**现状**  
对 (query_i, cell_j)，Object 相似度张量 `cc[i,j]` 形状为 `[num_mentioned, object_size]`，先对 object_size 维取 **max**，再对 num_mentioned 维取 **mean**，得到标量 score。

**问题**  
- **max** 只保留「一个最像的物体对」，其余信息丢弃，容易受噪声或单一强匹配主导。  
- **mean** 对所有 hint 一视同仁，而有的 hint 对定位更关键（如「红色车」vs「旁边」），与检索重要性不一致。

**对准确度的影响**  
Object/Relation 分支的表达能力未完全用上，检索时同一分支的区分度可能不足，尤其当正确 cell 与错误 cell 在「最强单点匹配」上接近时。

**改进建议**  
1. **Top-k mean 替代全局 max**  
   - 对 object_size 维取 top-k（如 k=3）再 mean，既抑制噪声又保留多对匹配信息。  
2. **Attention 聚合**  
   - 用 query 的 Global 或 Object 特征做 attention weight，对 num_mentioned 维加权聚合，让重要 hint 权重大。  
3. **可学习聚合**  
   - 用小型 MLP 或一层 attention 把 `cc[i,j]` 压成标量，用验证集调参或端到端学习。  
4. **训练/推理一致**  
   - 上述改动需在 `train_epoch` 与 `eval_epoch` 中同步，否则会有 train–eval 鸿沟。

---

### 2.2 dist_Object / dist_Relation 的数值与语义

**现状**  
- `dist = || mean_F_Object_P - mean_F_Object_T ||`（按样本对），再 `dist /= dist.max()`，然后 `(1-dist+1e-6)^(1/alpha)`，最后 clip 到 `[thresh=0.8, 1]`。  
- 该 dist 在 CCL_input_score 里作为「困难负样本」权重：越大表示 query–cell 在物体/关系空间越近，负样本越难，损失权重越大。

**问题**  
1. **数值**：若某 batch 内所有对都较远，`dist.max()` 仍可能较小，归一化后整体偏大；若 `dist.max()` 非常小（几乎为 0），除法不稳定。  
2. **语义**：dist 基于 **mean 池化** 的 Object/Relation 特征，与上面「max→mean」的 score 并非同一套表示，可能和「哪些对是困难负样本」不完全一致。  
3. **与 score 尺度**：Object/Relation 的 score 未做归一化就送入 CCL_input_score，而 Global 分支在 CCL 内部会先 L2 归一化再算相似度；三路尺度不一致可能影响 temperature 与 hardest-negative 选择。

**改进建议**  
1. **数值稳定**  
   - `dist = dist / (dist.max() + 1e-8)`，或改用 `dist = dist / (dist.max(dim=..., keepdim=True) + eps)` 避免 0。  
   - 对 `(1-dist+1e-6)` 再 clip 下界（如 1e-6），避免 alpha 较大时接近 0 导致梯度爆炸。  
2. **dist 与 score 一致**  
   - 用与 score 相同的聚合方式（例如同一套 mean 或同一套 attention）得到「代表向量」再算 dist，使「难负样本」定义与打分一致。  
3. **可选：用 score 本身做权重**  
   - 对负样本位置 (i,j)，用 `score[i,j]` 或 `relation_score[i,j]` 的单调变换作为权重（高分负样本权重大），替代或补充当前 dist，实现与 CCL 一致的「困难负样本」定义。

---

## 三、CCL 与困难负样本

### 3.1 CCL 的 ratio 未暴露

**现状**  
`CCL` / `CCL_input_score` 构造 hardest-negative mask 时，`num = n-1 if ratio<=0 or ratio>=1 else int(ratio*n)`；当前从 args 未传入 ratio，默认 `ratio=0`，即 `num = n-1`，即 **所有非对角线位置都当作 hardest 参与损失**。

**问题**  
- 等价于「所有负样本都加权」，与「只选最难的几个」的常见做法不同；若 batch 较大，梯度会被大量简单负样本稀释。  
- 无法通过命令行做「只取 top-k% 最难负样本」的消融，不利于调参。

**改进建议**  
1. **在 args 中增加 `--ccl_ratio`**  
   - 例如默认 0.5，表示每行只取概率最高的 50% 负样本参与 loss；ratio=0 保持当前「全部」行为。  
2. **观察 acc–loss 关系**  
   - 对比 ratio=0 / 0.3 / 0.5 / 1.0 下 val acc 与 train loss：若 ratio 适中时 acc 更高，说明「精选困难负样本」有利于准确度。

---

### 3.2 Temperature 与 Alpha

**现状**  
- `temperature`（args，默认 0.1）控制 CCL 中 softmax 的锐度；  
- `alpha`（args，默认 2）控制 dist 的变换 `(1-dist)^(1/alpha)`，alpha 越大曲线越平。

**问题**  
- Temperature 过小：分布过尖，梯度集中在极少数样本，易不稳定；过大：分布过平滑，区分度下降。  
- Alpha 与 dist 的 clip 共同决定「困难负样本」的权重曲线；若 alpha 与 thresh 固定，可能不是当前数据与 batch 的最优组合。

**改进建议**  
1. **Temperature**  
   - 在 [0.05, 0.2] 做小步长搜索，画 val top-1 vs temperature 曲线；通常 0.07–0.1 较稳。  
2. **Alpha**  
   - 尝试 1, 2, 4；alpha 越大 dist 权重越平滑，困难/简单负样本差异越小。  
3. **联合调**  
   - 固定 loss 权重与 ratio，只调 temperature 和 alpha，看 loss 曲线是否更平滑、val acc 是否提升。

---

## 四、训练与推理一致性

### 4.1 打分方式一致

**现状**  
- **训练**：三路分别算 loss，没有「三路融合成一个标量再优化」的显式项。  
- **推理**：`scores = 1.0*z_score(scores_1) + 1.0*z_score(scores_2) + 1.0*z_score(scores_3)`，即三路 **z_score 归一化后等权相加**。

**问题**  
- 训练目标是最小化三路各自的对比损失，而不是最小化「融合后的排序损失」；若某一路在验证集上方差很大或尺度偏小，z_score 后仍等权可能不是最优。  
- 验证集上若某分支长期优于另外两路，说明分支权重或 loss 权重可再调。

**改进建议**  
1. **训练时加「融合排序」辅助损失**  
   - 在每个 batch 内，用与 eval 相同的融合方式（如当前 1:1:1 z_score 相加）得到每个 (query, cell) 的融合分，再对该矩阵做 listwise / pairwise 排序损失（如 ListNet、ListMLE 或 pairwise hinge），使训练目标直接对齐「最终排序」。权重可设为 0.3–0.5，与现有 CCL 一起反传。  
2. **推理时可学习分支权重**  
   - 保留 z_score，但用可学习标量或小型 MLP 得到 `scores = w1*s1 + w2*s2 + w3*s3`，在验证集上调参或通过少量标注学习，使 top-k 准确度更高。

---

### 4.2 Eval 时 z_score 的统计量

**现状**  
`z_score(scores_1)` 等是对 **单个 query** 的 scores_1 向量（对所有 cell）做标准化；即每个 query 独立减均值、除标准差。

**问题**  
- 不同 query 的难度不同，有的 query 三路分数整体偏高/偏低，z_score 能缓解尺度差异，但若某个 query 的某一路方差极小（几乎常数），z_score 会放大噪声。  
- 若验证集很小，单个 query 的统计量不稳定。

**改进建议**  
1. **方差裁剪**  
   - `std = max(std, 1e-6)` 或类似，避免除零或极大倍数。  
2. **可选：全局 z_score**  
   - 用全验证集上该路的均值/方差做标准化（需先跑一遍得到统计量），再对比「per-query z_score」与「global z_score」的 val acc，选更稳的一种。

---

## 五、模型与数据侧（与 loss/acc 直接相关）

### 5.1 ObjectMsgEncoder 中 relation_level_masks 的返回值

**现状**  
`models/msg_encoder.py` 中 `ObjectMsgEncoder.forward` 最后一行：  
`return object_level_features, relation_level_features, index_list, object_level_masks, object_level_masks`  
即 **relation_level_masks 被错误地返回为 object_level_masks**。

**问题**  
- 在 coarse 训练与推理中，Relation 分支用的 mask 实际是 object_level_masks；若两者形状相同但语义不同（例如 relation 是 N×N 聚合后的 N×1），会导致 relation 的 padding 未被正确 mask，score 和 loss 被错误位置影响。

**对准确度的影响**  
Relation 分支可能长期在学「错误位置」或噪声，val 上 Relation 单独或融合 acc 可能偏弱。

**改进建议**  
1. **确认设计**  
   - 若 relation 维度和 object 一致且都按「每个物体一个向量」，需确认是否应返回 `relation_level_masks`（与 relation_level_features 对应的 mask）。  
2. **修正返回**  
   - 若确实有独立的 relation_level_masks（例如 shape 与 object_level_masks 不同），则改为返回该 mask；若当前实现意图就是「relation 与 object 共用同一 mask」，则保留现状但在注释中写清，并检查 relation_level_features 的 padding 是否已正确置 0。

---

### 5.2 Batch 内负样本与数据构造

**现状**  
负样本完全来自同一 batch 内其它 (query, cell) 对；batch 随机打乱，负样本随机。

**问题**  
- 若 batch 内碰巧有多个「相似描述」或「空间相近的 cell」，会变成困难负样本，有利于训练；反之若负样本都「明显不对」，梯度信号弱。  
- 没有显式的「难负样本挖掘」（如跨 batch 的 queue、或按距离采样的 hard negatives）。

**改进建议**  
1. **适当增大 batch_size**  
   - 在显存允许下增大 batch，负样本更多、更多样，对比学习更稳（需配合 learning rate / warmup）。  
2. **可选：Memory bank / queue**  
   - 维护一个 F_Global（或三路）的 queue，用历史 batch 的编码作为额外负样本，增加困难负样本比例（实现成本较高，可作为后续进阶）。  
3. **数据层面**  
   - 若数据集支持，可构造「同 scene 不同 cell」或「同描述不同 cell」的困难对，作为额外正/负样本或加权的 pair。

---

## 六、实现与调参优先级建议

**高优先级（先做，对 acc/loss 影响大）**  
1. 修正/确认 **ObjectMsgEncoder** 的 relation_level_masks 返回值。  
2. 为 **五项 loss 引入可调权重**（至少用超参），并做「无自监督 / 减权自监督」消融。  
3. **dist 的数值稳定**（除以 max+eps、clip）与 **CCL ratio 暴露**（如 `--ccl_ratio 0.5`）并做小规模搜索。

**中优先级（在验证集上明显再推广）**  
4. Object/Relation 的 **score 聚合**（top-k mean 或 attention）在 train 与 eval 中一致替换并对比 acc。  
5. **Temperature / alpha** 的网格搜索或贝叶斯搜索。  
6. **训练时加融合排序损失**（listwise/pairwise），与现有 CCL 联合训练。

**低优先级（锦上添花）**  
7. 自监督项改为更温和的正则或去掉。  
8. Eval 时 **可学习分支权重** 或全局 z_score 对比。  
9. 更大 batch / Memory bank 难负样本。

---

## 七、简要对照表（问题 → 影响 → 改法）

| 问题 | 对准确度/loss 的影响 | 建议改法 |
|------|----------------------|----------|
| 五项 loss 等权 | 某分支主导或自监督过强，val 震荡/上限低 | 可调/可学习权重；自监督降权或消融 |
| Object/Relation score 仅 max→mean | 分支区分度不足，易被单点噪声主导 | Top-k mean 或 attention 聚合，train/eval 一致 |
| dist 归一化与 clip 不稳定 | 梯度异常或权重失真 | max+eps、clip 下界；可选与 score 一致定义 dist |
| CCL ratio 固定为「全部」 | 简单负样本稀释梯度 | args 暴露 ratio，尝试 0.3–0.5 |
| 训练优化三路 loss、推理用融合分 | 训练目标与最终排序不完全一致 | 加融合排序损失；推理可学习分支权重 |
| relation_level_masks 返回错误 | Relation 分支学错/学弱 | 确认设计并修正返回或注释 |
| Temperature/alpha 未系统调 | 对比学习效果未最优 | 小网格搜索 temperature 与 alpha |

按上述顺序逐步改动并记录每次的 **val top-1/top-3/top-5 与 train/val loss**，即可量化每个改进对「准确度与 loss」的贡献。

# MNCL Coarse 阶段三分支损失函数 — 极度详细分析

本文档基于 `models/` 与 `training/` 目录下的实现，对 MNCL coarse 阶段的**三分支（Global / Object / Relation）**表示与损失函数做逐行级分析。

---

## 一、整体架构概览

Coarse 阶段的目标是：给定一句自然语言描述（多 hint 句子），在离散的 cell 集合中做检索，选出最可能包含目标位置的 cell。  
实现方式是多层次对齐：**文本侧**与**场景侧**各产出三种表示（Global / Object / Relation），再在三个层次上分别算相似度并参与损失。

- **分支 1 - Global**：整句描述 ↔ 整个 cell 的全局表示  
- **分支 2 - Object**：描述中的物体级 hint ↔ cell 内物体级表示  
- **分支 3 - Relation**：描述中的关系级 hint ↔ cell 内关系级表示  

训练时对这三路都施加对比损失（当前默认 CCL），并加上文本/场景的“自监督”项。

---

## 二、三分支特征的数据流（models）

### 2.1 文本侧：`LanguageEncoder` + `LanguageMsgEncoder`

**文件**: `models/language_encoder.py`  
**入口**: `CellRetrievalNetwork.encode_text(descriptions)` 内部调用 `self.language_encoder(descriptions)`。

**步骤简述**:

1. 每个 description 按句子切分 → `split_union_sentences`，再按 batch 组织成 `[总句子数, ...]`。
2. T5 encoder 得到 `description_encodings`（token 级）。
3. 经 `intra_module`（TransformerEncoderLayer）+ max 得到句子级向量，再经 `inter_mlp` 映射到 `embedding_dim`。
4. LSTM 聚合后得到 `description_encodings`: `[B, num_sentence, D]`。
5. **Multi-level Scene Graph**：`LanguageMsgEncoder(self.MSG)` 输入 `description_encodings`，输出：
   - `F_Object_T`: 物体级文本表示，形状 `[B, num_mentioned, D]`（与 `args.num_mentioned` 对齐）。
   - `F_Relation_T`: 关系级文本表示，形状同上。
6. 文本再经过 `mamba_layer_inter` + `inter_module` + **max(dim=0)** 得到全局句向量：
   - **anchor** = 该全局向量，形状 `[B, D]`，并做 L2 归一化。

因此文本侧三个输出为：

- **anchor** `[B, D]`: Global 分支文本表示；
- **F_Object_T** `[B, num_mentioned, D]`: Object 分支文本表示；
- **F_Relation_T** `[B, num_mentioned, D]`: Relation 分支文本表示。

### 2.2 场景侧：`ObjectEncoder` + `ObjectMsgEncoder` + `ConvBlock`

**文件**: `models/object_encoder.py`, `models/msg_encoder.py`, `models/cell_retrieval.py`, `models/bir_lstm.py`  
**入口**: `CellRetrievalNetwork.encode_objects(objects, object_points)`。

**步骤简述**:

1. **ObjectEncoder**：对每个 cell 内的物体做点云 + 类别/颜色/位置等编码，得到 `embeddings`（每个物体一维向量）。
2. **ObjectMsgEncoder (MSG)**：输入 `objects` 与 `object_embedding`，按 cell 组织，并利用物体中心坐标构造关系（差值向量），经多层 message passing 得到：
   - **F_Object_P**: 物体级场景表示 `[B, object_size, D]`；
   - **F_Relation_P**: 关系级场景表示 `[B, object_size, D]`（对关系做 ESA 聚合到每个物体）；
   - **object_level_masks** / **relation_level_masks**：有效物体/关系的 mask，用于后续算分时屏蔽 padding。
3. **ConvBlock (self.layers)**：  
   - 将每个 cell 的物体嵌入按 `index_list` 填成 `F_Global` 的初始 `[B, object_size, D]`；
   - 经 ConvBlock（1D 卷积、FFT、Attention、GRU、Transformer 等）得到 `F_Global`；
   - 再 **max(dim=0)** 得到每个 cell 的全局向量，形状 `[B, D]`，并 L2 归一化。

因此场景侧三个输出为：

- **F_Global** `[B, D]`: Global 分支场景表示；
- **F_Object_P** `[B, object_size, D]`: Object 分支场景表示；
- **F_Relation_P** `[B, object_size, D]`: Relation 分支场景表示；

以及 **p_object_level_masks**、**p_relation_level_masks**，在训练中用于 Object/Relation 相似度矩阵的 mask。

---

## 三、训练时三分支相似度与 dist 的计算（training/coarse.py）

所有计算在 `train_epoch()` 内、对当前 batch 的 `anchor, F_Object_T, F_Relation_T` 与 `F_Global, F_Object_P, F_Relation_P` 进行。  
`batch_size = len(anchor)`。

### 3.1 Global 分支

- **表示**：文本 `anchor` `[B, D]`，场景 `F_Global` `[B, D]`（均已 L2 归一化）。
- **相似度**：不在此处显式建矩阵；直接送入 `criterion(anchor, F_Global)`。  
  在 CCL 内部会做 `scores = torch.mm(im, s.T)`，即 `[B,B]` 的相似度矩阵（cosine 等价于点积）。

即：Global 分支的“分数”就是 batch 内所有 (text_i, scene_j) 的成对相似度，用于对比学习。

### 3.2 Object 分支：score 与 dist_Object

**相似度矩阵**（Object-level score）：

- `F_Object_T`: `[B, num_mentioned, D]`  
- `F_Object_P`: `[B, object_size, D]`  
- 目标：得到每个 (batch_i, batch_j) 的标量相似度。

实现（约 52–61 行）：

```python
aa_expanded = F_Object_T.unsqueeze(1).expand(batch_size, batch_size, -1, -1)   # [B, B, num_mentioned, D]
bb_transposed = F_Object_P.transpose(1, 2)                                      # [B, D, object_size]
cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)           # [B, B, num_mentioned, object_size]
```

即对每对 (query_i, cell_j)，得到 “第 i 句的每个 hint 物体” 与 “第 j 个 cell 的每个物体” 的相似度矩阵 `[num_mentioned, object_size]`。  
然后用 `p_object_level_masks` 把 padding 位置置为 -100，再：

- `cc.max(dim=-1)[0]` → 每个 hint 对 cell j 的最大物体相似度，形状 `[B, B, num_mentioned]`；
- `.mean(dim=-1)` → 对 hint 维求平均，得到 **score** `[B, B]`。

**dist_Object**（用于 CCL_input_score 的权重）：

- 对 Object 特征在“物体维”上取平均并 L2 归一化：
  - `mean_F_Object_P`: `[B, D]`；`mean_F_Object_T`: `[B, D]`。
- 成对 L2 距离：
  - `dist_Object = || mean_F_Object_P.unsqueeze(1) - mean_F_Object_T.unsqueeze(0) ||` → `[B, B]`。
- 归一化与变换：
  - `dist_Object = dist_Object / dist_Object.max()`
  - `dist_Object = (1 - dist_Object + 1e-6)^(1/args.alpha)`，再 clip 到 `[thresh=0.8, 1]`。

这样，**query 与 cell 在物体级表示越接近，dist_Object 越大**，在 CCL_input_score 里会作为权重放大该负样本的惩罚（hard negative 加权）。

### 3.3 Relation 分支：relation_score 与 dist_Relation

与 Object 完全对称，只是把 `F_Object_T/P` 和 object-level mask 换成 `F_Relation_T/P` 和 relation-level mask：

- **relation_score** `[B, B]`：同上，matmul → mask → max(dim=-1) → mean(dim=-1)。
- **dist_Relation**：用 `mean_F_Relation_P` 与 `mean_F_Relation_T` 做同样步骤得到 `[B, B]`，再归一化、`(1-dist)^(1/alpha)`、clip。

---

## 四、损失函数定义（training/losses.py）

Coarse 阶段用到的两类损失：

1. **criterion**：输入两组向量 `im, s`（或仅用其构造的 score），做**标准 CCL**（带内部归一化与 temperature）。
2. **criterion_2**：**CCL_input_score**，输入已算好的 **score 矩阵** 和 **dist 权重矩阵**，用于 Object/Relation 两路。

### 4.1 CCL（用于 Global 与自监督项）

**类**: `CCL`（约 334–392 行）

- **输入**: `im` `[B, D]`, `s` `[B, D]`（通常已 L2 归一化）。
- **内部**:
  - `scores = torch.mm(im, s.T)` → `[B, B]`；
  - temperature scaling: `scores = (scores / tau).exp()`；
  - 行/列归一化得到 i2t、t2i 的“概率”分布；
  - **Hard negative 选择**：按 `ratio` 在每行选概率最高的若干负样本位置，构造 mask；
  - **距离加权**：用 `im` 与 `s` 的成对 L2 距离构造 `dist`，并 `(1-dist)^(1/alpha)`、clip 到 `[thresh, 1]`，与 mask 一起作用在 criterion 上；
  - **criterion** 可选 `log` / `tan` / `abs` / `exp` / `gce`，默认对“错误概率”做惩罚（如 `-log(1-x)`）；
- **输出**: 标量，对 i2t 与 t2i 两方向加权求和再平均。

用于：`criterion(anchor, F_Global)`、`criterion(anchor, anchor)`、`criterion(F_Global, F_Global)`。

### 4.2 CCL_input_score（用于 Object / Relation）

**类**: `CCL_input_score`（约 395–424 行）

- **输入**:  
  - `score`: 已算好的相似度矩阵 `[B, B]`（如上面的 `score` 或 `relation_score`）；  
  - `dist`: 与 score 同形的权重矩阵 `[B, B]`（如 `dist_Object` 或 `dist_Relation`）。
- **内部**:
  - 不再做 `im/s` 的归一化，直接用 `score` 做 temperature 归一化得到 i2t、t2i；
  - 同样做 hardest-negative 的 mask；
  - 用 `dist` 对每个位置的损失加权：`(criterion(i2t) * dist * mask).sum(1).mean() + (criterion(t2i) * dist.T * mask).sum(1).mean()`。
- **输出**: 标量。

用于：Object 分支 `criterion_2(score, dist_Object)`，Relation 分支 `criterion_2(relation_score, dist_Relation)`。

---

## 五、总损失组合（training/coarse.py，约 104–106 行）

当 `args.ranking_loss == "CCL"` 时（见 412–414 行），`criterion = CCL(...)`，`criterion_2 = CCL_input_score(...)`。总损失为：

```text
loss =
  1.0 * criterion(anchor, F_Global)                    # Global: 文本–场景全局对齐
+ 1.0 * criterion_2(score, dist_Object)                # Object: 物体级对齐，按 dist_Object 加权
+ 1.0 * criterion_2(relation_score, dist_Relation)     # Relation: 关系级对齐，按 dist_Relation 加权
+ 1.0 * criterion(anchor, anchor)                     # 文本自监督：batch 内文本–文本
+ 1.0 * criterion(F_Global, F_Global)                 # 场景自监督：batch 内场景–场景
```

五项均为标量，权重均为 1.0。

- **前三项**：三分支对比损失，拉近匹配的 (query, cell)，推远不匹配的。
- **后两项**：在 batch 内对文本和场景分别做“自对比”，起到正则、稳定表示的作用（注释中提到与某 checkpoint 配合带来约 0.004 的普遍提升）。

---

## 六、推理时的三分支融合（eval_epoch）

**文件**: `training/coarse.py`，`eval_epoch()` 内约 224–262 行。

对每个 query_idx：

1. **Global**: `scores_1 = cell_encodings_global_level @ text_encodings[query_idx]`，即一维向量与全库 cell 的全局向量点积。
2. **Object**: 用 `text_encodings_object_level[query_idx]` 与所有 cell 的 object-level 编码做与训练时相同的 matmul + mask + max + mean，得到 `scores_2`。
3. **Relation**: 同理得到 `scores_3`。

最终检索分数为三者 z-score 归一化后等权相加：

```python
scores = 1.0 * z_score(scores_1) + 1.0 * z_score(scores_2) + 1.0 * z_score(scores_3)
```

排序时按 `scores` 降序取 top-k。

---

## 七、关键超参与配置（training/args.py）

- **ranking_loss**: `"pairwise"` | `"hardest"` | `"triplet"` | `"contrastive"` | `"CCL"`（当前分析以 CCL 为准）。
- **temperature**: CCL 温度，默认 0.1。
- **alpha**: 用于 `(1-dist)^(1/alpha)`，默认 2。
- **num_mentioned** / **object_size**: 文本侧 hint 数与场景侧最大物体数，影响 Object/Relation 的矩阵维数。
- **coarse_embed_dim**: 全局嵌入维度 D。

---

## 八、小结表

| 分支     | 文本表示        | 场景表示        | 相似度形式           | 损失形式        | 权重/备注           |
|----------|-----------------|-----------------|----------------------|-----------------|---------------------|
| Global   | anchor [B,D]    | F_Global [B,D]  | CCL 内建 [B,B]       | CCL(anchor, F_Global) | -                   |
| Object   | F_Object_T      | F_Object_P      | score [B,B] (max+mean) | CCL_input_score(score, dist_Object) | dist_Object 加权    |
| Relation | F_Relation_T    | F_Relation_P    | relation_score [B,B] | CCL_input_score(relation_score, dist_Relation) | dist_Relation 加权  |
| 自监督   | anchor          | F_Global        | -                    | CCL(anchor, anchor) + CCL(F_Global, F_Global) | 各 1.0              |

整体上，MNCL coarse 阶段通过 **Global + Object + Relation** 三个分支的对比损失，以及 **文本/场景自监督** 两项，共同训练多层次的文本–场景对齐，用于后续的 cell 检索与 fine 阶段。

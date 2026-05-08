# OLA-Place: Cross-Modal Place Recognition without Global Descriptors

## Abstract

Cross-modal place recognition aims to retrieve a target 3D location from a spatial map using a natural language description. Most existing methods follow a global descriptor matching paradigm, where the text query and each 3D scene cell are independently compressed into two single vectors and compared by global similarity. Although simple and widely adopted, this paradigm tends to discard the fine-grained object correspondences that are essential for language-guided localization. In this paper, we challenge the necessity of global descriptors and propose **OLA-Place**, an **Object-Level Alignment** framework for cross-modal place recognition without global descriptors. Instead of representing a query or a scene cell as a single holistic embedding, OLA-Place formulates place recognition as set-to-set semantic alignment between textual object mentions and 3D object instances. The framework contains three key modules: **ObjectSet Encoder**, which extracts object-level representations from both language descriptions and 3D scene cells; **Contextual Object Graph Encoder**, which injects intra-set contextual information while preserving object-level granularity; and **Masked Max Alignment**, which computes the query-cell matching score by aligning each textual object mention to its most similar valid 3D object. This formulation is permutation-invariant, naturally handles variable-size object sets, and requires no object-level correspondence annotations. Extensive experiments show that object-level alignment alone significantly outperforms global descriptor-based methods, demonstrating that cross-modal place recognition is better understood as an object-level semantic correspondence problem rather than a global embedding retrieval problem.

## 1. Introduction

Cross-modal place recognition is a fundamental problem for language-guided localization, embodied navigation, mobile robotics, and autonomous driving. Given a natural language query such as "a red car is parked near a pole in front of a building", the goal is to retrieve the corresponding 3D location from a pre-built spatial map. In contrast to image-based or LiDAR-based place recognition, the query and the map belong to different modalities: the query is expressed in free-form language, while the map is represented by 3D geometry, semantic objects, and spatial layouts. This modality gap makes cross-modal place recognition a challenging retrieval problem.

Most text-guided point-cloud localization systems follow a coarse-to-fine strategy. The coarse stage first retrieves a candidate point-cloud submap or scene cell from a large gallery, and the fine stage subsequently estimates the precise coordinate inside the retrieved cell. Since the fine localization result is strongly bounded by the quality of coarse retrieval, the coarse stage is a critical component of the full localization pipeline. This paper follows this coarse-to-fine setting and focuses specifically on improving the coarse cross-modal place recognition stage.

A dominant solution is to learn global descriptors for both modalities. In this paradigm, a text encoder maps the entire language query into a global textual descriptor, while a 3D scene encoder maps each candidate scene cell into a global scene descriptor. The retrieval score is then computed by cosine similarity or dot product between the two global vectors. Formally, given a query \(q\) and a scene cell \(C\), previous methods usually compute

\[
z_q = f_T(q), \qquad z_C = f_C(C), \qquad S(q,C)=\cos(z_q,z_C),
\]

where \(z_q \in \mathbb{R}^d\) and \(z_C \in \mathbb{R}^d\) are compact global descriptors. This strategy is convenient because it reduces cross-modal place recognition to standard vector retrieval. However, it also imposes a strong bottleneck: all objects, attributes, and spatial cues in the query and the scene must be compressed into a single embedding before matching.

We argue that this global descriptor assumption is not well aligned with the nature of language-based place recognition. Natural language descriptions are typically object-centric. A query often specifies a place by mentioning several objects, their attributes, and sometimes their relations, such as "red car", "traffic sign", "building", or "tree beside the road". Similarly, a 3D scene cell is naturally composed of multiple object instances. Therefore, the common semantic units shared by the two modalities are not global scenes, but objects. Compressing the entire query and the entire 3D cell into two holistic vectors may obscure which textual object is grounded by which scene object, dilute discriminative landmarks with irrelevant objects, and reduce interpretability.

Motivated by this observation, we propose a different view: **cross-modal place recognition should be formulated as object-level semantic alignment rather than global descriptor matching**. Instead of asking whether two global descriptors are close, we ask whether each object mentioned in the language query can find a semantically consistent object instance in the candidate 3D cell. This leads to a global-descriptor-free retrieval formulation, where the matching score is directly computed from fine-grained object correspondences.

To this end, we introduce **OLA-Place**, a simple yet effective object-level alignment framework for cross-modal place recognition without global descriptors. OLA-Place represents a text query as a set of textual object embeddings and represents each 3D cell as a set of 3D object embeddings. To preserve context while avoiding global compression, we use a **Contextual Object Graph Encoder** to propagate information among objects within the same modality. The final retrieval score is computed by **Masked Max Alignment**: for each textual object embedding, we search for the most similar valid 3D object embedding in the candidate cell, and then average the best-match similarities over all textual objects. Invalid padded objects are removed by a binary object mask.

This design has several important properties. First, it does not rely on any global descriptor during retrieval. The query-cell score is produced directly from object-level pairwise similarities. Second, it is permutation-invariant with respect to the object order in the 3D cell, which is essential because object instances have no canonical ordering. Third, it supports variable-size object sets through masking. Fourth, it does not require object-level correspondence annotations; only query-cell pair supervision is needed. Finally, the resulting alignment is interpretable, since the model can explicitly indicate which 3D object is selected as the best match for each textual object mention.

The main contributions of this paper are summarized as follows:

1. We propose **OLA-Place**, a global-descriptor-free framework for cross-modal place recognition that reformulates text-to-3D place retrieval as object-level set alignment between textual object mentions and 3D object instances.

2. We introduce three core components: **ObjectSet Encoder** for object-level cross-modal representation learning, **Contextual Object Graph Encoder** for context-aware object embedding, and **Masked Max Alignment** for weakly supervised object-level query-cell matching without object correspondence labels.

3. We show that object-level alignment alone can significantly outperform global descriptor-based methods, suggesting that cross-modal place recognition is fundamentally an object-level semantic correspondence problem rather than a global descriptor retrieval problem.

## 2. Method

### 2.1 Task Formulation

Following the established coarse-to-fine pipeline for text-guided point-cloud localization, OLA-Place focuses on the **coarse place recognition stage**. Given a natural language query, the coarse stage retrieves the most relevant 3D scene cell or point-cloud submap from a gallery. The retrieved cell can then be passed to a downstream fine localization module, such as Text2Loc-style coordinate regression, to estimate a precise position inside the selected submap. This paper does not redesign the fine localization stage; instead, it studies whether the coarse retrieval stage truly requires global descriptors.

Let

\[
\mathcal{Q}=\{q_i\}_{i=1}^{N_q}
\]

denote a set of language queries, and let

\[
\mathcal{G}=\{C_j\}_{j=1}^{N_c}
\]

denote the gallery of 3D scene cells. Each query \(q_i\) describes one target location, and each cell \(C_j\) contains a set of 3D object instances extracted from the point-cloud map. The coarse retrieval objective is

\[
\widetilde{C}_i
=\arg\max_{C_j\in\mathcal{G}} S(q_i,C_j),
\label{eq:coarse_retrieval_obj}
\]

where \(S(q_i,C_j)\) is the cross-modal query-cell matching score. Equivalently, if a distance metric \(d(\cdot,\cdot)\) is used, the objective can be written as

\[
\widetilde{C}_i
=\arg\min_{C_j\in\mathcal{G}} d(q_i,C_j).
\]

Most previous coarse retrieval methods instantiate this objective by learning two global encoders:

\[
\mathbf{z}_i^q=f_q(q_i), \qquad \mathbf{z}_j^c=f_c(C_j),
\]

and then compute

\[
S_{\mathrm{global}}(q_i,C_j)=\cos(\mathbf{z}_i^q,\mathbf{z}_j^c).
\]

This global formulation collapses the entire query and the entire cell into two holistic vectors before comparison. In contrast, OLA-Place keeps the coarse retrieval formulation but replaces global descriptor matching with object-level alignment. Specifically, the query and the candidate cell are represented as two object sets:

\[
\mathcal{T}_i = \{\mathbf{t}_{i1}, \mathbf{t}_{i2}, \ldots, \mathbf{t}_{iM}\},
\]

\[
\mathcal{O}_j = \{\mathbf{o}_{j1}, \mathbf{o}_{j2}, \ldots, \mathbf{o}_{jN_j}\},
\]

where \(\mathbf{t}_{im} \in \mathbb{R}^d\) is the embedding of the \(m\)-th textual object-level unit in query \(q_i\), \(\mathbf{o}_{jn} \in \mathbb{R}^d\) is the embedding of the \(n\)-th 3D object instance in cell \(C_j\), \(M\) is the number of textual object-level units, \(N_j\) is the number of valid 3D objects in cell \(C_j\), and \(d\) is the embedding dimension.

The coarse retrieval score is defined by an object-level alignment function:

\[
S(q_i,C_j)=\operatorname{Align}(\mathcal{T}_i,\mathcal{O}_j),
\label{eq:object_align_score}
\]

rather than by comparing global descriptors. After the coarse cell \(\widetilde{C}_i\) is retrieved, the fine stage can follow existing coordinate regression or pose refinement methods:

\[
\widehat{\mathbf{p}}_i
= h_{\mathrm{fine}}(q_i,\widetilde{C}_i),
\]

where \(\widehat{\mathbf{p}}_i\in\mathbb{R}^2\) or \(\mathbb{R}^3\) denotes the final estimated position. Since our contribution lies in the retrieval formulation, the remainder of this section focuses on the coarse object-level alignment model.

### 2.2 Overview of OLA-Place

OLA-Place consists of three key modules:

1. **ObjectSet Encoder**. This module maps the language query and the 3D scene cell into object-level embedding sets. On the language side, it encodes textual object mentions or object-centric semantic units. On the 3D side, it encodes object instances using their semantic, appearance, geometric, and point-cloud features.

2. **Contextual Object Graph Encoder**. This module refines object embeddings by modeling intra-set contextual interactions. For 3D cells, it propagates information among objects according to object features and relative geometry. For text queries, it models contextual dependencies among textual object mentions. Importantly, it preserves object-level outputs instead of pooling all objects into a global descriptor.

3. **Masked Max Alignment**. This module computes the final query-cell score by pairwise object similarity. Each textual object mention is softly grounded to the most similar valid 3D object instance, and the final score is the average of these best-match similarities.

The overall pipeline can be summarized as:

\[
q_i \xrightarrow{\text{ObjectSet Encoder}} \mathcal{T}_i
\xrightarrow{\text{Contextual Object Graph Encoder}} \widetilde{\mathcal{T}}_i,
\]

\[
C_j \xrightarrow{\text{ObjectSet Encoder}} \mathcal{O}_j
\xrightarrow{\text{Contextual Object Graph Encoder}} \widetilde{\mathcal{O}}_j,
\]

\[
S(q_i,C_j)=\operatorname{MaskedMaxAlign}(\widetilde{\mathcal{T}}_i,\widetilde{\mathcal{O}}_j).
\]

For clarity, we use \(\mathcal{T}_i\) and \(\mathcal{O}_j\) to denote the final context-aware object sets in the following sections.

### 2.3 ObjectSet Encoder

The ObjectSet Encoder converts both modalities into object-indexed feature sets. Its role is not to summarize the entire query or scene, but to produce a sequence of comparable object-level embeddings in a shared \(d\)-dimensional space.

#### Textual ObjectSet Encoding

Given a language query \(q_i\), we first decompose it into \(M\) textual units, where each unit corresponds to an object-centric description or an object-related sentence fragment. Let these units be

\[
q_i = \{u_{i1},u_{i2},\ldots,u_{iM}\}.
\]

Each unit is encoded by a language backbone \(\phi_T(\cdot)\), producing token-level hidden states. A local sequence encoder then aggregates token features within each unit:

\[
\mathbf{h}_{im}=\operatorname{Pool}\bigl(\phi_T(u_{im})\bigr) \in \mathbb{R}^{d_T},
\]

where \(\operatorname{Pool}(\cdot)\) can be max pooling over contextual token embeddings. The resulting unit representation is projected to the shared retrieval dimension:

\[
\mathbf{e}_{im}^{T}=\psi_T(\mathbf{h}_{im})\in\mathbb{R}^{d},
\]

where \(\psi_T(\cdot)\) is a learnable projection layer. Stacking all textual object-level embeddings gives

\[
\mathbf{E}_i^{T}=[\mathbf{e}_{i1}^{T},\mathbf{e}_{i2}^{T},\ldots,\mathbf{e}_{iM}^{T}]\in\mathbb{R}^{M\times d}.
\]

In the implementation, the language branch further applies recurrent and transformer-style contextual layers before the object graph encoding stage. This allows each textual object unit to incorporate local linguistic context while remaining a separate object-level representation. After normalization, the textual object set is

\[
\mathbf{t}_{im}=\frac{\mathbf{e}_{im}^{T}}{\|\mathbf{e}_{im}^{T}\|_2},
\qquad
\mathbf{T}_i=[\mathbf{t}_{i1},\ldots,\mathbf{t}_{iM}]\in\mathbb{R}^{M\times d}.
\]

Unlike global text encoding, \(\mathbf{T}_i\) is not pooled into a single vector for retrieval. Each row of \(\mathbf{T}_i\) remains available for object-level matching.

#### 3D ObjectSet Encoding

Each scene cell \(C_j\) is represented as a set of 3D object instances:

\[
C_j=\{x_{j1},x_{j2},\ldots,x_{jN_j}\}.
\]

For each object \(x_{jn}\), the implementation uses multiple complementary object attributes, including semantic class, color, 3D position, point count, and object-level point-cloud geometry. We denote these raw attributes as

\[
x_{jn}=\left(c_{jn},\rho_{jn},\mathbf{p}_{jn},\nu_{jn},\mathcal{P}_{jn}\right),
\]

where \(c_{jn}\) is the semantic class, \(\rho_{jn}\) is the color attribute, \(\mathbf{p}_{jn}\in\mathbb{R}^{3}\) is the object center, \(\nu_{jn}\) is the number of points, and \(\mathcal{P}_{jn}\) is the object point cloud.

The geometric point-cloud feature is extracted by a PointNet-style encoder:

\[
\mathbf{g}_{jn}=\phi_P(\mathcal{P}_{jn})\in\mathbb{R}^{d_P}.
\]

It is projected into the common embedding space:

\[
\mathbf{a}_{jn}^{\mathrm{pc}}=\psi_P(\mathbf{g}_{jn})\in\mathbb{R}^{d}.
\]

Class, color, position, and point-count attributes are encoded as

\[
\mathbf{a}_{jn}^{\mathrm{cls}}=\psi_{\mathrm{cls}}(c_{jn}),
\]

\[
\mathbf{a}_{jn}^{\mathrm{col}}=\psi_{\mathrm{col}}(\rho_{jn}),
\]

\[
\mathbf{a}_{jn}^{\mathrm{pos}}=\psi_{\mathrm{pos}}(\mathbf{p}_{jn}),
\]

\[
\mathbf{a}_{jn}^{\mathrm{num}}=\psi_{\mathrm{num}}\left(\frac{\nu_{jn}-\mu_{\nu}}{\sigma_{\nu}}\right),
\]

where \(\mu_{\nu}\) and \(\sigma_{\nu}\) are the mean and standard deviation used to normalize the point count. Depending on the selected feature configuration, a subset \(\mathcal{F}_{jn}\) of these feature embeddings is used:

\[
\mathcal{F}_{jn}\subseteq\{\mathbf{a}_{jn}^{\mathrm{pc}},\mathbf{a}_{jn}^{\mathrm{cls}},\mathbf{a}_{jn}^{\mathrm{col}},\mathbf{a}_{jn}^{\mathrm{pos}},\mathbf{a}_{jn}^{\mathrm{num}}\}.
\]

Each component is first \(\ell_2\)-normalized and then fused by a learnable projection:

\[
\mathbf{e}_{jn}^{O}
=\psi_{\mathrm{merge}}\left(
\operatorname{Concat}\left(\left\{\frac{\mathbf{a}}{\|\mathbf{a}\|_2}:\mathbf{a}\in\mathcal{F}_{jn}\right\}\right)
\right)
\in\mathbb{R}^{d}.
\]

The cell-level object matrix before padding is

\[
\mathbf{E}_{j}^{O}=[\mathbf{e}_{j1}^{O},\mathbf{e}_{j2}^{O},\ldots,\mathbf{e}_{jN_j}^{O}]\in\mathbb{R}^{N_j\times d}.
\]

Because different cells contain different numbers of objects, we pad each object set to a fixed maximum size \(N_{\max}\). The padded object matrix is

\[
\bar{\mathbf{E}}_{j}^{O}\in\mathbb{R}^{N_{\max}\times d}.
\]

A binary validity mask distinguishes real objects from padded slots:

\[
\mathbf{m}_j=[m_{j1},m_{j2},\ldots,m_{jN_{\max}}]\in\{0,1\}^{N_{\max}},
\]

with

\[
m_{jn}=\begin{cases}
1, & 1\leq n\leq N_j,\\
0, & N_j<n\leq N_{\max}.
\end{cases}
\]

Finally, valid object embeddings are normalized:

\[
\mathbf{o}_{jn}=\frac{\bar{\mathbf{e}}_{jn}^{O}}{\|\bar{\mathbf{e}}_{jn}^{O}\|_2},
\qquad
\mathbf{O}_{j}=[\mathbf{o}_{j1},\ldots,\mathbf{o}_{jN_{\max}}]\in\mathbb{R}^{N_{\max}\times d}.
\]

### 2.4 Contextual Object Graph Encoder

Object-level alignment should not be confused with context-free object matching. A textual mention or a 3D object instance can be ambiguous when considered independently. For example, a "pole" becomes more informative when it appears near a road, a car, or a traffic sign. Therefore, OLA-Place uses a **Contextual Object Graph Encoder** to inject intra-set context into each object embedding while preserving object-level outputs.

#### 3D Contextual Object Graph

For each cell \(C_j\), we define a complete object graph

\[
\mathcal{G}_j^{O}=(\mathcal{V}_j^{O},\mathcal{E}_j^{O}),
\]

where each node corresponds to an object slot and each edge describes pairwise relative geometry. Let \(\mathbf{p}_{jn}\in\mathbb{R}^{3}\) be the center of the \(n\)-th object. The relative displacement from object \(b\) to object \(a\) is

\[
\mathbf{r}_{jab}=\mathbf{p}_{ja}-\mathbf{p}_{jb}\in\mathbb{R}^{3}.
\]

The relative geometry is embedded as

\[
\mathbf{q}_{jab}=\psi_r(\mathbf{r}_{jab})\in\mathbb{R}^{d},
\]

where \(\psi_r(\cdot)\) is a learnable linear projection followed by a non-linear activation. Let \(\mathbf{o}_{jn}^{(0)}=\mathbf{o}_{jn}\) be the initial object feature. At graph layer \(\ell\), OLA-Place updates object-level features and relation-level features. The object update is implemented with a sequential/contextual operator over object slots:

\[
\mathbf{u}_{jn}^{(\ell)}=\operatorname{LSTM}_{O}^{(\ell)}\left(\mathbf{o}_{j1}^{(\ell-1)},\ldots,\mathbf{o}_{jN_{\max}}^{(\ell-1)}\right)_n.
\]

The object mask is then applied:

\[
\mathbf{o}_{jn}^{(\ell)}=m_{jn}\,\mathbf{u}_{jn}^{(\ell)}.
\]

In parallel, a relation feature is constructed for each object pair by concatenating the source object, target object, and their relative geometric feature:

\[
\mathbf{b}_{jab}^{(\ell)}=
\left[
\mathbf{o}_{ja}^{(\ell-1)};
\mathbf{o}_{jb}^{(\ell-1)};
\mathbf{q}_{jab}^{(\ell-1)}
\right]
\in\mathbb{R}^{3d}.
\]

The relation update is

\[
\mathbf{q}_{jab}^{(\ell)}
=m_{ja}m_{jb}\,\sigma\left(\mathbf{W}_r^{(\ell)}\mathbf{b}_{jab}^{(\ell)}+\mathbf{b}_r^{(\ell)}\right),
\]

where \(\sigma(\cdot)\) is a \(\tanh\) activation. This update corresponds to the implementation where relation features are repeatedly updated from object features and relative position encodings.

After \(L\) graph layers, the final 3D context-aware object representation is

\[
\widetilde{\mathbf{O}}_j
=[\widetilde{\mathbf{o}}_{j1},\ldots,\widetilde{\mathbf{o}}_{jN_{\max}}]
=[\mathbf{o}_{j1}^{(L)},\ldots,\mathbf{o}_{jN_{\max}}^{(L)}].
\]

The relation stream can also be aggregated by an edge-wise spatial attention operation. For each target object \(a\), relation features from all neighboring objects are first projected:

\[
\mathbf{s}_{jab}=\psi_s(\mathbf{q}_{jab}^{(L)}),
\]

and normalized over neighboring objects:

\[
\alpha_{jab}
=\frac{\exp(\mathbf{s}_{jab})}{\sum_{k=1}^{N_{\max}}\exp(\mathbf{s}_{jak})}.
\]

The relation-aware object context is

\[
\mathbf{c}_{ja}=m_{ja}\sum_{b=1}^{N_{\max}}\alpha_{jab}\mathbf{s}_{jab}.
\]

Although this relation-aware stream can be used by relation-level branches, OLA-Place uses the object-level stream \(\widetilde{\mathbf{O}}_j\) as the retrieval representation. Thus, contextual information is injected into each object, but the representation remains object-indexed and is not collapsed into a scene-level descriptor.

#### Textual Contextual Object Encoding

On the language side, textual object units also interact with one another. Let the initial textual object embeddings be \(\mathbf{t}_{im}^{(0)}=\mathbf{t}_{im}\). A textual graph or sequence encoder updates them as

\[
\mathbf{t}_{im}^{(\ell)}
=g_T^{(\ell)}\left(\mathbf{t}_{im}^{(\ell-1)},\{\mathbf{t}_{ik}^{(\ell-1)}\}_{k=1}^{M}\right).
\]

In practice, this contextualization can be implemented with recurrent layers and self-attention modules. After \(L_T\) layers, the language-side output is

\[
\widetilde{\mathbf{T}}_i
=[\widetilde{\mathbf{t}}_{i1},\ldots,\widetilde{\mathbf{t}}_{iM}]
=[\mathbf{t}_{i1}^{(L_T)},\ldots,\mathbf{t}_{iM}^{(L_T)}]
\in\mathbb{R}^{M\times d}.
\]

Both object sets are normalized before alignment:

\[
\widetilde{\mathbf{t}}_{im}\leftarrow \frac{\widetilde{\mathbf{t}}_{im}}{\|\widetilde{\mathbf{t}}_{im}\|_2},
\qquad
\widetilde{\mathbf{o}}_{jn}\leftarrow \frac{\widetilde{\mathbf{o}}_{jn}}{\|\widetilde{\mathbf{o}}_{jn}\|_2}.
\]

The key distinction from global descriptor learning is that no pooling operation is used to collapse \(\widetilde{\mathbf{T}}_i\) or \(\widetilde{\mathbf{O}}_j\) into a single retrieval descriptor. The model keeps an object-indexed representation until the final alignment stage.

For notational simplicity, we omit the tilde in the next section and use \(\mathbf{T}_i\) and \(\mathbf{O}_j\) to denote the context-aware object embeddings.

### 2.5 Masked Max Alignment

The central component of OLA-Place is **Masked Max Alignment**, which computes a query-cell matching score from pairwise object similarities.

Given a query object set

\[
\mathbf{T}_i = [\mathbf{t}_{i1},\ldots,\mathbf{t}_{iM}] \in \mathbb{R}^{M\times d}
\]

and a cell object set

\[
\mathbf{O}_j = [\mathbf{o}_{j1},\ldots,\mathbf{o}_{jN_{\max}}] \in \mathbb{R}^{N_{\max}\times d},
\]

we first compute the pairwise object similarity matrix:

\[
\mathbf{A}_{ij} = \mathbf{T}_i \mathbf{O}_j^\top \in \mathbb{R}^{M\times N_{\max}},
\]

where the \((m,n)\)-th entry is

\[
A_{ij}^{mn}=\mathbf{t}_{im}^{\top}\mathbf{o}_{jn}=\cos(\mathbf{t}_{im},\mathbf{o}_{jn}).
\]

To remove padded object slots, we apply the object validity mask \(\mathbf{m}_j\). For invalid slots, the similarity is set to a very small value:

\[
\widehat{A}_{ij}^{mn}=
\begin{cases}
A_{ij}^{mn}, & \text{if } m_{jn}=1, \\
-\infty, & \text{if } m_{jn}=0.
\end{cases}
\]

Then, for each textual object mention \(\mathbf{t}_{im}\), we select the most similar valid 3D object instance in the candidate cell:

\[
a_{ij}^{m}=\max_{1\leq n\leq N_{\max}} \widehat{A}_{ij}^{mn}.
\]

The final object-level alignment score between query \(q_i\) and cell \(C_j\) is obtained by averaging the best-match scores over all textual objects:

\[
S(q_i,C_j)=\frac{1}{M}\sum_{m=1}^{M} a_{ij}^{m}
=\frac{1}{M}\sum_{m=1}^{M}\max_{n:m_{jn}=1}\cos(\mathbf{t}_{im},\mathbf{o}_{jn}).
\]

This score directly measures whether the objects mentioned in the query can be grounded in the 3D cell. The max operation makes the alignment invariant to object ordering, while the mask allows the method to handle variable-size object sets. Since no ground-truth object correspondence is needed, the alignment can be learned using only query-cell retrieval supervision.

### 2.6 Batch-wise Contrastive Training

During training, each mini-batch contains \(B\) matched query-cell pairs:

\[
\mathcal{B}=\{(q_i,C_i)\}_{i=1}^{B}.
\]

For all query-cell combinations in the mini-batch, we compute the object-level score matrix

\[
\mathbf{S}^{O}\in\mathbb{R}^{B\times B},
\qquad
S_{ij}^{O}=S(q_i,C_j).
\]

Following the implementation, this matrix is computed efficiently by tensorized pairwise multiplication. Let

\[
\mathbf{T}\in\mathbb{R}^{B\times M\times d}
\]

be the batch of textual object embeddings and

\[
\mathbf{O}\in\mathbb{R}^{B\times N_{\max}\times d}
\]

be the batch of 3D object embeddings. The four-dimensional pairwise similarity tensor is

\[
\mathcal{A}_{ijmn}=\mathbf{t}_{im}^{\top}\mathbf{o}_{jn},
\qquad
\mathcal{A}\in\mathbb{R}^{B\times B\times M\times N_{\max}}.
\]

The object mask tensor is broadcast to all query-cell pairs:

\[
\mathcal{M}_{ijmn}=m_{jn}.
\]

Masked similarities are defined as

\[
\widehat{\mathcal{A}}_{ijmn}=
\begin{cases}
\mathcal{A}_{ijmn}, & \mathcal{M}_{ijmn}=1,\\
-\infty, & \mathcal{M}_{ijmn}=0.
\end{cases}
\]

The score matrix used by the loss is then

\[
S_{ij}^{O}
=\frac{1}{M}\sum_{m=1}^{M}\max_{1\leq n\leq N_{\max}}\widehat{\mathcal{A}}_{ijmn}.
\]

The diagonal entries \(S_{ii}^{O}\) correspond to positive query-cell pairs, while off-diagonal entries \(S_{ij}^{O}\), \(i\neq j\), serve as in-batch negatives.

The default training objective is a bidirectional contrastive loss over the score matrix. For query-to-cell retrieval, we use

\[
\mathcal{L}_{q\rightarrow C}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(S_{ii}^{O}/\tau)}
{\sum_{j=1}^{B}\exp(S_{ij}^{O}/\tau)},
\]

where \(\tau\) is the temperature parameter. The symmetric cell-to-query loss is

\[
\mathcal{L}_{C\rightarrow q}
= -\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(S_{ii}^{O}/\tau)}
{\sum_{j=1}^{B}\exp(S_{ji}^{O}/\tau)}.
\]

The total objective is

\[
\mathcal{L}_{\mathrm{NCE}}
=\mathcal{L}_{q\rightarrow C}+\mathcal{L}_{C\rightarrow q}.
\]

In addition to the standard contrastive view, the implementation adopts a contrastive calibration loss that emphasizes hard negatives according to the current score distribution and feature-space distance. First, the score matrix is converted into row-wise and column-wise probabilities:

\[
P_{ij}^{q\rightarrow C}
=\frac{\exp(S_{ij}^{O}/\tau)}{\sum_{k=1}^{B}\exp(S_{ik}^{O}/\tau)},
\]

\[
P_{ij}^{C\rightarrow q}
=\frac{\exp(S_{ij}^{O}/\tau)}{\sum_{k=1}^{B}\exp(S_{kj}^{O}/\tau)}.
\]

Let \(\bar{\mathbf{t}}_i\) and \(\bar{\mathbf{o}}_j\) be mean object embeddings:

\[
\bar{\mathbf{t}}_i
=\frac{1}{M}\sum_{m=1}^{M}\mathbf{t}_{im},
\]

\[
\bar{\mathbf{o}}_j
=\frac{1}{\sum_{n=1}^{N_{\max}}m_{jn}}
\sum_{n=1}^{N_{\max}}m_{jn}\mathbf{o}_{jn}.
\]

Both are normalized:

\[
\bar{\mathbf{t}}_i\leftarrow\frac{\bar{\mathbf{t}}_i}{\|\bar{\mathbf{t}}_i\|_2},
\qquad
\bar{\mathbf{o}}_j\leftarrow\frac{\bar{\mathbf{o}}_j}{\|\bar{\mathbf{o}}_j\|_2}.
\]

The pairwise feature distance is

\[
D_{ij}=\left\|\bar{\mathbf{t}}_i-\bar{\mathbf{o}}_j\right\|_2.
\]

It is normalized and converted to a distance-aware weight:

\[
\widetilde{D}_{ij}=\frac{D_{ij}}{\max_{a,b}D_{ab}},
\]

\[
W_{ij}=\left(1-\widetilde{D}_{ij}+\epsilon\right)^{1/\alpha},
\]

where \(\epsilon\) is a small constant and \(\alpha\) controls the sharpness. Large weights are assigned to pairs that are close in the current embedding space and therefore more likely to be hard negatives.

To focus on hard negatives, a binary mining mask \(H\in\{0,1\}^{B\times B}\) is constructed from the highest-probability non-matching entries. For each query \(i\), let \(\mathcal{N}_i\) be the selected negative cell indices:

\[
\mathcal{N}_i
=\operatorname{TopK}\left(\{P_{ij}^{q\rightarrow C}:j\neq i\},K\right).
\]

Then

\[
H_{ij}=\begin{cases}
1, & j\in\mathcal{N}_i,\\
0, & \text{otherwise}.
\end{cases}
\]

A general contrastive penalty \(\varphi(\cdot)\) is applied to the negative probabilities. In the logarithmic setting used by default,

\[
\varphi(x)=-\log(1-x+\epsilon).
\]

The calibrated object-level loss can be written as

\[
\mathcal{L}_{\mathrm{obj}}
=\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}
H_{ij}W_{ij}\varphi(P_{ij}^{q\rightarrow C})
+
\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}
H_{ij}W_{ji}\varphi(P_{ij}^{C\rightarrow q}).
\]

This loss suppresses highly competitive negative cells while preserving the same object-level retrieval signal \(S_{ij}^{O}\). Importantly, both \(\mathcal{L}_{\mathrm{NCE}}\) and \(\mathcal{L}_{\mathrm{obj}}\) operate on the masked object-alignment score matrix, not on global descriptors.

### 2.7 Inference

At inference time, all database cells are encoded into object-level sets and stored with their object masks. Given a query \(q\), OLA-Place computes its textual object set and evaluates the masked alignment score against every candidate cell:

\[
S(q,C_j)=\frac{1}{M}\sum_{m=1}^{M}\max_{n:m_{jn}=1}\cos(\mathbf{t}_{m},\mathbf{o}_{jn}).
\]

The database cells are ranked according to \(S(q,C_j)\), and the top-ranked cells are returned as retrieval results:

\[
\operatorname{Rank}(q)=\operatorname{argsort}_{C_j\in\mathcal{D}}\left(-S(q,C_j)\right).
\]

The entire inference procedure is global-descriptor-free: no query-cell score is computed from holistic scene embeddings. Instead, every score is derived from explicit object-level semantic alignment.

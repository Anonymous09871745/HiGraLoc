"""
ObjectBranch: 三分支独立模型之一

该分支专注于对象级特征学习:
- language_encoder: 编码文本描述为对象级向量 (F_Object_T)
- object_encoder: 编码场景点云为对象级向量 (F_Object_P)
- MSG.object_encoder: 多层场景图编码器，处理对象之间的关系
- RiemannianFeatureTransform: 黎曼流形特征变换（创新点）

训练目标: 使 F_Object_T 与 F_Object_P 最大化相似度

数学创新（Riemannian Geometry）:
- 将对象嵌入建模到双曲空间 (Hyperbolic Space)
- 使用可学习曲率 c 控制空间的"弯曲程度"
- 指数映射和对数映射在流形与切空间之间转换
- 仅需 3 个可学习参数 (c, translation, geodesic_scale)
"""

from models.object_encoder import ObjectEncoder
from models.language_encoder import LanguageEncoder
from models.msg_encoder import ObjectMsgEncoder
from models.riemannian_geometry import RiemannianFeatureTransform
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List


class ObjectBranch(nn.Module):
    """
    Object-Level Branch for cell retrieval.

    输出:
        F_Object_T: 文本描述的对象级向量 [B, num_mentioned, embed_dim]
        F_Object_P: 场景的对象级向量 [B, object_size, embed_dim]
        masks: 对象级掩码
    """

    def __init__(
        self,
        known_classes: List[str],
        known_colors: List[str],
        args
    ):
        super(ObjectBranch, self).__init__()
        self.embed_dim = args.coarse_embed_dim
        self.object_size = args.object_size

        # Textual module
        self.language_encoder = LanguageEncoder(
            args.coarse_embed_dim,
            hungging_model=args.hungging_model,
            fixed_embedding=args.fixed_embedding,
            intra_module_num_layers=args.intra_module_num_layers,
            intra_module_num_heads=args.intra_module_num_heads,
            is_fine=False,
            inter_module_num_layers=args.inter_module_num_layers,
            inter_module_num_heads=args.inter_module_num_heads,
        )

        # Object module
        self.object_encoder = ObjectEncoder(
            args.coarse_embed_dim,
            known_classes,
            known_colors,
            args
        )

        # Multi-level Scene Graph Encoder
        self.MSG = ObjectMsgEncoder(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            layer=args.num_of_hidden_layer,
            object_size=args.object_size
        )

        # Riemannian Manifold Feature Transform (创新点 - 默认禁用以兼容旧权重)
        # 训练时设置 --use_riemannian 启用
        self.use_riemannian = getattr(args, 'use_riemannian', False)
        self.riemannian_transform = None
        if self.use_riemannian:
            manifold_type = getattr(args, 'riemannian_manifold', 'hyperbolic')
            self.riemannian_transform = RiemannianFeatureTransform(
                embed_dim=self.embed_dim,
                manifold_type=manifold_type,
                learnable_curvature=True
            )
            print("[ObjectBranch] Riemannian Geometry 已启用")

    def encode_text(self, descriptions):
        """编码文本描述为对象级向量"""
        _, F_Object_T, _ = self.language_encoder(descriptions)
        F_Object_T = F.normalize(F_Object_T, dim=-1)
        return F_Object_T

    def encode_objects(self, objects, object_points):
        """编码场景为对象级向量"""
        embeddings, _ = self.object_encoder(objects, object_points)

        # Multi-level Scene Graph Encoder
        F_Object_P, _, index_list, object_level_masks, _ = self.MSG(objects, embeddings)
        F_Object_P = F.normalize(F_Object_P, dim=-1)

        # Riemannian Manifold Feature Transform (创新点)
        if self.use_riemannian and self.riemannian_transform is not None:
            F_Object_P = self.riemannian_transform(F_Object_P)

        return F_Object_P, object_level_masks

    @property
    def device(self):
        return next(self.language_encoder.parameters()).device

    def get_device(self):
        return next(self.language_encoder.parameters()).device

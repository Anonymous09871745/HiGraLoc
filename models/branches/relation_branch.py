"""
RelationBranch: 三分支独立模型之一

该分支专注于关系级特征学习:
- language_encoder: 编码文本描述为关系级向量 (F_Relation_T)
- object_encoder: 编码场景点云为基础对象向量
- MSG + ISRE: 多层场景图编码器 + 信息-辛几何增强

训练目标: 使 F_Relation_T 与 F_Relation_P 最大化相似度

核心架构:
object_embedding → [多层消息传递] → relation_level_features [N,N,D] 
                                              ↓
                                    ISRE（信息-辛几何增强）
                                              ↓
                                    [ESA pooling] → [N,D] → F_Relation_P

数学创新（ISRE）:
- 信息几何层：将关系建模为指数族分布，使用 Fisher-Rao 度量
- 辛几何层：关系传播遵循哈密顿动力学，保持相空间体积
- ESA pooling：对增强后的成对关系进行注意力池化
"""

from models.object_encoder import ObjectEncoder
from models.language_encoder import LanguageEncoder
from models.msg_encoder import ObjectMsgEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List, Tuple


class RelationBranch(nn.Module):
    """
    Relation-Level Branch for cell retrieval.

    输出:
        F_Relation_T: 文本描述的关系级向量 [B, num_mentioned, embed_dim]
        F_Relation_P: 场景的关系级向量 [B, object_size, embed_dim]
        masks: 关系级掩码
    """

    def __init__(
        self,
        known_classes: List[str],
        known_colors: List[str],
        args
    ):
        super(RelationBranch, self).__init__()
        self.embed_dim = args.coarse_embed_dim
        self.object_size = args.object_size
        self.num_mentioned = args.num_mentioned

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

        # MSG + ISRE: 多层场景图编码器 + 信息-辛几何增强
        # ISRE 集成在 MSG 内部，位于 ESA pooling 之前
        # 流程: [多层消息传递] → relation_level_features → ISRE → ESA → F_Relation_P
        # 默认启用 ISRE，使用 --no_isre 禁用
        use_isre = not getattr(args, 'no_isre', False)
        self.msg = ObjectMsgEncoder(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            layer=args.num_of_hidden_layer,
            object_size=args.object_size,
            use_isre=use_isre,
            args=args
        )

    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        """编码文本描述为关系级向量"""
        _, _, F_Relation_T = self.language_encoder(descriptions)
        F_Relation_T = F.normalize(F_Relation_T, dim=-1)
        return F_Relation_T

    def encode_objects(
        self, 
        objects: List, 
        object_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码场景为关系级向量
        
        流程:
        object_embedding → MSG → relation_level_features → ISRE → ESA → F_Relation_P
        
        Args:
            objects: 对象列表
            object_points: 点云数据
            
        Returns:
            F_Relation_P: [B, object_size, D] 关系级特征
            masks: [B, object_size, 1] 掩码
        """
        # 基础对象编码
        embeddings, _ = self.object_encoder(objects, object_points)
        
        # MSG + ISRE 编码
        # 内部流程: object_embedding → [多层消息传递] → relation_level_features [N,N,D]
        #                                          ↓
        #                                ISRE（信息-辛几何增强）
        #                                          ↓
        #                                [ESA pooling] → [N,D] → F_Relation_P
        _, F_Relation_P, _, object_level_masks, relation_level_masks = self.msg(
            objects, embeddings
        )
        
        # 归一化
        F_Relation_P = F.normalize(F_Relation_P, dim=-1)
        
        # 确保维度匹配
        B = len(objects)
        D = self.embed_dim
        if F_Relation_P.size(1) < self.object_size:
            padding = torch.zeros(
                B, self.object_size - F_Relation_P.size(1), D,
                device=F_Relation_P.device,
                dtype=F_Relation_P.dtype
            )
            F_Relation_P = torch.cat([F_Relation_P, padding], dim=1)
        
        # 返回 object_level_masks 用于训练（形状 [B, N, 1]）
        # 注意：relation_level_masks 形状是 [B, N, N, 1]，用于其他场景
        return F_Relation_P, object_level_masks

    @property
    def device(self):
        return next(self.language_encoder.parameters()).device

    def get_device(self):
        return next(self.language_encoder.parameters()).device

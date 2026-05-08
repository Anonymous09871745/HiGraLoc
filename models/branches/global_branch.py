"""
GlobalBranch: 三分支独立模型之一

该分支专注于全局特征学习:
- language_encoder: 编码文本描述为全局向量 (anchor)
- object_encoder: 编码场景点云为全局向量 (F_Global)
- bir_lstm: 处理对象序列提取全局特征

训练目标: 使 anchor 与 F_Global 最大化相似度
"""

from models.object_encoder import ObjectEncoder
from models.language_encoder import LanguageEncoder
from models.bir_lstm import ConvBlock
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List


class GlobalBranch(nn.Module):
    """
    Global Branch for cell retrieval.

    输出:
        anchor: 文本描述的全局向量 [B, embed_dim]
        F_Global: 场景的全局向量 [B, embed_dim]
    """

    def __init__(
        self,
        known_classes: List[str],
        known_colors: List[str],
        args
    ):
        super(GlobalBranch, self).__init__()
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

        # LSTM for global feature extraction
        self.layers = ConvBlock()

    def encode_text(self, descriptions):
        """编码文本描述为全局向量"""
        anchor, _, _ = self.language_encoder(descriptions)
        anchor = F.normalize(anchor, dim=-1)
        return anchor

    def encode_objects(self, objects, object_points):
        """编码场景为全局向量"""
        embeddings, pos_positions = self.object_encoder(objects, object_points)
        embeddings = F.normalize(embeddings, dim=-1)

        object_size = self.object_size
        index_list = [0]
        last = 0

        F_Global = torch.zeros(
            len(objects), object_size, self.embed_dim, device=self.device
        )

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            F_Global[idx, :num_object] = embeddings[start:(start + num_object)]

        F_Global, XR = self.layers(F_Global)
        F_Global = F_Global.permute(1, 0, 2).contiguous()
        del embeddings, pos_positions

        F_Global = F_Global.max(dim=0)[0]
        F_Global = F.normalize(F_Global, dim=-1)

        return F_Global

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

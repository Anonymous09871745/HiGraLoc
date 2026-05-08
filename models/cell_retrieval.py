from models.msg_encoder import ObjectMsgEncoder
from models.object_encoder import ObjectEncoder
from models.language_encoder import LanguageEncoder
import numpy as np
import torch_geometric.nn as gnn
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List
import os
import sys
from models.bir_lstm import ConvBlock
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]
                              ), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                for i in range(1, len(channels))
            ]
        )


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Coarse module for cell retrieval.
        Implemented as a language encoder and an object encoder.
        The object encoder aggregates information about a varying count of multiple objects through a DGCNN architecture.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim

        """
        Object module
        """

        # CARE: possibly handle variation in forward()!

        self.object_encoder = ObjectEncoder(
            args.coarse_embed_dim, known_classes, known_colors, args)
        self.object_size = args.object_size
        num_heads = args.object_inter_module_num_heads
        num_layers = args.object_inter_module_num_layers

        self.obj_inter_module = nn.ModuleList([nn.TransformerEncoderLayer(
            args.coarse_embed_dim, num_heads,  dim_feedforward=2 * args.coarse_embed_dim) for _ in range(num_layers)])

        """
        Textual module
        """
        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,
                                                hungging_model=args.hungging_model,
                                                fixed_embedding=args.fixed_embedding,
                                                intra_module_num_layers=args.intra_module_num_layers,
                                                intra_module_num_heads=args.intra_module_num_heads,
                                                is_fine=False,
                                                inter_module_num_layers=args.inter_module_num_layers,
                                                inter_module_num_heads=args.inter_module_num_heads,
                                                )

        # Multi-level Scene Graph Generater
        self.MSG = ObjectMsgEncoder(in_features=self.embed_dim, out_features=self.embed_dim,
                                    layer=args.num_of_hidden_layer, object_size=args.object_size)
        self.layers = ConvBlock()
    def encode_text(self, descriptions):

        description_encodings, F_Object_T, F_Relation_T = self.language_encoder(
            descriptions)  # [B, DIM]

        description_encodings = F.normalize(description_encodings, dim=-1)

        return description_encodings, F_Object_T, F_Relation_T

    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """

        batch = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        embeddings, pos_postions = self.object_encoder(objects, object_points)

        # Multi-level Scene Graph Generater
        F_Object_P, F_Relation_P, index_list, object_level_masks, relation_level_masks = self.MSG(
            objects, embeddings)
        F_Object_P = F.normalize(F_Object_P, dim=-1)
        F_Relation_P = F.normalize(F_Relation_P, dim=-1)

        object_size = self.object_size

        index_list = [0]
        last = 0

        F_Global = torch.zeros(len(objects), object_size,
                               self.embed_dim).to(self.device)

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)

        embeddings = F.normalize(embeddings, dim=-1)

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            F_Global[idx, : num_object] = embeddings[start: (start + num_object)]
        F_Global, XR = self.layers(F_Global)
        F_Global = F_Global.permute(1, 0, 2).contiguous()

        del embeddings, pos_postions

        F_Global = F_Global.max(dim=0)[0]
        F_Global = F.normalize(F_Global, dim=-1)

        return F_Global, F_Object_P, F_Relation_P, object_level_masks, relation_level_masks

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

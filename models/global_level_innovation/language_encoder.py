from models.msg_encoder import LanguageMsgEncoder
import torch_geometric.nn as gnn
from transformers import AutoTokenizer, T5EncoderModel
from nltk import tokenize as text_tokenize
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from models.mamba_model import Mamba2

torch.autograd.set_detect_anomaly(True)


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
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]
                              ), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )


class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,  hungging_model=None, fixed_embedding=False,
                 intra_module_num_layers=2, intra_module_num_heads=4,
                 is_fine=False, inter_module_num_layers=2, inter_module_num_heads=4,
                 num_of_hidden_layer=3
                 ):
        """Language encoder to encode a set of hints"""
        super(LanguageEncoder, self).__init__()
        self.is_fine = is_fine
        self.tokenizer = AutoTokenizer.from_pretrained(hungging_model)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.llm_model = T5EncoderModel.from_pretrained(hungging_model)
        if fixed_embedding:
            self.fixed_embedding = True
            for para in self.llm_model.parameters():
                para.require_grads = False
        else:
            self.fixed_embedding = False

        input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]

        self.intra_module = nn.ModuleList([nn.TransformerEncoderLayer(
            input_dim, intra_module_num_heads,  dim_feedforward=input_dim * 4) for _ in range(intra_module_num_layers)])

        self.inter_mlp = get_mlp(
            [input_dim, embedding_dim], add_batchnorm=True)

        # Multi-level Scene Graph Generater
        self.MSG = LanguageMsgEncoder(
            in_features=embedding_dim, out_features=embedding_dim, layer=num_of_hidden_layer)
        if not is_fine:
            self.inter_module = nn.ModuleList([nn.TransformerEncoderLayer(
                embedding_dim, inter_module_num_heads,  dim_feedforward=embedding_dim * 4) for _ in range(inter_module_num_layers)])
        self.mamba_layer_intra = Mamba2(input_dim)
        self.mamba_layer_inter = Mamba2(embedding_dim)
        self.lstm1 = torch.nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=1024, hidden_size=1024)
    def forward(self, descriptions):
        split_union_sentences = []
        for description in descriptions:
            split_union_sentences.extend(
                text_tokenize.sent_tokenize(description))

        batch_size = len(descriptions)
        num_sentence = len(split_union_sentences) // batch_size

        inputs = self.tokenizer(split_union_sentences,
                                return_tensors="pt", padding="longest")
        shorten_sentences_indices = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        shorten_sentences_indices = shorten_sentences_indices.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.llm_model(input_ids=shorten_sentences_indices,
                             attention_mask=attention_mask,
                             output_attentions=False)
        description_encodings = out.last_hidden_state

        if self.fixed_embedding:
            description_encodings = description_encodings.detach()

        description_encodings = description_encodings.permute(1, 0, 2)
        
        description_encodings, (h_n, c_n) = self.lstm2(description_encodings)

        for idx in range(len(self.intra_module)):
            description_encodings = self.intra_module[idx](description_encodings)
        description_encodings = description_encodings.permute(1, 0, 2).contiguous()
        description_encodings = description_encodings.max(dim=1)[0]

        description_encodings = self.inter_mlp(description_encodings)
        description_encodings, (h_n, c_n) = self.lstm1(description_encodings)
        description_encodings = description_encodings.view(
            batch_size, num_sentence, -1)

        # Multi-level Scene Graph Generater
        F_Object_T, F_Relation_T = self.MSG(
            description_encodings.clone().to('cuda'))
        F_Object_T = F.normalize(F_Object_T, dim=-1)
        F_Relation_T = F.normalize(F_Relation_T, dim=-1)

        if self.is_fine:
            return description_encodings
        description_encodings = self.mamba_layer_inter(description_encodings)
        
        description_encodings = description_encodings.permute(1, 0, 2)

        for idx in range(len(self.inter_module)):
            description_encodings += self.inter_module[idx](
                description_encodings)

        description_encodings = description_encodings.max(
            dim=0)[0]  # [N_TEXT,B,D] baseline
        return description_encodings, F_Object_T, F_Relation_T

    @property
    def device(self):
        return next(self.inter_mlp.parameters()).device

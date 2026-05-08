import torch.nn as nn
import torch
import torch.nn.functional as F


class ObjectMsgEncoder(nn.Module):
    def __init__(self, in_features, out_features, layer=3, object_size=28, use_isre=False, args=None):
        super(ObjectMsgEncoder, self).__init__()
        self.embed_dim = in_features
        self.bias = True
        self.use_isre = use_isre

        self.linear_relation_object = nn.Linear(
            3, self.embed_dim, bias=self.bias).to('cuda')
        self.tanh = nn.Tanh().to('cuda')
        self.layer = layer

        self.go = nn.Linear(in_features, out_features).to('cuda')
        self.gr = nn.Linear(3*in_features, out_features).to('cuda')

        self.esa = nn.Linear(in_features, out_features).to('cuda')

        self.object_size = object_size
        
        self.lstm_for_go = nn.LSTM(input_size=in_features, hidden_size=out_features, batch_first=True, bidirectional=False)
        
        # ISRE 增强（可选）
        self.isre = None
        if use_isre and args is not None:
            from models.information_symplectic_encoder_light import LiteInformationSymplecticEncoder
            bottleneck_dim = getattr(args, 'lite_isre_bottleneck_dim', 16)
            self.isre = LiteInformationSymplecticEncoder(
                embed_dim=self.embed_dim,
                bottleneck_dim=bottleneck_dim
            )
            self.isre_alpha = getattr(args, 'lite_isre_alpha', 1.0)
            print(f"[MSG+ISRE] ISRE 瓶颈维度: {bottleneck_dim} | 参数量: {self.isre.count_parameters():,} | α={self.isre_alpha}")
    def device(self):
        return self.go.weight.device

    def forward(self, objects, object_embedding):
        object_size = self.object_size
        index_list = [0]
        last = 0
        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)

        centers = ...  # [index_list[-1],3]
        for i in range(len(objects)):
            for j in range(len(objects[i])):
                if (i == 0 and j == 0):
                    centers = torch.tensor(objects[i][j].get_center())
                else:
                    centers = torch.vstack(
                        (centers, torch.tensor(objects[i][j].get_center())))

        new_centers = torch.zeros(len(objects), object_size, 3)

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            new_centers[idx, : num_object] = centers[start: (
                start + num_object)]

        relation = new_centers.unsqueeze(2)-new_centers.unsqueeze(1)
        relation = relation.to('cuda')
        relation = self.linear_relation_object(relation)

        object_level_features = torch.zeros(
            len(objects), object_size, self.embed_dim).to('cuda')
        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            object_level_features[idx, : num_object] = object_embedding[start: (
                start + num_object)]

        object_level_masks = torch.zeros(
            (len(objects), object_size, 1), dtype=torch.float32).to('cuda')
        relation_level_masks = torch.zeros(
            (len(objects), object_size, object_size, 1), dtype=torch.float32).to('cuda')

        # create object-level mask
        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            object_level_masks[idx, :num_object_raw, 0] = 1
            relation_level_masks[idx, :num_object_raw, :num_object_raw, 0] = 1

        relation_level_features = relation

        for i in range(self.layer):
            # update object level features
            # new_object_level_features = self.tanh(self.go(object_level_features))
            new_object_level_features = self.lstm_for_go(object_level_features)[0]
            new_object_level_features = new_object_level_features*object_level_masks

            # update realtion level features
            object_level_features_repeated = torch.unsqueeze(
                object_level_features, 2).repeat(1, 1, object_size, 1)
            object_level_features_unsqueezed_transposed = torch.unsqueeze(
                object_level_features, 1).repeat(1, object_size, 1, 1)
            rlf = torch.cat((object_level_features_repeated,
                            object_level_features_unsqueezed_transposed), dim=3)
            rlf = torch.cat((rlf, relation_level_features), dim=3)
            rlf = self.tanh(self.gr(rlf))
            relation_level_features = rlf*relation_level_masks

            object_level_features = new_object_level_features

        # ========== ISRE 几何增强（在 ESA pooling 之前）==========
        # 流程: relation_level_features [B,N,N,D] → ISRE → ESA → [B,N,D]
        if self.use_isre and self.isre is not None:
            # ISRE 增强
            enhanced_relations, _ = self.isre(relation_level_features)
            # 残差连接：原始关系 + 增强关系
            relation_level_features = relation_level_features + self.isre_alpha * enhanced_relations
            # 重新应用掩码
            relation_level_features = relation_level_features * relation_level_masks

        # esa
        relation_level_features = self.esa(relation_level_features)
        attn = nn.Softmax(dim=1)(relation_level_features -
                                 torch.max(relation_level_features, dim=1)[0].unsqueeze(1))
        relation_level_features = torch.sum(
            attn*relation_level_features, dim=-2)*object_level_masks

        return object_level_features, relation_level_features, index_list, object_level_masks, relation_level_masks


class LanguageMsgEncoder(nn.Module):
    def __init__(self, in_features, out_features, layer=3):
        super(LanguageMsgEncoder, self).__init__()
        self.embed_dim = in_features
        self.bias = True

        self.linear_relation_text = nn.Linear(
            2*self.embed_dim, self.embed_dim, bias=self.bias).to('cuda')
        self.tanh = nn.Tanh().to('cuda')
        self.layer = layer

        self.go = nn.Linear(in_features, out_features).to('cuda')
        self.gr = nn.Linear(3*in_features, out_features).to('cuda')
        self.esa = nn.Linear(in_features, out_features).to('cuda')

    def forward(self, text_embedding):
        text_level_features = text_embedding

        for i in range(self.layer):
            # udpate object level features
            new_text_level_features = self.tanh(
                self.go(text_level_features)).to('cuda')

            # udpate relation level features
            object_level_features_repeated = torch.unsqueeze(
                text_level_features, 2).repeat(1, 1, text_embedding.size(-2), 1)
            object_level_features_unsqueezed_transposed = torch.unsqueeze(
                text_level_features, 1).repeat(1, text_embedding.size(-2), 1, 1)
            rlf = torch.cat((object_level_features_repeated,
                            object_level_features_unsqueezed_transposed), dim=3).to('cuda')
            if (i == 0):
                relation_level_features = self.linear_relation_text(rlf)
            rlf_2 = torch.cat((rlf, relation_level_features), dim=3)
            rlf_3 = self.tanh(self.gr(rlf_2))
            relation_level_features = rlf_3

            text_level_features = new_text_level_features

        # esa
        relation_level_features = self.esa(relation_level_features)
        attn = nn.Softmax(dim=1)(relation_level_features -
                                 torch.max(relation_level_features, dim=1)[0].unsqueeze(1))
        relation_level_features = torch.sum(
            attn*relation_level_features, dim=-2)

        return text_level_features, relation_level_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralManifoldBlock(nn.Module):
    """
    谱图流形变换块 (Spectral Manifold Transform) - 数值稳定版本
    
    使用 Chebyshev 多项式近似避免显式特征分解:
    - 数值稳定
    - 支持梯度回传
    - 计算效率高
    
    数学原理:
    1. 构建相似度图: A_ij = exp(-||x_i - x_j||² / T)
    2. 归一化拉普拉斯: L = I - D^{-1/2} A D^{-1/2}
    3. Chebyshev 多项式滤波: g(L) ≈ Σ c_k T_k(λ)
    4. 投影到超球面
    """
    
    def __init__(self, num_chebyshev=8):
        super().__init__()
        
        # Chebyshev 多项式阶数
        self.num_chebyshev = num_chebyshev
        
        # ========== 可学习参数 ==========
        # Chebyshev 系数 (替代显式特征值滤波)
        self.cheb_coeffs_low = nn.Parameter(torch.ones(num_chebyshev) / num_chebyshev)
        self.cheb_coeffs_mid = nn.Parameter(torch.ones(num_chebyshev) / num_chebyshev)
        self.cheb_coeffs_high = nn.Parameter(torch.ones(num_chebyshev) / num_chebyshev)
        
        # 相似度图温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def build_similarity_graph(self, x):
        """构建自相似度图"""
        B, D, N = x.shape
        
        # 成对距离
        x_i = x.unsqueeze(3)
        x_j = x.unsqueeze(2)
        dist_sq = (x_i - x_j).pow(2).sum(dim=1)
        
        # 高斯核 (使用 ReLU 稳定 exp)
        temp = F.relu(self.temperature) + 0.1
        attention = torch.exp(-dist_sq / temp.unsqueeze(-1).unsqueeze(-1))
        
        # 自环 + 行归一化
        attention = attention + torch.eye(N, device=x.device).unsqueeze(0)
        row_sum = attention.sum(dim=2, keepdim=True) + 1e-6
        attention = attention / row_sum
        
        return attention
    
    def compute_normalized_laplacian(self, A):
        """计算归一化拉普拉斯 (缩放到 [-1, 1])"""
        B, N, _ = A.shape
        
        # 度矩阵
        D = A.sum(dim=2, keepdim=True)
        D_inv_sqrt = D.pow(-0.5).clamp(min=1e-6, max=1e6)
        D_diag = D_inv_sqrt * torch.eye(N, device=A.device)
        
        # L_sym = I - D^{-1/2} A D^{-1/2}
        L = torch.eye(N, device=A.device).unsqueeze(0) - torch.matmul(torch.matmul(D_diag, A), D_diag)
        L = (L + L.transpose(1, 2)) / 2  # 确保对称
        
        # 缩放到 [-1, 1] 以便 Chebyshev 近似
        L_max = L.abs().max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0].clamp(min=1e-6)
        L_scaled = L / L_max
        
        return L_scaled
    
    def chebyshev_polynomial(self, L, k):
        """计算 Chebyshev 多项式 T_k(L)"""
        # T_0 = I, T_1 = L, T_{k+1} = 2*L*T_k - T_{k-1}
        results = [torch.eye(L.shape[1], device=L.device).unsqueeze(0).expand_as(L)]
        
        if k == 0:
            return results[0]
        
        results.append(L.clone())
        
        for _ in range(2, k + 1):
            results.append(2 * torch.bmm(L, results[-1]) - results[-2])
        
        return results[k]
    
    def chebyshev_filter(self, x, L, coeffs):
        """Chebyshev 谱滤波"""
        B, D, N = x.shape
        K = len(coeffs)
        
        # x: [B, D, N] -> [B, N, D]
        x_t = x.transpose(1, 2)
        
        # 计算 T_k(L) @ x
        filtered = torch.zeros_like(x_t)
        
        for i, coeff in enumerate(coeffs):
            T_k = self.chebyshev_polynomial(L, i)
            # [B, N, N] @ [B, N, D] -> [B, N, D]
            filtered = filtered + coeff * torch.bmm(T_k, x_t)
        
        # [B, N, D] -> [B, D, N]
        return filtered.transpose(1, 2)
    
    def project_to_sphere(self, x):
        """投影到超球面"""
        norm = x.norm(p=2, dim=1, keepdim=True) + 1e-6
        return x / norm
    
    def forward(self, x):
        """
        x: [B, D, N] 输入特征
        返回: [B, D, N] 谱图流形变换后的特征
        """
        B, D, N = x.shape
        
        # 1. 构建相似度图
        A = self.build_similarity_graph(x)
        
        # 2. 计算归一化拉普拉斯
        L = self.compute_normalized_laplacian(A)
        
        # 3. Chebyshev 谱滤波 (Softmax 归一化系数)
        coeffs_low = F.softmax(self.cheb_coeffs_low, dim=0)
        coeffs_mid = F.softmax(self.cheb_coeffs_mid, dim=0)
        coeffs_high = F.softmax(self.cheb_coeffs_high, dim=0)
        
        # 三分支滤波
        filtered_low = self.chebyshev_filter(x, L, coeffs_low)
        filtered_mid = self.chebyshev_filter(x, L, coeffs_mid)
        filtered_high = self.chebyshev_filter(x, L, coeffs_high)
        
        # 4. 融合
        output = filtered_low + filtered_mid + filtered_high
        
        # 5. 投影到超球面
        output = self.project_to_sphere(output)
        
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, residual_in_fp32=False,
                 lstm_hidden_size=128):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.relu = nn.ReLU()

        # 定义3个1D卷积层，分别使用不同的kernel_size
        self.conv_layer_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)
        self.conv_layer_5 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0)

        self.conv_layer_lstm = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                             nn.Conv1d(128, 512, kernel_size=1, bias=False))

        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=2, batch_first=True)

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=256, batch_first=True, bidirectional=True)

        self.gru = nn.GRU(input_size=out_channels, hidden_size=256, batch_first=True, bidirectional=True)
        # 定义层归一化
        self.norm = nn.LayerNorm(out_channels)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=4,  # 使用1个头简化实现
            batch_first=True
        )

        # 线性层：将双向LSTM的输出映射到embed_dim
        # self.fc1 = nn.Linear(in_channels, in_channels)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),  # 批归一化
            nn.GELU(),  # 更平滑的激活函数
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(in_channels * 4, in_channels)
        )
        # self.fc2 = nn.Linear(2*in_channels, in_channels)
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels)
        )
        # self.fc3 = nn.Linear(in_channels, in_channels)
        self.mlp3 = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.BatchNorm1d(in_channels * 4),  # 批归一化
            nn.GELU(),  # 更平滑的激活函数
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(in_channels * 4, in_channels)
        )
        # 替换 conv_layer_lstm 为 Transformer 层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,  # 输入/输出维度（需匹配双向GRU的输出维度）
            nhead=4,  # 注意力头数（根据需求调整）
            dim_feedforward=256,  # 前馈网络隐藏层维度
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # 谱图流形变换块 (替代 FFT)
        self.spectral_manifold_block = SpectralManifoldBlock()

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        r"""Pass the input through the convolution block.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = ConvBlock(LN(residual))
        """
        hidden_states_residual = hidden_states
        # 1. 如果有残差连接，进行加和操作
        if residual is not None:
            residual = hidden_states + residual
        else:
            residual = hidden_states
        k_v = hidden_states
        # 3. 通过卷积层进行操作
        # hidden_states1 = self.conv_layer_1(hidden_states) + hidden_states
        batchsize = hidden_states.shape[0]
        cluster_number = hidden_states.shape[1]

        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        # 3. 通过卷积层进行操作
        hidden_states1 = self.conv_layer_1(hidden_states) + hidden_states
        hidden_states2 = self.conv_layer_3(hidden_states) + hidden_states
        hidden_states3 = self.conv_layer_5(hidden_states) + hidden_states

        # ========== 替换 FFT 为谱图流形变换 (SMT) ==========
        # hidden_states1: 低频/全局特征
        hidden_states1 = self.spectral_manifold_block(hidden_states1) + hidden_states
        # hidden_states2: 中频/中层特征
        hidden_states2 = self.spectral_manifold_block(hidden_states2) + hidden_states
        # hidden_states3: 高频/局部特征
        hidden_states3 = self.spectral_manifold_block(hidden_states3) + hidden_states

        hidden_states1 = hidden_states1.permute(0, 2, 1).contiguous()
        hidden_states2 = hidden_states2.permute(0, 2, 1).contiguous()
        hidden_states3 = hidden_states3.permute(0, 2, 1).contiguous()

        atten_states1, _ = self.attention(hidden_states1, hidden_states1, hidden_states1)
        atten_states1 = atten_states1 + hidden_states1
        atten_states2, _ = self.attention(hidden_states2, hidden_states2, hidden_states2)
        atten_states3, _ = self.attention(hidden_states3, hidden_states3, hidden_states3)
        atten_states3 = atten_states3 + hidden_states3

        cross_states1, _ = self.attention(atten_states2, atten_states1, atten_states1)
        cross_states2, _ = self.attention(atten_states2, atten_states3, atten_states3)
        fused_feature_map = cross_states1 * cross_states2
        fused_feature = fused_feature_map * atten_states2
        encodings_point_cloud = hidden_states_residual + fused_feature

        attn_output, attn_weights = self.cross_attention(
            query=k_v,
            key=encodings_point_cloud,
            value=encodings_point_cloud
        )
        attn_output = attn_output + encodings_point_cloud
        # 双向LSTM处理
        lstm_output, _ = self.gru(attn_output)

        # === 替换部分：用 Transformer 替代卷积操作 ===
        lstm_output_res = lstm_output
        lstm_output = self.transformer_layer(lstm_output)  # Transformer 处理
        lstm_output = lstm_output + lstm_output_res  # 残差连接

        indices = list(range(0, 512, 2))  # 这将生成 [0, 2, 4, ..., 510]

        narrowed_tensor = lstm_output[:, :, indices]

        return narrowed_tensor, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # 如果需要，可能用于推理阶段的缓存分配
        return None
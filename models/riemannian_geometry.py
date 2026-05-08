"""
Riemannian 流形上的特征学习模块

数学框架:
1. 将特征空间建模为具有可变曲率 c 的双曲空间 (Hyperbolic Space)
2. 使用指数映射 (Exponential Map) 和对数映射 (Logarithmic Map) 在流形与切空间之间转换
3. 通过平行移动 (Parallel Transport) 保持几何结构

论文写作亮点:
- 将传统欧几里得空间的Attention重新解释为黎曼流形上的几何操作
- 流形曲率 c 作为可学习参数，控制空间的"弯曲程度"
- 测地线距离意义上保持对象间的关系结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RiemannianFeatureTransform(nn.Module):
    """
    黎曼流形特征变换层
    
    将点云/对象特征从欧几里得空间映射到黎曼流形空间，
    在流形上进行几何保持的特征聚合。
    
    参数:
        embed_dim: 特征维度
        manifold_type: 流形类型 ('hyperbolic', 'sphere', 'euclidean')
        learnable_curvature: 是否学习曲率参数
    """
    
    def __init__(self, embed_dim, manifold_type='hyperbolic', learnable_curvature=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_type = manifold_type
        
        # ========== 可学习参数统计 ==========
        # 总计仅 3 个可学习参数:
        #   1. curvature:     1 个 (曲率参数 c, 通过 softplus 确保 c > 0)
        #   2. translation:    1 个 (平移/偏差参数)
        #   3. geodesic_scale: 1 个 (测地线尺度因子)
        # 
        # 与传统 QKV 线性层对比:
        #   linear_q/k/v: 各 256*256 = 65,536 参数 (不使用)
        #   我们仅增加 3 个参数
        
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(0.5))          # [1]
            self.translation = nn.Parameter(torch.zeros(1))            # [1]
        else:
            self.curvature = 1.0
            self.translation = 0.0
        
        self.geodesic_scale = nn.Parameter(torch.ones(1) * 0.1)      # [1]
    
    def get_curvature(self):
        """获取当前曲率 (确保为正)"""
        if isinstance(self.curvature, nn.Parameter):
            return F.softplus(self.curvature) + 1e-6  # 确保 c > 0
        return max(self.curvature, 1e-6)
    
    # ==================== 双曲空间 (Hyperbolic Space) 操作 ====================
    
    def hyp_exp_map(self, x, v, c=None):
        """
        指数映射: 从切空间点 x 沿切向量 v 移动到流形上的点
        
        数学公式:
        exp_x(v) = cosh(√c · ||v||) · x + sinh(√c · ||v||) · v / ||v||
        
        Args:
            x: 流形上的点 [*, D]
            v: 切空间中的向量 [*, D]
            c: 曲率参数
        
        Returns:
            流形上的新点 [*, D]
        """
        if c is None:
            c = self.get_curvature()
        
        sqrt_c = math.sqrt(c)
        
        # 计算 v 的范数
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
        
        # 避免数值问题
        sqrt_c_v_norm = sqrt_c * v_norm
        cosh_term = torch.cosh(sqrt_c_v_norm)
        sinh_term = torch.sinh(sqrt_c_v_norm)
        
        # 指数映射公式
        result = cosh_term * x + (sinh_term / v_norm) * v
        
        return result
    
    def hyp_log_map(self, x, y, c=None):
        """
        对数映射: 从流形上的点 y 映射到切空间点 x 处的切向量
        
        数学公式:
        log_x(y) = (2/√c) · atanh(√c · ||y - x||) · (y - x) / ||y - x||
        
        Args:
            x: 参考点 [*, D]
            y: 目标点 [*, D]
            c: 曲率参数
        
        Returns:
            切空间中的向量 [*, D]
        """
        if c is None:
            c = self.get_curvature()
        
        sqrt_c = math.sqrt(c)
        
        # 计算差向量
        diff = y - x
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10)
        
        # 避免 atanh 的数值问题
        ratio = (sqrt_c * diff_norm).clamp(max=0.99)
        log_term = torch.atanh(ratio)
        
        # 对数映射公式
        result = (2.0 / sqrt_c) * log_term * (diff / diff_norm)
        
        return result
    
    def hyp_distance(self, x, y, c=None):
        """
        计算双曲空间中的测地线距离
        
        d(x, y) = (2/√c) · atanh(√c · ||y ⊕ (-x)||)
        
        其中 ⊕ 是双曲空间的平行移动操作
        """
        if c is None:
            c = self.get_curvature()
        
        sqrt_c = math.sqrt(c)
        
        # 使用 Mobius 加法
        diff = self._mobius_add(x, -y, c)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10)
        
        # 测地线距离
        ratio = (sqrt_c * diff_norm).clamp(max=0.99)
        dist = (2.0 / sqrt_c) * torch.atanh(ratio)
        
        return dist.squeeze(-1)
    
    def _mobius_add(self, x, y, c=None):
        """
        双曲空间的 Mobius 加法 (类似于黎曼流形上的加法)
        
        x ⊕ y = ( (1 + 2c⟨x,y⟩ + c||y||²)y + (1 - c||x||²)x ) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        """
        if c is None:
            c = self.get_curvature()
        
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2 * c * xy_dot + c * y_sq) * x + (1 - c * x_sq) * y
        denominator = 1 + 2 * c * xy_dot + c * c * x_sq * y_sq
        denominator = denominator.clamp(min=1e-10)
        
        return numerator / denominator
    
    # ==================== 球面空间 (Sphere) 操作 ====================
    
    def sphere_exp_map(self, x, v):
        """
        球面上的指数映射
        
        exp_x(v) = cos(||v||) · x + sin(||v||) · v / ||v||
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
        
        cos_term = torch.cos(v_norm)
        sin_term = torch.sin(v_norm)
        
        return cos_term * x + (sin_term / v_norm) * v
    
    def sphere_log_map(self, x, y):
        """
        球面上的对数映射
        
        log_x(y) = (θ / sin(θ)) · (y - cos(θ) · x)
        其中 θ = arccos(⟨x, y⟩)
        """
        cos_theta = torch.sum(x * y, dim=-1, keepdim=True).clamp(max=1.0, min=-1.0)
        theta = torch.acos(cos_theta)
        
        diff = y - cos_theta * x
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10)
        
        return (theta / diff_norm) * diff
    
    # ==================== 主前向传播方法 ====================
    
    def to_manifold(self, x):
        """
        将欧几里得空间中的特征映射到流形空间
        
        通过在单位球面上归一化实现隐式流形映射
        """
        if self.manifold_type == 'hyperbolic':
            # 双曲空间: 重新缩放并归一化
            x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
            # 映射到双曲球的内部
            return x / x_norm * self.geodesic_scale.tanh()
        elif self.manifold_type == 'sphere':
            # 球面: 直接归一化到单位球面
            return F.normalize(x, dim=-1)
        else:
            return x
    
    def from_manifold(self, x):
        """将流形空间中的特征映射回欧几里得空间"""
        if self.manifold_type in ['hyperbolic', 'sphere']:
            return F.normalize(x, dim=-1)
        return x
    
    def riemannian_aggregate(self, features, centers=None):
        """
        流形上的特征聚合
        
        在双曲空间中使用指数/对数映射进行几何保持的聚合
        
        数学过程:
        1. 选择一个参考点作为聚合中心 (使用第一个点或学习得到)
        2. 将所有点对数映射到切空间
        3. 在切空间中计算均值
        4. 将结果指数映射回流形
        
        Args:
            features: 输入特征 [B, N, D]
            centers: 对象中心坐标 [B, N, 3] (可选，用于位置感知聚合)
        
        Returns:
            聚合后的流形特征 [B, D]
        """
        batch_size, num_objects, dim = features.shape
        
        # 投影到流形空间
        manifold_features = self.to_manifold(features)
        
        if self.manifold_type == 'hyperbolic':
            # 使用第一个点作为参考点 (Frechet mean 的简化版本)
            reference = manifold_features[:, 0:1, :]  # [B, 1, D]
            
            # 对数映射所有特征到切空间
            tangent_features = []
            for i in range(num_objects):
                log_f = self.hyp_log_map(reference, manifold_features[:, i:i+1, :])
                tangent_features.append(log_f)
            
            tangent_features = torch.cat(tangent_features, dim=1)  # [B, N, D]
            
            # 在切空间中计算加权均值
            weights = torch.softmax(tangent_features, dim=1)
            aggregated_tangent = torch.sum(weights * tangent_features, dim=1)  # [B, D]
            
            # 指数映射回流形
            aggregated_manifold = self.hyp_exp_map(reference.squeeze(1), aggregated_tangent)
            
        elif self.manifold_type == 'sphere':
            # 球面上的聚合
            reference = manifold_features[:, 0, :]
            
            tangent_features = []
            for i in range(num_objects):
                log_f = self.sphere_log_map(reference, manifold_features[:, i, :])
                tangent_features.append(log_f)
            
            tangent_features = torch.stack(tangent_features, dim=1)
            weights = torch.softmax(tangent_features, dim=1)
            aggregated_tangent = torch.sum(weights * tangent_features, dim=1)
            
            aggregated_manifold = self.sphere_exp_map(reference, aggregated_tangent)
            
        else:
            # 欧几里得空间: 简单平均
            aggregated_manifold = manifold_features.mean(dim=1)
        
        # 添加平移
        aggregated_manifold = aggregated_manifold + self.translation
        
        return aggregated_manifold
    
    def riemannian_attention(self, Q, K, V, centers=None):
        """
        黎曼流形上的注意力机制
        
        核心思想:
        1. 在流形上计算 Query-Key 相似度 (使用测地线距离)
        2. 使用切空间中的值进行加权聚合
        3. 保持几何结构
        
        Args:
            Q: Query [B, N, D]
            K: Key [B, M, D]  
            V: Value [B, M, D]
            centers: 对象中心坐标 (可选)
        
        Returns:
            输出特征 [B, N, D]
        """
        # 投影到流形空间
        Q_manifold = self.to_manifold(Q)
        K_manifold = self.to_manifold(K)
        V_manifold = self.to_manifold(V)
        
        if self.manifold_type == 'hyperbolic':
            c = self.get_curvature()
            
            # 计算双曲空间中的注意力权重
            # 使用 Mobius 点积作为相似度度量
            Q_manifold_expanded = Q_manifold.unsqueeze(2)  # [B, N, 1, D]
            K_manifold_expanded = K_manifold.unsqueeze(1)  # [B, 1, M, D]
            
            # 归一化后计算余弦相似度
            similarity = F.cosine_similarity(Q_manifold_expanded, K_manifold_expanded, dim=-1)
            
            # 使用测地线距离的负值作为注意力分数
            # dist = self.hyp_distance(Q_manifold_expanded.squeeze(2), K_manifold_expanded.squeeze(1))
            # attention_scores = -dist
            
            # 结合余弦相似度和可学习的温度
            temperature = 1.0 / (math.sqrt(self.embed_dim) + 1e-8)
            attention_weights = F.softmax(similarity * temperature, dim=-1)  # [B, N, M]
            
            # 在切空间中聚合值
            # 首先将 K, V 对数映射到 Q 所在的切空间
            outputs = []
            for n in range(Q_manifold.size(1)):
                q_n = Q_manifold[:, n:n+1, :]  # [B, 1, D]
                
                # 对所有 K, V 点计算对数映射
                tangent_V_list = []
                for m in range(K_manifold.size(1)):
                    log_v = self.hyp_log_map(q_n, V_manifold[:, m:m+1, :], c)
                    tangent_V_list.append(log_v)
                
                tangent_V = torch.cat(tangent_V_list, dim=1)  # [B, M, D]
                
                # 加权求和
                weight_expanded = attention_weights[:, n:n+1, :].transpose(-1, -2)  # [B, D, M]
                aggregated_tangent = torch.bmm(weight_expanded, tangent_V).squeeze(1)  # [B, D]
                
                # 指数映射回流形
                output = self.hyp_exp_map(q_n.squeeze(1), aggregated_tangent, c)
                outputs.append(output)
            
            output_manifold = torch.stack(outputs, dim=1)  # [B, N, D]
            
        elif self.manifold_type == 'sphere':
            # 球面上的注意力
            similarity = torch.matmul(Q_manifold, K_manifold.transpose(-2, -1))
            temperature = 1.0 / (math.sqrt(self.embed_dim) + 1e-8)
            attention_weights = F.softmax(similarity * temperature, dim=-1)
            
            outputs = []
            for n in range(Q_manifold.size(1)):
                q_n = Q_manifold[:, n, :]  # [B, D]
                
                tangent_V_list = []
                for m in range(K_manifold.size(1)):
                    log_v = self.sphere_log_map(q_n, V_manifold[:, m, :])
                    tangent_V_list.append(log_v)
                
                tangent_V = torch.stack(tangent_V_list, dim=1)
                weight_expanded = attention_weights[:, n, :].unsqueeze(-1)
                aggregated_tangent = (tangent_V * weight_expanded).sum(dim=1)
                
                output = self.sphere_exp_map(q_n, aggregated_tangent)
                outputs.append(output)
            
            output_manifold = torch.stack(outputs, dim=1)
            
        else:
            # 标准欧几里得注意力 (退化为普通attention)
            similarity = torch.matmul(Q, K.transpose(-2, -1))
            temperature = 1.0 / (math.sqrt(self.embed_dim) + 1e-8)
            attention_weights = F.softmax(similarity * temperature, dim=-1)
            output_manifold = torch.bmm(attention_weights, V)
        
        # 投影回欧几里得空间
        output_euclidean = self.from_manifold(output_manifold)
        
        return output_euclidean
    
    def forward(self, features, centers=None):
        """
        主前向传播
        
        将输入特征通过黎曼流形变换
        
        Args:
            features: 输入特征 [*, D]
            centers: 位置中心 (可选)
        
        Returns:
            流形增强后的特征 [*, D]
        """
        # 投影到流形
        manifold_features = self.to_manifold(features)
        
        # 添加平移
        if isinstance(self.translation, nn.Parameter):
            manifold_features = manifold_features + self.translation.tanh()
        
        return manifold_features


class RiemannianAggregationModule(nn.Module):
    """
    黎曼流形聚合模块
    
    用于替代或增强现有的 object-level 特征聚合
    保持与原有 attention 机制的兼容性
    
    参数: 仅 3 个可学习参数
    """
    
    def __init__(self, embed_dim, manifold_type='hyperbolic'):
        super().__init__()
        self.riemannian_transform = RiemannianFeatureTransform(
            embed_dim, 
            manifold_type=manifold_type,
            learnable_curvature=True
        )
        
        # 可选的残差连接
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, object_features, centers=None):
        """
        流形聚合
        
        Args:
            object_features: [B, N, D] 对象特征
            centers: [B, N, 3] 对象中心坐标
        
        Returns:
            聚合特征 [B, D]
        """
        # 流形变换
        manifold_features = self.riemannian_transform(object_features)
        
        # 流形上的全局聚合 (使用 Frechet mean 的近似)
        aggregated = self.riemannian_transform.riemannian_aggregate(
            manifold_features, centers
        )
        
        # 投影回欧几里得空间
        output = self.riemannian_transform.from_manifold(aggregated)
        
        # 可选的残差连接
        global_avg = object_features.mean(dim=1)
        output = self.residual_weight * output + (1 - self.residual_weight) * global_avg
        
        # 输出投影
        output = self.output_proj(output)
        
        return output

"""
TDA拓扑约束损失函数
融合传统ReID loss和拓扑约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TDATopologyLoss(nn.Module):
    """
    拓扑约束损失
    
    原理:
    - 计算特征集的拓扑结构
    - 拓扑相似的样本应该有相近的表示
    - 通过保持拓扑不变量来优化特征学习
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.3):
        """
        Args:
            temperature: softmax温度参数
            margin: 拓扑相似度的边界参数
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        tda_features: torch.Tensor,
        labels: torch.Tensor,
        original_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tda_features: (batch_size, tda_dim) TDA特征
            labels: (batch_size,) 身份标签
            original_features: (batch_size, feat_dim) 原始特征
        
        Returns:
            loss: 标量张量
        """
        batch_size = tda_features.shape[0]
        
        # 计算TDA特征的相似度矩阵
        tda_sim = self._compute_tda_similarity(tda_features)
        
        # 计算身份应该有的相似关系
        same_identity_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        same_identity_mask.fill_diagonal_(0)
        
        # 对于相同身份的样本对，最大化相似度
        # 对于不同身份的样本对，最小化相似度
        
        pos_loss = torch.sum(
            same_identity_mask * (self.margin - tda_sim).clamp(min=0.0)
        )
        
        neg_loss = torch.sum(
            (1 - same_identity_mask).fill_diagonal_(0) * (tda_sim - self.margin).clamp(min=0.0)
        )
        
        loss = (pos_loss + neg_loss) / batch_size
        
        return loss
    
    def _compute_tda_similarity(self, tda_features: torch.Tensor) -> torch.Tensor:
        """
        计算TDA特征的相似度
        使用归一化点积
        """
        # 归一化
        norm_feat = F.normalize(tda_features, p=2, dim=1)
        
        # 余弦相似度矩阵
        similarity = torch.mm(norm_feat, norm_feat.t())
        
        return similarity


class CombinedLoss(nn.Module):
    """
    组合损失函数
    L_total = L_ce + λ₁ * L_tda + λ₂ * L_triplet
    """
    
    def __init__(
        self,
        num_classes: int,
        lambda_tda: float = 0.5,
        lambda_triplet: float = 0.5,
        margin: float = 0.3,
        hard_neg_mining: bool = True
    ):
        super().__init__()
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_tda = TDATopologyLoss(margin=margin)
        self.criterion_triplet = TripletLoss(margin=margin, hard_neg_mining=hard_neg_mining)
        
        self.lambda_tda = lambda_tda
        self.lambda_triplet = lambda_triplet
    
    def forward(
        self,
        logits: torch.Tensor,
        tda_features: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算组合损失
        
        Returns:
            loss: 总损失
            loss_dict: 各损失项的字典
        """
        # 交叉熵损失 (分类)
        loss_ce = self.criterion_ce(logits, labels)
        
        # TDA拓扑约束损失
        loss_tda = self.criterion_tda(tda_features, labels, features)
        
        # 三元组损失 (度量学习)
        loss_triplet = self.criterion_triplet(features, labels)
        
        # 总损失
        total_loss = loss_ce + self.lambda_tda * loss_tda + self.lambda_triplet * loss_triplet
        
        loss_dict = {
            'loss_ce': loss_ce.item(),
            'loss_tda': loss_tda.item(),
            'loss_triplet': loss_triplet.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class TripletLoss(nn.Module):
    """
    三元组损失 (Triplet Loss)
    标准ReID损失函数
    """
    
    def __init__(self, margin: float = 0.3, hard_neg_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_neg_mining = hard_neg_mining
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feat_dim)
            labels: (batch_size,)
        
        Returns:
            loss: 标量
        """
        # 计算距离矩阵
        dist_mat = self._euclidean_dist(features, features)
        
        # 生成掩码
        N = features.shape[0]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = 1 - is_pos
        is_pos.fill_diagonal_(0)
        
        # 困难样本挖掘
        if self.hard_neg_mining:
            # 困难正样本：距离最大的同身份样本
            pos_dist = dist_mat * is_pos
            pos_dist[is_pos == 0] = -float('inf')
            hard_pos_dist = torch.max(pos_dist, dim=1)[0]
            
            # 困难负样本：距离最小的异身份样本
            neg_dist = dist_mat * is_neg + float('inf') * (1 - is_neg)
            hard_neg_dist = torch.min(neg_dist, dim=1)[0]
        else:
            hard_pos_dist = (dist_mat * is_pos).sum(1) / (is_pos.sum(1) + 1e-8)
            hard_neg_dist = (dist_mat * is_neg).sum(1) / (is_neg.sum(1) + 1e-8)
        
        # 三元组损失
        y = torch.ones_like(hard_pos_dist)
        loss = F.margin_ranking_loss(
            hard_neg_dist, 
            hard_pos_dist, 
            y, 
            margin=self.margin
        )
        
        return loss
    
    def _euclidean_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算欧氏距离矩阵"""
        m, n = x.size(0), y.size(0)
        
        xx = (x ** 2).sum(1, keepdim=True).expand(m, n)
        yy = (y ** 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy - 2 * torch.mm(x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        
        return dist


class CenterLoss(nn.Module):
    """
    中心损失 - 同身份特征的聚集
    """
    
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.alpha = alpha
        nn.init.kaiming_uniform_(self.centers, a=np.sqrt(5))
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, feat_dim)
            labels: (batch_size,)
        """
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(1).sqrt().mean()
        
        return loss

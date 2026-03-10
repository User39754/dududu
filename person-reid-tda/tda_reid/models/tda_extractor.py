"""
TDA Feature Extractor
计算特征点云的拓扑不变量（持久同调）
"""

import numpy as np
import torch
import torch.nn as nn
from ripser import ripser
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TDAExtractor(nn.Module):
    """
    持久同调特征提取器
    
    功能:
    - 计算点云的持久同调
    - 提取拓扑不变量 (H0, H1等)
    - 生成拓扑特征向量
    """
    
    def __init__(
        self, 
        max_dim: int = 1,
        max_features: int = 50,
        normalize: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            max_dim: 最大同调维数 (0=点, 1=环, 2=空洞)
            max_features: 保留的最多拓扑特征数
            normalize: 是否归一化特征
            device: 计算设备
        """
        super().__init__()
        self.max_dim = max_dim
        self.max_features = max_features
        self.normalize = normalize
        self.device = device
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch_size, feature_dim) 或 (feature_dim,)
        输出: (batch_size, tda_feature_dim) 拓扑特征向量
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        batch_size = features.shape[0]
        tda_features = []
        
        for i in range(batch_size):
            feat = features[i].detach().cpu().numpy()
            tda_feat = self._compute_tda_features(feat)
            tda_features.append(tda_feat)
        
        tda_features = torch.tensor(
            np.array(tda_features), 
            dtype=features.dtype,
            device=features.device
        )
        
        return tda_features
    
    def _compute_tda_features(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        计算单个特征向量的持久同调特征
        
        Args:
            point_cloud: (D,) 特征向量，视为高维点云中的一个点
                        实际应该是多个点，这里简化为单点扩展
        
        Returns:
            tda_features: (K,) 拓扑特征向量
        """
        try:
            # 方法1: 考虑特征维度中的"邻域结构"
            # 将特征向量分解为多个子向量，构造点云
            point_cloud = point_cloud.reshape(-1, 1)
            
            # 使用Ripser计算持久同调
            result = ripser(point_cloud, maxdim=self.max_dim)
            
            # 提取拓扑特征
            tda_feat = self._extract_persistence_features(result)
            
            return tda_feat
            
        except Exception as e:
            # 如果计算失败，返回原始特征
            print(f"TDA计算失败: {e}, 使用备选方案")
            return self._fallback_tda_features(point_cloud.flatten())
    
    def _extract_persistence_features(self, ripser_result: Dict) -> np.ndarray:
        """
        从Ripser结果提取拓扑特征向量
        
        持久特征包括:
        - 各维同调的持久度统计
        - death-birth差值
        - 拓扑特征的数量
        """
        features = []
        
        for dim in range(self.max_dim + 1):
            if f'dgms' in ripser_result and dim < len(ripser_result['dgms']):
                dgm = ripser_result['dgms'][dim]
                
                if len(dgm) > 0:
                    # 计算持久度 (persistence = death - birth)
                    persistence = dgm[:, 1] - dgm[:, 0]
                    
                    # 移除无穷大
                    persistence = persistence[np.isfinite(persistence)]
                    
                    if len(persistence) > 0:
                        # 统计特征
                        features.extend([
                            np.sum(persistence),           # 总持久度
                            np.mean(persistence),          # 平均持久度
                            np.std(persistence),           # 持久度标准差
                            np.max(persistence),           # 最大持久度
                            len(persistence),              # 特征数量
                        ])
                    else:
                        features.extend([0, 0, 0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0, 0])
        
        tda_feat = np.array(features, dtype=np.float32)
        
        # 归一化
        if self.normalize and np.any(tda_feat != 0):
            tda_feat = tda_feat / (np.linalg.norm(tda_feat) + 1e-8)
        
        # 截断到max_features
        if len(tda_feat) > self.max_features:
            tda_feat = tda_feat[:self.max_features]
        else:
            # 补零到max_features维
            tda_feat = np.pad(tda_feat, (0, self.max_features - len(tda_feat)))
        
        return tda_feat
    
    def _fallback_tda_features(self, feat: np.ndarray) -> np.ndarray:
        """
        备选方案：使用简单的统计特征替代TDA
        """
        fallback_feat = np.array([
            np.sum(feat),
            np.mean(feat),
            np.std(feat),
            np.max(feat),
            np.min(feat),
        ], dtype=np.float32)
        
        # 补零
        if len(fallback_feat) < self.max_features:
            fallback_feat = np.pad(fallback_feat, (0, self.max_features - len(fallback_feat)))
        
        return fallback_feat
    
    def compute_persistence_diagram(self, features: torch.Tensor) -> Dict:
        """
        计算完整的持久性图表（用于可视化）
        
        Returns:
            Dict: 包含各维同调的持久图表
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        feat = features[0].detach().cpu().numpy()
        point_cloud = feat.reshape(-1, 1)
        
        result = ripser(point_cloud, maxdim=self.max_dim)
        return result
    
    def get_output_dim(self) -> int:
        """返回TDA特征向量维度"""
        return self.max_features


class TDABackbone(nn.Module):
    """
    TDA增强的特征提取骨干
    融合CNN特征和拓扑特征
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        tda_dim: int = 50,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.tda_extractor = TDAExtractor(
            max_dim=1,
            max_features=tda_dim
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim + tda_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )
        
    def forward(self, cnn_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cnn_features: (batch_size, feature_dim) CNN特征
        
        Returns:
            fused_features: (batch_size, feature_dim) 融合特征
            tda_features: (batch_size, tda_dim) 拓扑特征
        """
        # 计算TDA特征
        tda_feat = self.tda_extractor(cnn_features)
        
        # 融合
        combined = torch.cat([cnn_features, tda_feat], dim=1)
        fused_features = self.fusion_net(combined)
        
        return fused_features, tda_feat

"""
TDA-Enhanced ReID Model
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .tda_extractor import TDABackbone


class TDAReIDModel(nn.Module):
    """
    融合TDA的行人重识别模型
    
    架构:
    Input Image -> CNN Backbone -> TDA Enhancement -> Classification/Metric Head
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 2048,
        tda_dim: int = 50,
        pretrained_backbone: Optional[nn.Module] = None
    ):
        """
        Args:
            num_classes: 行人ID类别数
            feature_dim: 特征维度
            tda_dim: TDA特征维度
            pretrained_backbone: 预训练的CNN骨干 (如ResNet-50)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.tda_dim = tda_dim
        
        # CNN骨干
        if pretrained_backbone is not None:
            self.backbone = pretrained_backbone
        else:
            from torchvision import models
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # TDA增强模块
        self.tda_backbone = TDABackbone(
            feature_dim=feature_dim,
            tda_dim=tda_dim
        )
        
        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # 特征归一化
        self.feat_bn = nn.BatchNorm1d(feature_dim)
        self.feat_bn.bias.requires_grad_(False)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_tda_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, 3, H, W) 输入图像
            return_tda_features: 是否返回TDA特征
        
        Returns:
            Dict包含:
            - 'features': 融合特征
            - 'logits': 分类logits
            - 'tda_features': (可选) 拓扑特征
            - 'orig_features': 原始CNN特征
        """
        # CNN特征提取
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)  # flatten
        
        # TDA增强
        fused_feat, tda_feat = self.tda_backbone(feat)
        
        # 特征归一化
        feat_normed = self.feat_bn(fused_feat)
        
        # 分类
        logits = self.classifier(feat_normed)
        
        output = {
            'features': feat_normed,
            'logits': logits,
            'orig_features': feat.detach(),
        }
        
        if return_tda_features:
            output['tda_features'] = tda_feat
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅返回融合特征 (用于推理)
        """
        output = self.forward(x)
        return output['features']


class TDAReIDModelLightweight(nn.Module):
    """
    轻量级版本（用于快速实验）
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 1024,
        tda_dim: int = 30
    ):
        super().__init__()
        
        # 使用MobileNet作为骨干
        from torchvision import models
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        
        self.feature_dim = feature_dim
        
        # 特征投影
        self.feat_projection = nn.Linear(1280, feature_dim)
        
        # TDA增强
        self.tda_backbone = TDABackbone(
            feature_dim=feature_dim,
            tda_dim=tda_dim
        )
        
        # 分类
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feat_bn = nn.BatchNorm1d(feature_dim)
        self.feat_bn.bias.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.feat_projection(feat)
        
        fused_feat, tda_feat = self.tda_backbone(feat)
        feat_normed = self.feat_bn(fused_feat)
        logits = self.classifier(feat_normed)
        
        return {
            'features': feat_normed,
            'logits': logits,
            'tda_features': tda_feat,
        }
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        return output['features']

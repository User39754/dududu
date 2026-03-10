"""
主训练脚本
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tda_reid.models.reid_model import TDAReIDModel, TDAReIDModelLightweight
from tda_reid.losses.tda_loss import CombinedLoss, TripletLoss, CenterLoss
from tda_reid.metrics.reid_metrics import ReIDEvaluator, MetricsLogger
from tda_reid.datasets.data_loader import get_dataloader


class Trainer:
    """
    训练器类
    """
    
    def __init__(self, cfg: dict, device: str = 'cuda'):
        self.cfg = cfg
        self.device = device
        
        # 构建模型
        self.model = self._build_model()
        self.model.to(device)
        
        # 构建损失函数
        self.loss_fn = CombinedLoss(
            num_classes=cfg['num_classes'],
            lambda_tda=cfg.get('lambda_tda', 0.5),
            lambda_triplet=cfg.get('lambda_triplet', 0.5),
            margin=cfg.get('margin', 0.3)
        )
        self.loss_fn.to(device)
        
        # 优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.get('learning_rate', 1e-4),
            weight_decay=cfg.get('weight_decay', 5e-4)
        )
        
        # 学习率调度
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.get('lr_milestones', [30, 60]),
            gamma=cfg.get('lr_gamma', 0.1)
        )
        
        # 评估器
        self.evaluator = ReIDEvaluator()
        self.metrics_logger = MetricsLogger()
        
        # 日志目录
        self.log_dir = Path(cfg.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
    def _build_model(self):
        """构建模型"""
        if self.cfg.get('lightweight', False):
            model = TDAReIDModelLightweight(
                num_classes=self.cfg['num_classes'],
                feature_dim=self.cfg.get('feature_dim', 1024),
                tda_dim=self.cfg.get('tda_dim', 30)
            )
        else:
            from torchvision import models
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            model = TDAReIDModel(
                num_classes=self.cfg['num_classes'],
                feature_dim=self.cfg.get('feature_dim', 2048),
                tda_dim=self.cfg.get('tda_dim', 50),
                pretrained_backbone=backbone
            )
        return model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        训练主循环
        """
        epochs = self.cfg.get('epochs', 150)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 学习率更新
            self.lr_scheduler.step()
            
            # 验证阶段
            if (epoch + 1) % self.cfg.get('eval_freq', 10) == 0:
                metrics = self._evaluate(val_loader)
                self.metrics_logger.update(metrics)
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val mAP: {metrics.get('mAP', 0.0):.4f}")
                print(f"  Val Rank-1: {metrics.get('CMC@1', 0.0):.4f}")
                
                # 保存最佳模型
                if metrics.get('mAP', 0.0) >= self.metrics_logger.get_best('mAP'):
                    self._save_checkpoint(f'{self.log_dir}/best_model.pth')
            else:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}")
        
        print("\nTraining completed!")
        print("\n=" * 50)
        self.metrics_logger.print_summary()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        训练单个epoch
        """
        self.model.train()
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', ncols=80)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            output = self.model(images, return_tda_features=True)
            
            # 计算损失
            loss, loss_dict = self.loss_fn(
                output['logits'],
                output['tda_features'],
                output['features'],
                labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return total_loss / len(train_loader)
    
    def _evaluate(self, val_loader: DataLoader) -> dict:
        """
        验证阶段
        """
        self.model.eval()
        
        all_feats = []
        all_labels = []
        all_cams = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating', ncols=80):
                images = batch['image'].to(self.device)
                labels = batch['label'].numpy()
                cams = batch.get('camera_id', np.zeros_like(labels)).numpy()
                
                # 获取特征
                features = self.model.get_features(images)
                features = features.cpu().numpy()
                
                all_feats.append(features)
                all_labels.append(labels)
                all_cams.append(cams)
        
        all_feats = np.concatenate(all_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_cams = np.concatenate(all_cams, axis=0)
        
        # 简化: 使用全部数据作为gallery和query
        # 在实际应用中应分离query和gallery
        metrics = self.evaluator.evaluate(
            all_feats, all_labels, all_cams,
            all_feats, all_labels, all_cams
        )
        
        return metrics
    
    def _save_checkpoint(self, save_path: str):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg
        }, save_path)
        print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_market.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建trainer
    trainer = Trainer(cfg, device=device)
    
    # 获取数据加载器
    # train_loader, val_loader = get_dataloader(cfg)
    
    # 临时示例
    print("Note: Data loader needs to be configured for your dataset")
    print("Current setup: placeholder training loop")


if __name__ == '__main__':
    import numpy as np
    main()

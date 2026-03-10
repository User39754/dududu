"""
Person Re-ID 评估演示脚本
输入: 一组照片（查询集和图库集）
输出: Rank-1 准确率和 mAP 值
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tda_reid.datasets.data_loader import ReIDDataset
from tda_reid.models.reid_model import TDAReIDModel
from tda_reid.utils.config import ConfigManager
from tda_reid.metrics.reid_metrics import ReIDEvaluator


class ReIDEvaluator:
    """
    Person Re-ID 评估器
    """

    def __init__(self, config_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化评估器

        Args:
            config_path: 配置文件路径
            device: 计算设备
        """
        self.device = device
        # 加载配置
        config_manager = ConfigManager(config_path)
        self.config = config_manager.config

        # 加载模型
        self.model = TDAReIDModel(self.config)
        self.model.to(device)
        self.model.eval()

        # 加载检查点（如果存在）
        checkpoint_path = self.config.get('checkpoint_path', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载检查点: {checkpoint_path}")
        else:
            print("使用随机初始化的模型")

        self.metrics = ReIDEvaluator()

    def load_images_from_folder(self, folder_path: str) -> Tuple[List[str], torch.Tensor]:
        """
        从文件夹加载图像

        Args:
            folder_path: 图像文件夹路径

        Returns:
            image_paths: 图像路径列表
            features: 提取的特征张量 (N, feature_dim)
        """
        from PIL import Image
        import torchvision.transforms as transforms

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_paths = []
        features = []

        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    try:
                        # 加载和预处理图像
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(self.device)

                        # 提取特征
                        with torch.no_grad():
                            feature = self.model(image_tensor)
                            features.append(feature.cpu())

                        image_paths.append(img_path)
                        print(f"处理图像: {img_path}")

                    except Exception as e:
                        print(f"处理图像失败 {img_path}: {e}")
                        continue

        if not features:
            raise ValueError(f"在 {folder_path} 中未找到有效图像")

        features = torch.cat(features, dim=0)
        return image_paths, features

    def evaluate(self, query_folder: str, gallery_folder: str) -> Dict[str, float]:
        """
        评估 Re-ID 性能

        Args:
            query_folder: 查询图像文件夹
            gallery_folder: 图库图像文件夹

        Returns:
            results: 包含 rank1 和 mAP 的字典
        """
        print("加载查询图像...")
        query_paths, query_features = self.load_images_from_folder(query_folder)

        print("加载图库图像...")
        gallery_paths, gallery_features = self.load_images_from_folder(gallery_folder)

        print(f"查询图像数量: {len(query_paths)}")
        print(f"图库图像数量: {len(gallery_paths)}")

        # 简化评估：假设文件夹名表示身份，相机ID为0
        query_ids = [os.path.basename(os.path.dirname(path)) for path in query_paths]
        gallery_ids = [os.path.basename(os.path.dirname(path)) for path in gallery_paths]
        query_cams = [0] * len(query_ids)
        gallery_cams = [0] * len(gallery_ids)

        # 计算指标
        print("计算评估指标...")
        results = self.metrics.evaluate(
            query_features.numpy(),
            np.array(query_ids),
            np.array(query_cams),
            gallery_features.numpy(),
            np.array(gallery_ids),
            np.array(gallery_cams),
            metric='euclidean'
        )

        return results

    def _compute_distance_matrix(self, query_features: torch.Tensor,
                                gallery_features: torch.Tensor) -> np.ndarray:
        """
        计算查询和图库特征之间的距离矩阵

        Args:
            query_features: (N, D) 查询特征
            gallery_features: (M, D) 图库特征

        Returns:
            distmat: (N, M) 距离矩阵
        """
        # L2 距离
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        gallery_features = gallery_features / gallery_features.norm(dim=1, keepdim=True)

        distmat = torch.cdist(query_features, gallery_features, p=2).cpu().numpy()
        return distmat


def main():
    parser = argparse.ArgumentParser(description='Person Re-ID 评估演示')
    parser.add_argument('--config', type=str, default='../configs/config_market.yaml',
                       help='配置文件路径')
    parser.add_argument('--query_folder', type=str, default='../data/query_images',
                       help='查询图像文件夹路径')
    parser.add_argument('--gallery_folder', type=str, default='../data/gallery_images',
                       help='图库图像文件夹路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')

    args = parser.parse_args()

    # 创建必要的文件夹
    device_folder = '../data/device'
    os.makedirs(device_folder, exist_ok=True)
    os.makedirs(args.query_folder, exist_ok=True)
    os.makedirs(args.gallery_folder, exist_ok=True)

    print(f"创建文件夹: {device_folder}")
    print(f"查询文件夹: {args.query_folder}")
    print(f"图库文件夹: {args.gallery_folder}")

    # 检查文件夹是否存在图像
    if not os.path.exists(args.query_folder) or not any(os.scandir(args.query_folder)):
        print(f"警告: 查询文件夹 {args.query_folder} 为空，请添加图像")
        return
    if not os.path.exists(args.gallery_folder) or not any(os.scandir(args.gallery_folder)):
        print(f"警告: 图库文件夹 {args.gallery_folder} 为空，请添加图像")
        return

    # 创建评估器
    evaluator = ReIDEvaluator(args.config, args.device)

    # 评估
    results = evaluator.evaluate(args.query_folder, args.gallery_folder)

    # 输出结果
    print("\n" + "="*50)
    print("评估结果:")
    print(".2f")
    print(".2f")
    print("="*50)


if __name__ == '__main__':
    main()
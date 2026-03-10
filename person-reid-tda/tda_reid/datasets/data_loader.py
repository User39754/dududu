"""
数据加载器
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple


class ReIDDataset(Dataset):
    """
    行人重识别数据集
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transforms_fn=None,
        height: int = 256,
        width: int = 128
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train' 或 'test'
            transforms_fn: 图像变换函数
            height: 输入高度
            width: 输入宽度
        """
        self.root_dir = root_dir
        self.split = split
        self.height = height
        self.width = width
        
        # 默认变换
        if transforms_fn is None:
            self.transforms_fn = self._default_transforms()
        else:
            self.transforms_fn = transforms_fn
        
        # 加载数据列表
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """加载数据列表"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            print(f"WARNING: {split_dir} does not exist")
            return
        
        for person_id, person_dir in enumerate(sorted(os.listdir(split_dir))):
            person_path = os.path.join(split_dir, person_dir)
            
            if not os.path.isdir(person_path):
                continue
            
            for img_name in os.listdir(person_path):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_path, img_name)
                    
                    # 从文件名解析信息 (Market-1501格式: xxxx_cy_z.jpg)
                    # xxxx: 身份ID, c: 相机ID, y: 序列ID, z: 帧序号
                    parts = img_name.replace('.jpg', '').split('_')
                    if len(parts) >= 2:
                        try:
                            person_id_file = int(parts[0])
                            camera_id = int(parts[1][0]) if len(parts[1]) > 0 else 0
                        except:
                            person_id_file = person_id
                            camera_id = 0
                    else:
                        person_id_file = person_id
                        camera_id = 0
                    
                    self.data.append({
                        'path': img_path,
                        'label': person_id_file,
                        'camera_id': camera_id
                    })
        
        print(f"Loaded {len(self.data)} images from {self.split}")
    
    def _default_transforms(self):
        """默认图像变换"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # 读取图像
        try:
            image = Image.open(item['path']).convert('RGB')
            image = self.transforms_fn(image)
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # 返回黑色图像作为fallback
            image = torch.zeros(3, self.height, self.width)
        
        return {
            'image': image,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'camera_id': item['camera_id'],
            'path': item['path']
        }


def get_dataloader(
    cfg: dict,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    获取数据加载器
    
    Args:
        cfg: 配置字典
        batch_size: batch大小
        num_workers: 加载工作进程数
    
    Returns:
        Tuple: (train_loader, val_loader)
    """
    
    dataset_cfg = cfg.get('dataset', {})
    root_dir = dataset_cfg.get('root_dir', './data/market1501')
    height = dataset_cfg.get('height', 256)
    width = dataset_cfg.get('width', 128)
    
    # 训练数据集
    train_dataset = ReIDDataset(
        root_dir=root_dir,
        split='train',
        height=height,
        width=width
    )
    
    # 测试数据集
    test_dataset = ReIDDataset(
        root_dir=root_dir,
        split='test',
        height=height,
        width=width
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class SamplerFactory:
    """
    采样器工厂 - 用于处理ReID中的特殊采样策略
    """
    
    @staticmethod
    def get_balanced_sampler(dataset: Dataset, batch_size: int = 64, num_ids: int = 16):
        """
        获取平衡采样器
        每个batch包含num_ids个不同身份，每个身份num_samples张图像
        """
        from torch.utils.data import Sampler
        
        indices_per_id = {}
        for idx, sample in enumerate(dataset.data):
            label = sample['label']
            if label not in indices_per_id:
                indices_per_id[label] = []
            indices_per_id[label].append(idx)
        
        class BalancedSampler(Sampler):
            def __init__(self, indices_per_id, batch_size, num_ids):
                self.indices_per_id = indices_per_id
                self.batch_size = batch_size
                self.num_ids = num_ids
                self.num_samples_per_id = batch_size // num_ids
            
            def __iter__(self):
                ids = list(self.indices_per_id.keys())
                
                while ids:
                    # 随机选择num_ids个身份
                    selected_ids = np.random.choice(
                        ids, 
                        size=min(self.num_ids, len(ids)),
                        replace=False
                    )
                    
                    batch_indices = []
                    for pid in selected_ids:
                        # 为每个身份采样num_samples_per_id张图像
                        sampled = np.random.choice(
                            self.indices_per_id[pid],
                            size=min(self.num_samples_per_id, len(self.indices_per_id[pid])),
                            replace=True
                        )
                        batch_indices.extend(sampled)
                    
                    yield from batch_indices[:self.batch_size]
            
            def __len__(self):
                return len(self.indices_per_id) * self.num_samples_per_id
        
        return BalancedSampler(indices_per_id, batch_size, num_ids)

"""
TDA可视化和分析工具
"""

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from typing import Dict, Tuple, Optional
import seaborn as sns


class TDAVisualizer:
    """
    TDA结果的可视化
    """
    
    @staticmethod
    def plot_persistence_diagram(
        features: np.ndarray,
        max_dim: int = 1,
        figsize: Tuple[int, int] = (10, 10),
        save_path: Optional[str] = None
    ):
        """
        绘制持久性图表
        
        Args:
            features: (D,) 或 (number_of_features, ) 特征向量
            max_dim: 最大同调维数
            figsize: 图表大小
            save_path: 保存路径
        """
        if features.ndim == 1:
            point_cloud = features.reshape(-1, 1)
        else:
            point_cloud = features.reshape(-1, 1)
        
        result = ripser(point_cloud, maxdim=max_dim)
        
        fig, axes = plt.subplots(1, max_dim + 1, figsize=(5 * (max_dim + 1), 5))
        
        if max_dim == 0:
            axes = [axes]
        
        for dim in range(max_dim + 1):
            ax = axes[dim]
            
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                
                if len(dgm) > 0:
                    birth = dgm[:, 0]
                    death = dgm[:, 1]
                    
                    # 移除无穷大点
                    finite_mask = np.isfinite(death)
                    birth_finite = birth[finite_mask]
                    death_finite = death[finite_mask]
                    
                    # 绘制有限点
                    ax.scatter(birth_finite, death_finite, s=50, alpha=0.6, label=f'H{dim}')
                    
                    # 绘制对角线
                    max_val = max(birth_finite.max(), death_finite.max()) if len(birth_finite) > 0 else 1
                    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
                    
                    # 绘制无穷大点
                    if (~finite_mask).any():
                        inf_birth = birth[~finite_mask]
                        ax.scatter(inf_birth, [max_val] * len(inf_birth), s=100, 
                                 marker='^', alpha=0.6, color='red', label='Infinite')
            
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'Persistence Diagram H{dim}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Persistence diagram saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_barcode(
        features: np.ndarray,
        max_dim: int = 1,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制条形码图
        """
        if features.ndim == 1:
            point_cloud = features.reshape(-1, 1)
        else:
            point_cloud = features.reshape(-1, 1)
        
        result = ripser(point_cloud, maxdim=max_dim)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = 0
        colors = ['blue', 'green', 'red', 'orange']
        
        for dim in range(max_dim + 1):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                
                for birth, death in dgm:
                    if np.isfinite(death):
                        ax.barh(y_pos, death - birth, left=birth, 
                               color=colors[dim % len(colors)], 
                               alpha=0.7, height=0.8)
                    else:
                        # 无穷大条形
                        max_val = np.max([d for b, d in dgm if np.isfinite(d)])
                        ax.barh(y_pos, max_val - birth, left=birth, 
                               color=colors[dim % len(colors)], 
                               alpha=0.3, height=0.8, hatch='///')
                    
                    y_pos += 1
        
        ax.set_xlabel('Birth')
        ax.set_ylabel('Feature Index')
        ax.set_title('Persistence Barcode')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Barcode diagram saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_tda_feature_heatmap(
        tda_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制TDA特征热力图
        
        Args:
            tda_features: (N_samples, N_features) TDA特征矩阵
            labels: (N_samples,) 样本标签 (可选)
            figsize: 图表大小
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 如果有标签，按标签排序
        if labels is not None:
            sorted_idx = np.argsort(labels)
            tda_features_sorted = tda_features[sorted_idx]
        else:
            tda_features_sorted = tda_features
        
        sns.heatmap(tda_features_sorted.T, cmap='viridis', ax=ax, cbar_kws={'label': 'Feature Value'})
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('TDA Feature Index')
        ax.set_title('TDA Feature Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_intra_inter_distances(
        tda_features: np.ndarray,
        labels: np.ndarray,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制类内和类间距离分布
        """
        from scipy.spatial.distance import pdist, squareform
        
        dist_mat = squareform(pdist(tda_features, metric='euclidean'))
        
        intra_dists = []
        inter_dists = []
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            if mask.sum() <= 1:
                continue
            
            # 类内
            intra = dist_mat[np.ix_(mask, mask)]
            intra_dists.extend(intra[np.triu_indices_from(intra, k=1)])
            
            # 类间
            inter_mask = labels != label
            inter = dist_mat[np.ix_(mask, inter_mask)]
            inter_dists.extend(inter.flatten())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(intra_dists, bins=50, alpha=0.6, label='Intra-class', color='blue')
        ax.hist(inter_dists, bins=50, alpha=0.6, label='Inter-class', color='red')
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('TDA Feature Distance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distance distribution saved to {save_path}")
        else:
            plt.show()
        
        return fig


class TDAAnalyzer:
    """
    TDA特征分析器
    """
    
    @staticmethod
    def compute_topological_statistics(
        features: np.ndarray,
        max_dim: int = 1
    ) -> Dict:
        """
        计算拓扑统计信息
        """
        if features.ndim == 1:
            point_cloud = features.reshape(-1, 1)
        else:
            point_cloud = features.reshape(-1, 1)
        
        result = ripser(point_cloud, maxdim=max_dim)
        
        stats = {}
        
        for dim in range(max_dim + 1):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                
                stats[f'H{dim}_count'] = len(dgm)
                
                if len(dgm) > 0:
                    persistence = dgm[:, 1] - dgm[:, 0]
                    persistence = persistence[np.isfinite(persistence)]
                    
                    if len(persistence) > 0:
                        stats[f'H{dim}_sum_persistence'] = np.sum(persistence)
                        stats[f'H{dim}_avg_persistence'] = np.mean(persistence)
                        stats[f'H{dim}_max_persistence'] = np.max(persistence)
                        stats[f'H{dim}_std_persistence'] = np.std(persistence)
        
        return stats
    
    @staticmethod
    def compare_tda_features(
        features1: np.ndarray,
        features2: np.ndarray,
        method: str = 'euclidean'
    ) -> float:
        """
        比较两个TDA特征向量的相似度
        """
        from scipy.spatial.distance import euclidean
        
        if method == 'euclidean':
            dist = euclidean(features1, features2)
            similarity = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(similarity)


if __name__ == '__main__':
    # 测试
    print("TDA Visualization Module")
    
    # 生成随机特征
    features = np.random.randn(100)
    
    # 绘制持久性图表
    TDAVisualizer.plot_persistence_diagram(features, save_path='persistence_diagram.png')
    
    # 绘制条形码
    TDAVisualizer.plot_barcode(features, save_path='persistence_barcode.png')

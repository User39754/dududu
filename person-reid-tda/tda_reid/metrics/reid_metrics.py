"""
ReID 评估指标
mAP, CMC曲线, TDA特定指标
"""

import numpy as np
import torch
from typing import Dict, Tuple
from scipy.spatial.distance import cdist


class ReIDEvaluator:
    """
    行人重识别评估器
    计算标准指标: mAP, CMC@1/5/10, Ranks等
    """
    
    def __init__(self, dataset_name: str = 'market1501'):
        self.dataset_name = dataset_name
        
    def evaluate(
        self,
        query_feats: np.ndarray,
        query_labels: np.ndarray,
        query_cams: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_labels: np.ndarray,
        gallery_cams: np.ndarray,
        metric: str = 'cosine'
    ) -> Dict[str, float]:
        """
        Args:
            query_feats: (N_query, feat_dim) 查询特征
            query_labels: (N_query,) 查询标签
            query_cams: (N_query,) 查询相机ID
            gallery_feats: (N_gallery, feat_dim) 库特征
            gallery_labels: (N_gallery,) 库标签
            gallery_cams: (N_gallery,) 库相机ID
            metric: 距离度量 ('euclidean' 或 'cosine')
        
        Returns:
            Dict: 包含各项指标
        """
        # 计算距离矩阵
        dist_mat = cdist(query_feats, gallery_feats, metric=metric)
        
        # 计算CMC和mAP
        cmc, mAP = self._compute_cmc_map(
            dist_mat,
            query_labels,
            query_cams,
            gallery_labels,
            gallery_cams
        )
        
        # 提取关键指标
        results = {
            'mAP': mAP,
            'CMC@1': cmc[0],
            'CMC@5': cmc[4],
            'CMC@10': cmc[9],
            'CMC@20': cmc[19] if len(cmc) > 19 else cmc[-1],
        }
        
        return results
    
    def _compute_cmc_map(
        self,
        dist_mat: np.ndarray,
        query_labels: np.ndarray,
        query_cams: np.ndarray,
        gallery_labels: np.ndarray,
        gallery_cams: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        计算CMC曲线和mAP
        """
        num_q = dist_mat.shape[0]
        cmc = np.zeros(dist_mat.shape[1])
        ap_scores = []
        
        for i in range(num_q):
            # 排序
            sorted_idx = np.argsort(dist_mat[i])
            
            # 获取排序后的标签
            sorted_labels = gallery_labels[sorted_idx]
            sorted_cams = gallery_cams[sorted_idx]
            
            # 移除同一相机的同身份图像 (Market-1501规则)
            query_id = query_labels[i]
            query_cam = query_cams[i]
            
            valid_mask = (
                (sorted_labels == query_id) & (sorted_cams != query_cam)
            ) | (sorted_labels != query_id)
            
            sorted_labels = sorted_labels[valid_mask]
            
            # 计算CMC
            match = (sorted_labels == query_id)
            if match.any():
                first_match_idx = np.argmax(match)
                cmc[first_match_idx:] += 1
            
            # 计算AP
            num_match = (sorted_labels == query_id).sum()
            if num_match > 0:
                match_positions = np.where(sorted_labels == query_id)[0]
                precisions = np.arange(1, num_match + 1) / (match_positions + 1)
                ap_scores.append(precisions.mean())
            else:
                ap_scores.append(0.0)
        
        cmc = cmc / num_q
        mAP = np.mean(ap_scores)
        
        return cmc, mAP


class TDAMetrics:
    """
    TDA特定的评估指标
    - 拓扑相似度
    - 持久性图的相似度
    - 拓扑不变量的稳定性
    """
    
    @staticmethod
    def topological_similarity(
        tda_feat1: np.ndarray,
        tda_feat2: np.ndarray,
        metric: str = 'euclidean'
    ) -> float:
        """
        计算两个TDA特征向量之间的相似度
        较小的距离表示拓扑结构更相似
        """
        from scipy.spatial.distance import euclidean, cosine
        
        if metric == 'euclidean':
            dist = euclidean(tda_feat1, tda_feat2)
            # 转换为0-1的相似度
            similarity = 1.0 / (1.0 + dist)
        elif metric == 'cosine':
            dist = cosine(tda_feat1, tda_feat2)
            similarity = 1.0 - dist
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(similarity)
    
    @staticmethod
    def persistence_stability(
        dgm: np.ndarray,
        threshold: float = 0.05
    ) -> float:
        """
        计算持久性图的稳定性
        稳定性定义为 threshold上方的特征比例
        
        Args:
            dgm: (N, 2) 持久性图 [birth, death]
            threshold: 持久度阈值
        
        Returns:
            stability: 0-1, 越接近1表示拓扑越稳定
        """
        if len(dgm) == 0:
            return 0.0
        
        persistence = dgm[:, 1] - dgm[:, 0]
        persistence = persistence[np.isfinite(persistence)]
        
        if len(persistence) == 0:
            return 0.0
        
        stability = (persistence > threshold).sum() / len(persistence)
        return float(stability)
    
    @staticmethod
    def betti_number_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        """
        计算两个持久性图的Betti数距离
        Betti数 = 该维同调特征的数量
        """
        betti1 = len(dgm1) if len(dgm1) > 0 else 0
        betti2 = len(dgm2) if len(dgm2) > 0 else 0
        
        distance = abs(betti1 - betti2)
        return float(distance)
    
    @staticmethod
    def compute_tda_intra_inter_distances(
        tda_features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算类内和类间的TDA距离
        
        Returns:
            Dict: {'intra_distance': ..., 'inter_distance': ...}
        """
        from scipy.spatial.distance import pdist, squareform
        
        dist_mat = squareform(pdist(tda_features, metric='euclidean'))
        
        unique_labels = np.unique(labels)
        intra_dists = []
        inter_dists = []
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() <= 1:
                continue
            
            # 类内距离
            intra_dist = dist_mat[np.ix_(mask, mask)]
            intra_dists.extend(intra_dist[np.triu_indices_from(intra_dist, k=1)])
            
            # 类间距离
            inter_mask = labels != label
            inter_dist = dist_mat[np.ix_(mask, inter_mask)]
            inter_dists.extend(inter_dist.flatten())
        
        return {
            'intra_distance': np.mean(intra_dists) if intra_dists else 0.0,
            'inter_distance': np.mean(inter_dists) if inter_dists else 0.0,
        }


class MetricsLogger:
    """
    指标记录器
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def update(self, metrics_dict: Dict[str, float]):
        self.metrics_history.append(metrics_dict)
    
    def get_best(self, metric_name: str) -> float:
        """获取最佳指标值"""
        if not self.metrics_history:
            return 0.0
        values = [m.get(metric_name, 0.0) for m in self.metrics_history]
        return max(values)
    
    def print_summary(self):
        """打印摘要"""
        if not self.metrics_history:
            print("No metrics recorded")
            return
        
        latest = self.metrics_history[-1]
        best_map = self.get_best('mAP')
        best_rank1 = self.get_best('CMC@1')
        
        print(f"\nLatest Metrics:")
        for k, v in latest.items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\nBest Metrics:")
        print(f"  Best mAP: {best_map:.4f}")
        print(f"  Best Rank-1: {best_rank1:.4f}")

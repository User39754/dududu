"""
演示脚本 - TDA特征分析
展示如何使用TDA特征提取器和分析工具
"""

import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tda_reid.models.tda_extractor import TDAExtractor
from tda_reid.utils.visualization import TDAVisualizer, TDAAnalyzer
from tda_reid.metrics.reid_metrics import TDAMetrics


def demo_tda_extraction():
    """
    演示1: 基础TDA特征提取
    """
    print("=" * 60)
    print("Demo 1: TDA Feature Extraction")
    print("=" * 60)
    
    # 创建TDAExtractor
    extractor = TDAExtractor(max_dim=1, max_features=50)
    
    # 生成随机特征向量
    batch_size = 10
    feature_dim = 2048
    features = torch.randn(batch_size, feature_dim)
    
    # 计算TDA特征
    tda_features = extractor(features)
    
    print(f"Input shape: {features.shape}")
    print(f"TDA features shape: {tda_features.shape}")
    print(f"Sample TDA features:\n{tda_features[0, :10]}")
    print()


def demo_persistence_diagram():
    """
    演示2: 持久性图表可视化
    """
    print("=" * 60)
    print("Demo 2: Persistence Diagram Visualization")
    print("=" * 60)
    
    # 生成样本特征
    features = np.random.randn(100)
    
    # 绘制持久性图表
    visualizer = TDAVisualizer()
    visualizer.plot_persistence_diagram(
        features,
        max_dim=1,
        save_path='demo_persistence_diagram.png'
    )
    
    print("Persistence diagram saved as 'demo_persistence_diagram.png'")
    print()


def demo_topological_statistics():
    """
    演示3: 拓扑统计分析
    """
    print("=" * 60)
    print("Demo 3: Topological Statistics")
    print("=" * 60)
    
    # 生成两个不同的特征分布
    features1 = np.random.randn(100)  # 高斯分布
    features2 = np.random.uniform(-3, 3, 100)  # 均匀分布
    
    analyzer = TDAAnalyzer()
    
    stats1 = analyzer.compute_topological_statistics(features1)
    stats2 = analyzer.compute_topological_statistics(features2)
    
    print("Gaussian Distribution Statistics:")
    for k, v in stats1.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nUniform Distribution Statistics:")
    for k, v in stats2.items():
        print(f"  {k}: {v:.4f}")
    print()


def demo_similarity_comparison():
    """
    演示4: TDA特征相似度比较
    """
    print("=" * 60)
    print("Demo 4: TDA Feature Similarity Comparison")
    print("=" * 60)
    
    # 提取TDA特征
    extractor = TDAExtractor(max_dim=1, max_features=50)
    
    # 生成相似和不相似的特征对
    base_feat = torch.randn(1, 256)
    similar_feat = base_feat + torch.randn(1, 256) * 0.1  # 略微不同
    different_feat = torch.randn(1, 256)  # 完全不同
    
    tda_base = extractor(base_feat).numpy()[0]
    tda_similar = extractor(similar_feat).numpy()[0]
    tda_different = extractor(different_feat).numpy()[0]
    
    # 计算相似度
    analyzer = TDAAnalyzer()
    sim_same = analyzer.compare_tda_features(tda_base, tda_similar)
    sim_diff = analyzer.compare_tda_features(tda_base, tda_different)
    
    print(f"Similarity between similar features: {sim_same:.4f}")
    print(f"Similarity between different features: {sim_diff:.4f}")
    print(f"Difference: {sim_same - sim_diff:.4f} (should be positive)")
    print()


def demo_batch_analysis():
    """
    演示5: 批量样本分析
    """
    print("=" * 60)
    print("Demo 5: Batch Sample Analysis with TDA")
    print("=" * 60)
    
    # 生成多个身份的特征
    num_identities = 10
    samples_per_id = 5
    feature_dim = 256
    
    all_features = []
    all_labels = []
    
    for person_id in range(num_identities):
        # 为每个身份生成5个样本
        base_feature = np.random.randn(1, feature_dim)
        for _ in range(samples_per_id):
            noisy_feature = base_feature + np.random.randn(1, feature_dim) * 0.2
            all_features.append(noisy_feature)
            all_labels.append(person_id)
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    # 提取TDA特征
    extractor = TDAExtractor(max_dim=1, max_features=30)
    tda_features_list = []
    
    for feat in all_features:
        feat_tensor = torch.from_numpy(feat).unsqueeze(0).float()
        tda_feat = extractor(feat_tensor).numpy()[0]
        tda_features_list.append(tda_feat)
    
    tda_features = np.array(tda_features_list)
    
    # 计算类内和类间距离
    tda_metrics = TDAMetrics()
    metrics = tda_metrics.compute_tda_intra_inter_distances(tda_features, all_labels)
    
    print(f"Number of identities: {num_identities}")
    print(f"Samples per identity: {samples_per_id}")
    print(f"Total samples: {len(all_features)}")
    print(f"TDA feature dimension: {tda_features.shape[1]}")
    print()
    print("Distance Metrics:")
    print(f"  Intra-class distance: {metrics['intra_distance']:.4f}")
    print(f"  Inter-class distance: {metrics['inter_distance']:.4f}")
    print(f"  Distance ratio (inter/intra): {metrics['inter_distance'] / (metrics['intra_distance'] + 1e-8):.4f}")
    print()
    
    # 绘制距离分布
    visualizer = TDAVisualizer()
    visualizer.plot_intra_inter_distances(
        tda_features,
        all_labels,
        save_path='demo_distances.png'
    )
    print("Distance distribution saved as 'demo_distances.png'")


def main():
    """
    运行所有演示
    """
    print("\n" + "=" * 60)
    print("TDA FOR PERSON RE-IDENTIFICATION - DEMONSTRATION")
    print("=" * 60 + "\n")
    
    try:
        demo_tda_extraction()
        demo_persistence_diagram()
        demo_topological_statistics()
        demo_similarity_comparison()
        demo_batch_analysis()
        
        print("=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

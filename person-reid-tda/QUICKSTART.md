# 项目快速开始指南

## 一、环境设置

### 1.1 创建虚拟环境
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

## 二、核心模块说明

### 2.1 TDA特征提取 (tda_reid/models/tda_extractor.py)

**TDAExtractor** - 计算持久同调特征
```python
from tda_reid.models.tda_extractor import TDAExtractor
import torch

extractor = TDAExtractor(max_dim=1, max_features=50)
features = torch.randn(32, 2048)  # batch of 32 features
tda_features = extractor(features)  # (32, 50)
```

**TDABackbone** - TDA增强的特征融合
```python
from tda_reid.models.tda_extractor import TDABackbone

tda_backbone = TDABackbone(feature_dim=2048, tda_dim=50)
cnn_features = torch.randn(32, 2048)
fused_features, tda_features = tda_backbone(cnn_features)
```

### 2.2 ReID模型 (tda_reid/models/reid_model.py)

**TDAReIDModel** - 完整的行人重识别模型
```python
from tda_reid.models.reid_model import TDAReIDModel

model = TDAReIDModel(num_classes=1501, feature_dim=2048, tda_dim=50)
output = model(images)  # dict with 'features', 'logits', 'tda_features'
```

### 2.3 损失函数 (tda_reid/losses/tda_loss.py)

**CombinedLoss** - 综合损失 (分类 + TDA拓扑约束 + 三元组)
```python
from tda_reid.losses.tda_loss import CombinedLoss

loss_fn = CombinedLoss(num_classes=1501, lambda_tda=0.5, lambda_triplet=0.5)
loss, loss_dict = loss_fn(logits, tda_features, features, labels)
```

**TDATopologyLoss** - 独立的TDA损失
```python
from tda_reid.losses.tda_loss import TDATopologyLoss

tda_loss = TDATopologyLoss(margin=0.3)
loss = tda_loss(tda_features, labels, original_features)
```

### 2.4 评估指标 (tda_reid/metrics/reid_metrics.py)

**ReIDEvaluator** - 标准ReID指标 (mAP, CMC)
```python
from tda_reid.metrics.reid_metrics import ReIDEvaluator

evaluator = ReIDEvaluator()
metrics = evaluator.evaluate(
    query_feats, query_labels, query_cams,
    gallery_feats, gallery_labels, gallery_cams
)
# Returns: {'mAP': ..., 'CMC@1': ..., 'CMC@5': ...}
```

**TDAMetrics** - TDA特定指标
```python
from tda_reid.metrics.reid_metrics import TDAMetrics

# 计算拓扑相似度
similarity = TDAMetrics.topological_similarity(tda_feat1, tda_feat2)

# 计算类内和类间距离
metrics = TDAMetrics.compute_tda_intra_inter_distances(tda_features, labels)
```

## 三、演示和实验

### 3.1 运行TDA分析演示
```bash
cd experiments
python demo_tda_analysis.py
```

This will:
- 演示TDA特征提取
- 可视化持久性图表
- 计算拓扑统计
- 比较特征相似度
- 分析批量样本

### 3.2 查看生成结果
演示会生成以下文件：
- `demo_persistence_diagram.png` - 持久性图表
- `demo_distances.png` - 距离分布直方图

## 四、训练流程

### 4.1 准备数据

下载Market-1501数据集并放在 `data/market1501/` 目录：
```
data/
└── market1501/
    ├── train/
    ├── test/
    └── ...
```

### 4.2 修改配置

编辑 `configs/config_market.yaml`：
```yaml
dataset:
  root_dir: ./data/market1501
  num_classes: 1501

training:
  batch_size: 64
  learning_rate: 0.0003
  num_epochs: 150
```

### 4.3 开始训练

```bash
cd experiments
python train.py --config ../configs/config_market.yaml --gpu 0
```

## 五、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tda_dim` | 50 | TDA特征维度 |
| `lambda_tda` | 0.5 | TDA损失权重 |
| `lambda_triplet` | 0.5 | 三元组损失权重 |
| `margin` | 0.3 | 损失函数边界 |
| `learning_rate` | 1e-4 | 学习率 |
| `batch_size` | 64 | 批大小 |

## 六、可视化工具

### 6.1 持久性图表
```python
from tda_reid.utils.visualization import TDAVisualizer

visualizer = TDAVisualizer()
visualizer.plot_persistence_diagram(features, save_path='dgm.png')
visualizer.plot_barcode(features, save_path='barcode.png')
```

### 6.2 特征热力图
```python
visualizer.plot_tda_feature_heatmap(tda_features, labels, save_path='heatmap.png')
visualizer.plot_intra_inter_distances(tda_features, labels, save_path='distances.png')
```

## 七、项目扩展方向

1. **多尺度TDA** - 在不同特征层级应用TDA
2. **动态TDA** - 视频ReID中的时序拓扑分析
3. **跨域自适应** - 基于TDA的域适应方法
4. **图TDA** - 融合图神经网络的拓扑分析

## 八、常见问题

**Q: TDA特征提取速度慢？**
A: 使用 `lightweight=True` 模型或减少 `max_features`

**Q: 内存不足？**
A: 减小 `batch_size` 或使用梯度累积

**Q: 模型不收敛？**
A: 调整 `lambda_tda` 和 `lambda_triplet` 的权重比例

## 九、参考资源

- **Torchreid文档**: https://kaiyangzhou.github.io/deep-person-reid/
- **Ripser.py文档**: https://github.com/scikit-tda/ripser.py
- **TDA理论**: https://en.wikipedia.org/wiki/Persistent_homology

---

**项目维护**: 2026年
**最后更新**: 2026-03-09

# Person Re-ID 评估演示

这个脚本提供了一个简单的接口，用于评估Person Re-ID模型的性能。

## 文件夹结构

脚本会自动创建以下文件夹：
- `../data/device/` - 设备相关文件
- `../data/query_images/` - 查询图像文件夹
- `../data/gallery_images/` - 图库图像文件夹

## 使用方法

### 准备数据
将查询图像放在 `../data/query_images/` 中，图库图像放在 `../data/gallery_images/` 中。
文件夹结构示例：
```
data/query_images/
├── person_001/
│   ├── 001.jpg
│   └── 002.jpg
└── person_002/
    ├── 003.jpg
    └── 004.jpg

data/gallery_images/
├── person_001/
│   ├── 005.jpg
│   └── 006.jpg
└── person_002/
    ├── 007.jpg
    └── 008.jpg
```

### 运行评估

```bash
cd experiments
python demo_evaluation.py
```

脚本默认使用：
- 设备：cuda
- 查询文件夹：../data/query_images
- 图库文件夹：../data/gallery_images

### 自定义参数

```bash
python demo_evaluation.py --query_folder /path/to/query --gallery_folder /path/to/gallery --device cpu
```

### 参数说明

- `--config`: 配置文件路径（默认：../configs/config_market.yaml）
- `--query_folder`: 查询图像文件夹路径（默认：../data/query_images）
- `--gallery_folder`: 图库图像文件夹路径（默认：../data/gallery_images）
- `--device`: 计算设备（默认：cuda）

### 输出

脚本将输出：
- Rank-1 准确率（CMC@1）
- mAP（mean Average Precision）

## 注意事项

1. 图像文件夹名将被用作身份ID
2. 所有图像被假设来自同一摄像头（cam_id=0）
3. 模型使用配置文件中的设置，如果有检查点会自动加载
4. 支持的图像格式：PNG, JPG, JPEG, BMP
# 基于 Foundation Stereo 和 YOLO v11 的3D目标跟踪系统

这是一个离线3D物体检测系统，将用于物体检测的YOLOv11与用于深度估计的Foundation Stereo相结合，从而生成3D边界框并实现BEV可视化。

# 项目结构

```
├── detect_and_track/          # 主要代码
│   ├── run_stereo.py          # 主入口
│   ├── stereo_depth_model.py  # Fast-FoundationStereo 深度封装
│   ├── detection_model.py     # YOLOv11 检测器
│   ├── bbox3d_utils.py        # 3D 框估计 + BEV 可视化
│   ├── load_camera_params.py  # 相机参数加载
│   └── test_single.py         # 单帧诊断脚本
├── core/                      # Fast-FoundationStereo 核心模型
│   ├── foundation_stereo.py   # 模型定义
│   ├── extractor.py           # 共享 backbone 特征提取
│   ├── geometry.py            # 几何编码体
│   ├── submodule.py           # 子模块 (GWC, 注意力, etc.)
│   ├── update.py              # GRU 迭代更新
│   ├── distill_block.py       # 蒸馏块
│   └── utils/
├── depth_anything/            # Depth Anything V2 backbone
│   └── ...
├── dinov2/                    # DINOv2 backbone 配置文件
│   └── ...
├── Utils.py                   # 工具函数
├── requirements.txt           # Python 依赖
├── readme.md                  # 项目说明
└── .gitignore
```

# 安装环境

```
conda env create -f environment.yml
conda activate foundation_stereo
```

# 模型权重
- 下载 foundation model 预训练权重. 把整个权重文件(如 `23-51-11`)解压到`./pretrained_models/`.

| 模型     | 特点                                                                 |
|-----------|-----------------------------------------------------------------------------|
| [23-51-11](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | 性能最强（Vit-large）               |
| [11-33-40](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  | 性能稍差但推理速度提升（Vit-small）            |

# 运行

```
python detect_and_track/run_stereo.py --left_file ./data_left --right_file ./data_right --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./output/
```

## 演示视频

若视频不可加载请点击超链接。

[FULL](https://www.bilibili.com/video/BV1WMjt6bEXi/?)

https://github.com/user-attachments/assets/56ba7cce-12a5-4920-a474-68b0cbc006a3

[SLIGHT](https://www.bilibili.com/video/BV1WMjt6bEXi?p=2&)

https://github.com/user-attachments/assets/ff13981b-509c-4d51-8538-0edf07ee1532

| 模型     | 推理速度(3080 10G)||||
|----------|-|-|-|----------------------------|
|      | total | Detect | Depth | Avg FPS |
| FULL  | 1106ms|19ms|1084ms|0.8 |
| SLIGHT  | 925ms|23ms|892ms|1.0            |

注：
- 左右眼输入数据需要是序列图像，且应经过校正，无畸变，左右图像之间的极线应呈水平方向。可以使用[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)官方数据
- 请勿调换左右图像的位置。左图必须确实来自左侧摄像头（图像中的物体应向右偏移）。
- 我们建议使用无损压缩的 PNG 文件。
- 对于高分辨率图像（>1000px），您可以：(1) 使用 `--hiera 1` 启用分层推理，以获得全分辨率深度图，但速度较慢；或者 (2) 使用较小的缩放比例，例如 `--scale 0.5`，以获得缩小分辨率但速度更快的深度图。
若需加快推理速度，可通过 `--scale 0.5` 等参数降低输入图像分辨率，并减少精化迭代次数，例如使用 `--valid_iters 16`。
- 如果需要更快的推理速度可以选择[Fast版本](https://github.com/Lao-G-G/Fast-FoundationStereo-and-yolov11-based-detector)。

## 配置选项

您可以在 `run.py` 或命令行修改以下参数：

### 输入/输出:

`--left_dir`: 左眼视图

`--right_dir`: 右眼视图

`--output_path`: 输出路径

### 模型选择:

`--yolo_model`: YOLOv11 模型大小 ("nano", "small", "medium", "large", "extra")

`--ckpt_dir`: Foundation Stereo 模型大小 (编码器大小："Vit-large", "Vit-small")

### 检测设置:

`--conf_threshold`: 物体检测的置信阈值

`--iou_threshold`: NMS的IOU阈值

### 功能开关:

`--no_tracking`: 目标跟踪

`--no_bev`: BEV可视化

  `--z_far`: BEV可视化最大深度

# 工作流

```
左目图像 ──→ YOLOv11 2D 检测 ──→ 2D 边界框 + 类别 + 跟踪 ID

左目+右目 ──→ FoundationStereo ──→ 全图像素级 metric 深度图
              ↑                         ↓
        2D 框 + 深度值 + 相机内参 → 反向投影 → 3D 边界框 (x, y, z, 尺寸, 朝向)
                                           ↓
                                      可视化：3D 立体框 + BEV 俯视图
```

1. **目标检测**：YOLOv11 检测左眼视图中的目标并提供 2D 边界框
2. **深度估计**：Foundation Stereo 根据左右眼视差为整个画面生成深度图
3. **3D 边界框估计**：根据检测框的深度估计以及左右眼视差生成 3D 边界框
4. **可视化**：渲染 3D 边界框和俯视图，以更好地理解空间关系

# 致谢

[YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)

[Depth Anything v2 by TikTok](https://github.com/DepthAnything/Depth-Anything-V2)

[Foundation Stereo by Nvidia](https://github.com/NVlabs/FoundationStereo)

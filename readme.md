# 基于 Foundation Stereo 和 YOLO v11 的3D目标跟踪系统

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

[FULL](https://www.bilibili.com/video/BV1WMjt6bEXi/?)



https://github.com/user-attachments/assets/56ba7cce-12a5-4920-a474-68b0cbc006a3



<video width="630" height="300" src="https://github.com/user-attachments/assets/56ba7cce-12a5-4920-a474-68b0cbc006a3"></video>

https://github.com/user-attachments/assets/c2b1e24b-3cc2-41c0-ac09-006fc5a33ae6


[SLIGHT](https://www.bilibili.com/video/BV1WMjt6bEXi?p=2&)

注：
- 左右眼输入数据需要是序列图像，且应经过校正，无畸变，左右图像之间的极线应呈水平方向。可以使用[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)官方数据
- 请勿调换左右图像的位置。左图必须确实来自左侧摄像头（图像中的物体应向右偏移）。
- 我们建议使用无损压缩的 PNG 文件。
- 对于高分辨率图像（>1000px），您可以：(1) 使用 `--hiera 1` 启用分层推理，以获得全分辨率深度图，但速度较慢；或者 (2) 使用较小的缩放比例，例如 `--scale 0.5`，以获得缩小分辨率但速度更快的深度图。
若需加快推理速度，可通过 `--scale 0.5` 等参数降低输入图像分辨率，并减少精化迭代次数，例如使用 `--valid_iters 16`.

# 工作流



https://github.com/user-attachments/assets/bf68cdc4-9beb-45fd-9edd-b529dbbcf72d    full
https://github.com/user-attachments/assets/8e0c2b83-c02c-430d-a7f8-b22341543db4    slight

# Acknowledgement
We would like to thank Gordon Grigor, Jack Zhang, Karsten Patzwaldt, Hammad Mazhar and other NVIDIA Isaac team members for their tremendous engineering support and valuable discussions. Thanks to the authors of [DINOv2](https://github.com/facebookresearch/dinov2), [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Selective-IGEV](https://github.com/Windsrain/Selective-Stereo) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for their code release. Finally, thanks to CVPR reviewers and AC for their appreciation of this work and constructive feedback.

# 引用
For commercial inquiries, additional technical support, and other questions, please reach out to [Bowen Wen](https://wenbowen123.github.io/) (bowenw@nvidia.com).

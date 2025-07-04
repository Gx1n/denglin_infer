 # denglin_infer

## 项目简介

`denglin_infer` 使用登临denglin显卡推理yolov10模型。

## 特性

- 支持 ONNX
- 支持图片批量推理与结果可视化
- 可配置的推理参数与后处理流程
- 详细的日志与性能统计

## 目录结构

```
.
├── dl_infer.py           # 主推理脚本
├── configs/              # 配置文件目录
│   └── config.yaml
├── models/               # 存放模型文件（.onnx, .rlym等）
├── data/                 # 数据目录
│   ├── images/           # 测试图片
│   └── coco.names        # 类别名文件
├── detect_res/           # 推理结果输出目录
├── utils/                # 工具模块
│   ├── img_util.py
│   ├── nms.py
│   ├── model_util.py
│   ├── constants.py
│   └── nne_util.py
└── requirements.txt      # 依赖包列表
```

## 安装依赖

建议使用 Python 3.8 环境。

```bash
pip install -r requirements.txt
```

## 模型与数据准备

1. 将你的模型文件（如 `v10m-fs-0703.onnx` 或 `v10m-fs-0703.rlym`）放入 `models/` 目录。
2. 将待推理图片放入 `data/images/` 目录，支持 `.jpg`、`.jpeg`、`.png` 格式。
3. 根据需要修改 `configs/config.yaml`，指定模型路径、输入类型、后处理参数等。

## 快速开始

运行主推理脚本：

```bash
python dl_infer.py
```

推理结果图片将保存在 `detect_res/` 目录，并在终端输出每张图片的推理耗时。

## 配置说明

配置文件位于 `configs/config.yaml`，主要参数说明：

- `model_path`：模型文件路径
- `input_type`：输入类型（支持 `img`、`random`、`npy`）
- `input_types`：各输入类型的详细参数
- `do_compares`：对比模式（如 `fp32`、`int8`）
- `compare_templates`：对比参数模板
- `logging`：日志格式与级别

## 推理结果

- 检测结果图片保存在 `detect_res/`，检测框和类别信息已绘制在图片上。
- 支持火焰（红色框）、烟雾（蓝色框）等类别。

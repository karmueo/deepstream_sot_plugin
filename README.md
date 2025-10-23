<h1 align="center">DeepStream 单目标跟踪插件 (libsot.so)</h1>

本仓库中的 `src/sot_plugin` 目录提供一个可插入 NVIDIA DeepStream Pipeline 的 **单目标跟踪 (Single Object Tracking, SOT)** 动态库插件，支持多种主流 Transformer / Siamese 结构跟踪模型，将其通过 TensorRT 部署后在 DeepStream 中以自定义 Tracker 的形式使用。核心输出为目标的连续位置（含打分），用于替换或补充基于检测 + 多目标跟踪（MOT）的场景中对某个指定对象的精细跟踪需求。

---
## 目录
1. 项目目的
2. 支持模型与功能特性
3. 目录结构说明
4. 依赖与环境要求
5. 编译与安装
6. 模型转换 (ONNX -> TensorRT Engine)
7. 配置文件说明 (`config_sot.yml`)
8. 在 DeepStream 中的使用示例
9. 调试与测试脚本
10. 常见问题 (FAQ)
11. 许可证

---
## 1. 项目目的
在实时视频分析 (Video Analytics) 场景中，往往需要对某个已知或临时交互选中的单一目标进行高精度、低漂移的持续跟踪，而不再依赖每帧检测。该插件旨在：

* 将高性能 SOT 模型（SuTrack / OSTrack / MixFormerV2）以 TensorRT 形式高效运行。
* 提供统一的 C++ 接口封装 (`DeepTracker` + 各模型派生类) 与简单 YAML 配置。
* 以 DeepStream 自定义 tracker 插件（生成 `libsot.so`）方式无缝集成到现有 DeepStream pipeline 中。
* 提供目标管理逻辑（置信度阈值、丢失恢复、模板更新、中心稳定性判断等）。

---
## 2. 支持模型与功能特性
当前支持三类模型（通过 `BaseConfig.modelName` 选择）：

| 数值 | 枚举 | 模型 | 特性概述 |
|------|------|------|----------|
| 0 | MODEL_SUTRACK | SuTrack | 轻量级、较快响应 |
| 1 | MODEL_OSTRACK | OSTrack | 稳定的 Transformer 跟踪 |
| 2 | MODEL_MIXFORMERV2 | MixFormerV2 | 融合特征、在线模板更新能力强 |

核心特性：
* TensorRT 引擎加载与推理（FP32 / FP16 引擎均可，只要已生成）
* 在线模板更新（MixFormerV2 专用参数）
* 目标状态管理（年龄、丢失次数、确认阈值）
* 可选“跟踪中心稳定性”判定，剔除疑似抖动噪声
* 可配置的 IOU / 分数阈值与尺度约束

---
## 3. 目录结构说明 (节选)
```
src/sot_plugin/
  CMakeLists.txt           # 仅编译本插件的 CMake 文件
  config_sot.yml           # 跟踪配置（模型与阈值等）
  includes/                # 头文件
    baseTrack_trt.h        # 抽象基类，封装通用 TensorRT 推理与预后处理逻辑
    deepTracker.h          # 统一对外的跟踪控制类 + 配置结构体
    mixformerv2_trt.h
    ostrack_trt.h
    suTrack_trt.h
    Tracker.h              # 与 DeepStream NvMOT 接口衔接的上下文定义
  src/                     # 各模型与封装的实现 *.cpp
  models/                  # ONNX / TRT engine 及转换脚本
    convert2trt.sh         # 示例：使用 trtexec 生成 engine
    test_model.py          # Python 侧简单验证脚本
    test_engine.sh         # 引擎推理测试脚本
lib/
  libsot.so                # 编译产物（共享库）
```

---
## 4. 依赖与环境要求
硬件：
* NVIDIA GPU（建议 >= Turing；越新的架构对 Transformer 类模型推理更友好）

软件 / 库：
* Ubuntu 20.04 / 22.04（示例环境）
* CUDA（CMake 中示例指向 `/usr/local/cuda`，请与本机实际路径一致）
* TensorRT（随 DeepStream 安装提供）
* NVIDIA DeepStream SDK（`DS_ROOT_DIR` 默认 `/opt/nvidia/deepstream/deepstream`）
* OpenCV (用于图像预处理、可视化裁剪等)
* GLib (DeepStream 依赖)
* yaml-cpp (解析 `config_sot.yml`，若未安装请自行安装)

可能需要的系统包 (按发行版调整)：
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libglib2.0-dev libyaml-cpp-dev
```

确保 DeepStream 安装后其 `lib` 与 `includes` 可被找到（或修改 CMakeLists 中的 `DS_ROOT_DIR`）。

---
## 5. 编译与安装
进入 `src/sot_plugin` 目录后：

```bash
# 1. 生成构建目录
mkdir -p build && cd build

# 2. 运行 CMake（如 DeepStream 不在默认路径，添加 -DDS_ROOT_DIR=/your/path）
cmake ..

# 3. 编译
sudo cmake --build .

# 4. (可选) 安装到 DeepStream 插件目录 (CMake 中已设定 GST_PLUGINS_DIR)
sudo cmake --install .

# 结果：生成 libsot.so (同时复制到 /opt/nvidia/deepstream/deepstream/lib)
```

编译选项说明（在 `CMakeLists.txt` 中）：
* `CMAKE_CXX_STANDARD 14`
* 打开警告：`-Wall -Wextra`
* 生成位置：`${PROJECT_SOURCE_DIR}/lib/libsot.so`

如需调试：可增加 `-O0 -g` 或使用 `set(CMAKE_BUILD_TYPE Debug)`。

---
## 6. 模型转换 (ONNX -> TensorRT Engine)
`models/` 目录中已放置示例 ONNX 与部分已转换的 `.engine` 文件；若需重新转换：

```bash
cd models
# 示例脚本（内部通常调用 trtexec）
bash convert2trt.sh your_model.onnx output.engine fp16
```

注意：
* 转换时请匹配部署 GPU 的 Compute Capability。
* FP16 引擎需 GPU 支持半精度；否则请使用 FP32。
* MixFormerV2 / OSTrack / SuTrack 的输入尺寸、动态 shape 需与脚本或代码一致。

---
## 7. 配置文件说明 (`config_sot.yml`)
示例：
```yaml
BaseConfig:
  modelName: 2                 # 0=sutrack, 1=ostrack, 2=mixformer_v2
  modelFilePath: /abs/path/mixformerv2_online_base_fp16.engine
  enableTrackCenterStable: 1   # 是否启用跟踪中心稳定性判定
  trackCenterStablePixelThreshold: 3

TargetManagement:
  expandFactor: 1.0
  probationAge: 2
  maxMiss: 10
  scoreThreshold: 0.3
  iouThreshold: 0.5
  trackBoxWidthThreshold: 0.3
  trackBoxHeightThreshold: 0.3
  maxTrackAge: 30

MixformerV2Config:
  update_interval: 100
  max_score_decay: 0.95
  template_update_score_threshold: 0.5
```

字段含义：
* BaseConfig.modelName：选择具体模型枚举（见第 2 节）。
* BaseConfig.modelFilePath：TensorRT Engine 绝对路径（或相对工作目录）。
* enableTrackCenterStable：启用后，会统计跟踪中心位置波动，过滤稳定但可能错误的目标。
* trackCenterStablePixelThreshold：中心像素标准差阈值，越小越严格。
* TargetManagement.expandFactor：初始化 / 更新时对目标框进行放大裁剪的比例。
* probationAge：达到该“年龄”后才认定为稳定跟踪（可避免一开始噪声）。
* maxMiss：连续多少帧未成功更新则判定丢失。
* scoreThreshold：模型输出分数低于该值可视为无效/低可信。
* iouThreshold：当前预测与上一帧 / 检测框的匹配 IOU 下限。
* trackBoxWidth/HeightThreshold：用于限制预测框与初始尺度差异的比例（防止尺度漂移）。
* maxTrackAge：内部保存的历史长度 / 生命周期上限。
* MixformerV2Config.update_interval：模板在线更新的帧间隔。
* MixformerV2Config.max_score_decay：最大历史分数的衰减系数（防止一直保持旧峰值）。
* template_update_score_threshold：达到该分数且到达更新间隔才会刷新在线模板。

---
## 8. 在 DeepStream 中的使用示例
在 DeepStream 的应用配置（例如 `deepstream_app_config.txt` 或自定义管线）中，通常 tracker 段类似：
```ini
[tracker]
enable=1
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libsot.so
ll-config-file=/workspace/deepstream-app-custom/src/sot_plugin/config_sot.yml
tracker-width=0    # 由内部模型自行处理，可保持 0
tracker-height=0
```

说明：
* `ll-lib-file` 指向安装后的 `libsot.so`。
* `ll-config-file` 指向本文档的 YAML 配置。
* 若你的 DeepStream Pipeline 仅需该单目标跟踪，可在逻辑上先设定一个初始化框（例如通过鼠标交互或第一帧检测输出）再交给插件维护。

初始化策略（代码侧）可在接到第一帧含检测或用户指定 ROI 时调用 `init`，后续每帧调用 `processFrame` / `track` 接口。

---
## 9. 调试与测试脚本
`models/test_model.py`：
* 可用于加载 ONNX/Engine 做一次性推理验证（请根据脚本内部实现调整）。

`models/test_engine.sh`：
* 调用 `trtexec` 或内部工具对 engine 做性能基准测试。

典型调试要点：
1. 确认引擎输入输出 Tensor 名称与代码中一致（见各 `*_trt.h` 中的 `input_` / `output_` 字符串）。
2. 如出现维度不匹配，重新用正确的 `--optShapes/--minShapes/--maxShapes` 生成动态形状引擎或使用固定尺寸。
3. 打开 `gdb --args <your deepstream app>` 进行崩溃定位。

---
## 10. 常见问题 (FAQ)
Q: 运行时报找不到 `libsot.so`？
A: 确认已执行 `sudo make install` 或在环境变量 `LD_LIBRARY_PATH` 中加入编译输出的 `lib/` 路径。

Q: 性能不达预期？
A: 检查是否使用 FP16 引擎；确认无不必要的图像拷贝；使用 `nvidia-smi dmon` 观察显存与 GPU 利用率。

Q: 想切换模型？
A: 修改 `config_sot.yml` 中 `modelName` 与 `modelFilePath`，保证引擎文件与实际模型结构匹配。

Q: MixFormerV2 模板一直不更新？
A: 检查当前最大分数是否达到 `template_update_score_threshold` 且是否满足 `update_interval`。

---
## 11. 许可证
本项目基于 MIT License，详情参见根目录 `LICENSE` 文件。

---
## 进一步计划 (TODO)
* 支持多实例并行跟踪（当前核心面向单对象，可扩展多个上下文）
* 引入半精度 / INT8 自动校准脚本

---
如有问题或改进建议，欢迎提交 Issue / PR。

<div align="right">—— END ——</div>

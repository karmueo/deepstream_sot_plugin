# SOT 四路验收记录（2026-09-11）

## 环境与产物

- DeepStream 7.1，TensorRT 10.3，Jetson Orin。
- 测试配置基于 `src/deepstream-app/configs/yml/app_config.yml`，仅在临时副本中关闭 `sink2` 并将 `ll-lib-file` 指向仓库构建产物。
- 正式 `file_sources.csv` 未被测试过程修改。
- 运行日志确认加载：`sot_plugin/lib/libsot.so`。

## 自动测试

```bash
cmake -S sot_plugin -B sot_plugin/build -DSOT_BUILD_TESTS=ON
cmake --build sot_plugin/build -j4
ctest --test-dir sot_plugin/build --output-on-failure
cmake --build build -j4
```

结果：3/3 SOT 测试通过，`deepstream-app` 构建通过。

## 四路同源进程测试

- 输入：仓库当前 `src/deepstream-app/configs/yml/file_sources.csv`（四路 `test.mp4`）。
- 持续时间：65 秒；由 SIGINT 结束，输出 `App run successful`。
- 最终累计 FPS：43.18、43.09、43.28、43.34，平均 43.22 FPS/路。
- frame 1080、1440、1800 的四路 tracking ID、age、miss、bbox、score 完全一致；未发现同一 batch 内串行增长。
- 未发现崩溃、double-free、CUDA error、nvtracker fatal error 或 TensorRT 跨设备型号警告。

按性能日志采样的帧间隔：median 23.436 ms，P95 27.871 ms。

## 差异化串扰测试

测试过程复制 CSV 到 `/tmp`，只把第 2 路替换为 `bird.mp4`，未覆盖仓库文件。第 2 路在 frame 300/600/900 的状态分别为 ID 5、age 132/432/732；其余三路保持完全一致的 ID 4/6/9、age 35/172/58，证明模板、ID、age、miss 和重置状态按 stream 隔离。

## Nsight Systems / NVTX

split 模式设备特征直连后的 15 秒采样：

| NVTX 段 | Median | P95 |
|---|---:|---:|
| crop/resize/RGBA→BGR-CHW | 0.464 ms | 1.179 ms |
| search backbone（设备输出） | 2.321 ms | 9.646 ms |
| head（设备输入） | 2.808 ms | 12.526 ms |
| CPU 后处理 | 0.015 ms | 0.020 ms |
| 每帧最终同步 | 0.078 ms | 0.114 ms |

## Engine 校验

- 三个 NanoTrack FP16 engine 已在当前设备重建并通过 `trtexec --loadEngine --skipInference`。
- YOLO engine 通过 `trtexec` 校验，无跨设备警告；I/O 为 input `(4,3,352,640)`、output `(4,4620,6)`。
- 可使用仓库根目录 `rebuild_sot_yolo_engines.sh` 原子重建并逐个验证四个 engine。

## 安装状态

当前会话无免密 sudo，`sudo -n cmake --install sot_plugin/build` 返回“a password is required”。因此 `/opt/nvidia/deepstream/deepstream/lib/libsot.so` 仍是旧文件；正式运行前必须由有权限的用户执行安装并再次比较 SHA-256。

## 后续第二阶段边界

本次已完成 split backbone→head 设备特征直连、模板特征驻留和每帧一次同步。`NVBUF_MEM_SURFACE_ARRAY` 的 CUDA/EGL 映射预处理、GPU penalty/window/argmax 后处理，以及跨 stream 共享只读 `ICudaEngine` 尚未合入；当前仍在 CPU 小区域生成 CHW，并只回传 head 的小型输出张量。以上项目应在独立 CUDA 源文件和显存回归测试齐备后继续实施。

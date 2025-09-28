import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# 1. 定义输入数据（与模型匹配）
dummy_template = np.random.randn(1, 3, 112, 112).astype(np.float32)  # 假设模板图像
dummy_search = np.random.randn(1, 3, 224, 224).astype(np.float32)    # 假设搜索图像
dummy_template_anno = np.random.randn(1, 4).astype(np.float32)  # 假设模板标注

# =================================================================
# 2. ONNX Runtime 推理（基准输出）
# =================================================================
onnx_session = ort.InferenceSession("sutrack.onnx", providers=["CUDAExecutionProvider"])
ort_inputs = {'template': dummy_template, 'search': dummy_search, 'template_anno': dummy_template_anno}
ort_outs = onnx_session.run(None, ort_inputs)
print("onnx outputs:")
print(ort_outs)

# =================================================================
# 3. TensorRT 引擎推理（需提前通过 trtexec 生成 model.engine）
# =================================================================
def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(engine_data)
    

engine = load_engine("sutrack_fp32.engine")
context = engine.create_execution_context()

# =================================================================
# 3. 准备输入/输出的 GPU 内存
# =================================================================
# 获取所有绑定的名称和形状
bindings = []
num_io_tensors = engine.num_io_tensors
for i in range(num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(f"Binding {i}: name={name}, dtype={dtype}, shape={shape}")
    bindings.append((name, dtype, shape))

# 分配输入/输出的 GPU 内存
inputs = {
    "template": dummy_template,
    "search": dummy_search,
    "template_anno": dummy_template_anno
}

# 创建 GPU 内存缓冲区
buffers = []
for name, dtype, shape in bindings:
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        # 输入绑定：从 inputs 字典中获取数据
        data = inputs[name]
        buffer = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod(buffer, data)
    else:
        # 输出绑定：分配空内存
        buffer = cuda.mem_alloc(int(np.prod(shape) * np.dtype(np.float32).itemsize))
    context.set_tensor_address(name, int(buffer))  # 新增：设置 tensor 地址
    buffers.append(buffer)

# =================================================================
# 4. 执行推理
# =================================================================
stream = cuda.Stream()
context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()


# =================================================================
# 5. 读取输出结果
# =================================================================
outputs = {}
for i, (name, dtype, shape) in enumerate(bindings):
    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        # 从 GPU 拷贝输出数据到 CPU
        output_data = np.empty(shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_data, buffers[i])
        outputs[name] = output_data
        print(f"Output {name}: {output_data}")
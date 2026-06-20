# 数据格式约定

跨设备共享的中间产物必须遵循以下 JSON Schema。

## 1. GPU Ground Truth 输出

每次 benchmark 运行产出一个 JSON 文件，放在 `artifacts/gpu-ground-truth/<suite>/<test_case_id>.json`。

```json
{
  "test_case_id": "bench_v1.0.0/L1/conv2d/001",
  "level": "L1",
  "operator": "torch.nn.Conv2d",
  "input_shape": [1, 3, 224, 224],
  "dtype": "float32",
  "seed": 42,
  "output": {
    "tensor_shape": [1, 64, 112, 112],
    "tensor_binary_path": "artifacts/gpu-ground-truth/L1/conv2d_001_output.pt"
  },
  "intermediate": {
    "activations": [
      {"layer": "conv", "shape": [1, 64, 112, 112], "binary_path": "..."}
    ]
  },
  "gradients": {
    "layers": [
      {"name": "weight", "shape": [64, 3, 7, 7], "binary_path": "..."},
      {"name": "bias", "shape": [64], "binary_path": "..."}
    ]
  },
  "environment": {
    "python": "3.10.14",
    "pytorch": "2.3.0",
    "cuda": "12.1",
    "gpu": "NVIDIA A100",
    "timestamp": "2026-06-15T10:00:00Z"
  }
}
```

## 2. NPU 比对结果

NPU 服务器跑完桥接器测试后，产出比对结果：

```json
{
  "test_case_id": "bench_v1.0.0/L1/conv2d/001",
  "bridge": "torch4ms",
  "track": "intercept",
  "reference_path": "artifacts/gpu-ground-truth/L1/conv2d_001_output.pt",
  "tier1": {
    "passed": true,
    "allclose": {"atol": 1e-5, "rtol": 1e-5, "result": true},
    "mae": 1.3e-5,
    "p95": 2.1e-5,
    "p99": 5.0e-5,
    "cosine_similarity": 0.9998
  },
  "tier2": {
    "fne_per_layer": [
      {"layer": "conv", "cosine": 0.9995}
    ],
    "gc_per_layer": [
      {"name": "weight", "cosine": 0.998}
    ],
    "first_deviated_layer": null
  },
  "environment": {
    "python": "3.10.14",
    "mindspore": "2.3.1",
    "cann": "8.0.RC1",
    "npu": "Ascend 910B",
    "torch4ms_version": "0.3.1",
    "timestamp": "2026-06-15T12:00:00Z"
  }
}
```

## 3. 张量二进制格式

- 使用 `torch.save(tensor, path)` 保存
- 文件名以 `.pt` 结尾
- 如需跨设备传输，二进制文件放在 `artifacts/` 下对应目录，Git 不追踪（`.gitignore`）
- 路径约定：`artifacts/<suite>/<test_case_id>_<tensor_name>.pt`

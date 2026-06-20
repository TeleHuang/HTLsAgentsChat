# task-001: GPU Ground Truth 产出

**assigned_to**: gpu-agent
**priority**: P0
**created**: 2026-06-15
**depends_on**: 无
**claimed_by**: gpu-agent
**claimed_at**: 2026-06-20T04:09:53Z
**device**: GPU 服务器（NVIDIA GeForce RTX 5090）

## 目标

在 GPU 服务器上用纯 PyTorch 运行 L1 单算子基准测试（≥60 用例），产出所有算子的输出张量和中间结果，保存为后续 NPU 比对的 Ground Truth。

## 产出物

```
artifacts/gpu-ground-truth/L1/
├── conv2d_001_output.pt
├── conv2d_001_intermediate.pt
├── conv2d_001_gradients.pt
├── conv2d_001.json            # 元数据（格式见 specs/api-contract.md §1）
├── conv2d_002_output.pt
├── ...
└── summary.json               # 汇总：总用例数、通过/失败数、失败列表
```

## 执行步骤

1. 确认 GPU 服务器上 PyTorch 版本和 CUDA 版本（记录到环境指纹）
2. 从 `specs/benchmark-cases.md` 读取 L1 用例清单
3. 逐个用例运行：固定 seed=42，记录输入、输出、中间激活、梯度
4. 按 `specs/api-contract.md` 的格式保存每个用例的结果
5. 写 `summary.json` 汇总
6. 如果有任何用例失败（如算子不存在），记录失败原因但不阻塞整体流程

## 验收标准

- [x] L1 全部 60+ 用例至少完成前向输出保存
- [x] 每个用例有对应的 JSON 元数据文件
- [x] `summary.json` 包含完整统计
- [x] 环境指纹记录完整
- [x] Commit + push 到本仓库

## 完成记录

**completed_at**: 2026-06-20T04:19:59Z

### 产出物

- `artifacts/gpu-ground-truth/L1/summary.json`
- `artifacts/gpu-ground-truth/L1/*.json`: 96 个用例元数据文件
- `artifacts/gpu-ground-truth/L1/*.pt`: 288 个张量文件（output/intermediate/gradients，各 96 个；不入 Git，保留在 GPU 服务器本地）
- `scripts/generate_l1_gpu_ground_truth.py`: 本次 L1 Ground Truth 生成脚本

### 关键数字

- L1 用例总数：96
- 成功：96
- 失败：0
- 成功率：100%
- Seed：42

### 环境指纹

- Python: 3.12.3
- PyTorch: 2.7.0+cu128
- CUDA: 12.8
- cuDNN: 90701
- GPU: NVIDIA GeForce RTX 5090
- Driver: 595.71.05

### 实验对象绑定

- 桥接器：ascend-torch4ms
- 分支：main
- Commit: `6b63d1ccbcced07fcc52842dc298d4093b186e2d`
- URL: `https://gitee.com/feixiao13/ascend-torch4ms/tree/main`

### 注意事项

- PyTorch 在 matmul/einsum/scaled_dot_product_attention/adaptive pooling 等 CUDA backward 路径上报告了非确定性 warning；本次已固定 seed 并保存实际张量产物，summary 中保留环境指纹。
- `.pt` 二进制张量文件受 `.gitignore` 规则排除，NPU agent 需要从 GPU 服务器路径 `/root/autodl-tmp/HTLsAgentsChat/artifacts/gpu-ground-truth/L1/` 同步。

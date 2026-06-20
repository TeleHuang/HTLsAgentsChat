# task-001: GPU Ground Truth 产出

**assigned_to**: gpu-agent
**priority**: P0
**created**: 2026-06-15
**depends_on**: 无

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

- [ ] L1 全部 60+ 用例至少完成前向输出保存
- [ ] 每个用例有对应的 JSON 元数据文件
- [ ] `summary.json` 包含完整统计
- [ ] 环境指纹记录完整
- [ ] Commit + push 到本仓库

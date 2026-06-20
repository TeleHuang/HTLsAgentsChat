# task-002: NPU torch4ms L1 比对

**assigned_to**: npu-agent
**priority**: P0
**created**: 2026-06-15
**depends_on**: task-001 (需要 GPU Ground Truth 产出后才能开始)

## 目标

在 NPU 服务器上跑 torch4ms 桥接器，运行 L1 单算子基准测试，将结果与 GPU Ground Truth 进行 Tier-1 比对。

## 前置条件

- task-001 完成，`artifacts/gpu-ground-truth/L1/` 目录有完整数据
- NPU 服务器上 torch4ms 已安装且 Coding Bot 可正常运行

## 产出物

```
artifacts/npu-results/torch4ms/L1/
├── conv2d_001.json            # 比对结果（格式见 specs/api-contract.md §2）
├── conv2d_002.json
├── ...
└── summary.json               # 汇总：Tier-1 pass rate, MAE/P95/P99/Cosine 分布
```

## 执行步骤

1. `git pull` 拉取 GPU Ground Truth 的元数据文件（JSON 部分）
2. 从 GPU 服务器同步张量二进制文件（`.pt`）——如果 Git LFS 不可用，用 scp 手动传输并在任务文件中记录路径
3. 确认 NPU 环境：MindSpore 版本、CANN 版本、torch4ms 版本
4. 逐个 L1 用例：通过 torch4ms 运行 → 获取输出 → 与 GPU Ground Truth 做 Tier-1 比对
5. 按 `specs/api-contract.md` 格式保存每个用例的比对结果
6. 写 `summary.json` 汇总
7. 如果 Tier-1 失败，记录失败分类（OperatorNotFound / TypeMismatch / etc.）

## 验收标准

- [ ] L1 全部 60+ 用例完成比对
- [ ] 每个用例有对应的比对结果 JSON
- [ ] `summary.json` 包含：总通过率、按算子类型的通过率分布、MAE/P95/Cosine 统计
- [ ] 失败用例有明确的失败分类
- [ ] Commit + push

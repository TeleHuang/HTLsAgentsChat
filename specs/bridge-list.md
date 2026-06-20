# 被评测桥接器清单

## 赛道 A：运行时拦截

PyTorch 源码运行时不变，算子调用被拦截并路由到非 PyTorch 后端。

| # | 桥接器 | 简介 | 安装方式 | 状态 |
|---|--------|------|---------|------|
| 1 | **torch4ms** | 基于 `__torch_dispatch__` 的兼容层，运行时拦截 ATen 算子 | 源码安装 | ✅ 可用 |
| 2 | **torch-npu** | 带 Ascend NPU 后端的 PyTorch（硬件原生基线） | pip | 待引入 |
| 3 | **mindtorch** | torch.* → mindspore.* 静态 API 命名空间映射 | pip | 待引入 |
| 4 | **mindtorch_v2** | mindtorch 的升级版本 | pip | 待引入 |
| 5 | **mindnlp_patch** | 基于规则的 AST 级代码转换 | 源码安装 | 待引入 |

## 赛道 B：代码翻译

PyTorch 测试用例 → 翻译为 MindSpore 代码 → 原生执行。

| # | 方案 | 简介 | 状态 |
|---|------|------|------|
| 1 | **LLM-translate** | 非结构化 LLM 直接翻译（temperature=0.0, DeepSeek-Chat） | 待实现 |
| 2 | **ONNX-route** | PyTorch → ONNX → MindSpore 中间格式（远期） | 暂缓 |

## 优先级

- **P0**（当下）: torch4ms（主力研究对象）
- **P1**（后续）: mindtorch, mindtorch_v2, mindnlp_patch, LLM-translate
- **P2**（远期）: torch-npu, ONNX-route

# GPU Agent 日志

设备：GPU 服务器（NVIDIA A100 / NVIDIA GeForce RTX 5090）
Coding Bot：Claude Code

---

## 2026-06-24

### GPU 侧 canonical reference 对照实验

- 当前设备：GPU 服务器（NVIDIA GeForce RTX 5090）
- 已拉取 TBBCC 远端新版本到 `26a9284 feat: add canonical GPU reference workflow`
- 按指引运行 `scripts/tbbcc_gpu_reference.py --suite benchmarks/v1.0.0/suites/all_noop.json --out reports/gpu_reference_all_noop --device cuda`
- 首轮采集发现 175 cases 中 18 个 PyTorch reference 失败；定位为 benchmark 生成逻辑问题，不是 bridge 或硬件问题：
  - AMP_FP16 训练 case 将模型参数转为 FP16 后使用 GradScaler，触发 `Attempting to unscale FP16 gradients`
  - YOLO 训练 target 尺寸为 20x20，但模型 head 输出为 10x10
  - UNet bilinear upsample decoder 拼接通道数配置错误
- 已在 TBBCC 修复 `scripts/generate_benchmark_library.py` 并重新生成 canonical cases
- 修复后重新采集 GPU reference：175/175 passed，0 failed，耗时 372.673s
- 对齐检查通过：`direct_overlap_count=175`，`mapping_required=false`
- 产物目录：`/root/autodl-tmp/TBBCC/reports/gpu_reference_all_noop/`，大小约 156M（受 TBBCC `.gitignore` 的 `reports/` 规则忽略，未提交）
- 环境：Python 3.12.3, PyTorch 2.7.0+cu128, CUDA 12.8, cuDNN 90701, RTX 5090, capability 12.0
- 验证：`claude plugin validate .` 通过；`python -m pytest tests -q` 为 23 passed, 1 skipped
- TBBCC 本地提交：`957b84226218a3340ec1c6f8b6a1f2902eabf833 fix: make GPU reference all_noop pass on RTX 5090`
- TBBCC push 暂未成功：GitHub HTTPS 连接多次在 443 端口超时或 TLS 断连

## 2026-06-20

### 会话启动

- 当前设备：GPU 服务器（NVIDIA GeForce RTX 5090）
- 已执行 `git pull`，仓库状态为最新
- 已读取 `README.md`、`CLAUDE.md`、`tasks/open/`、`logs/gpu-agent.md`
- 已认领 `task-001: GPU Ground Truth 产出`
- 当前状态：task-001 执行中，下一步确认 PyTorch/CUDA 环境并生成 L1 Ground Truth

### task-001 完成

- 已生成 L1 PyTorch Ground Truth：96/96 passed，0 failed
- 产物目录：`artifacts/gpu-ground-truth/L1/`
- 元数据：96 个用例 JSON + `summary.json`
- 张量：288 个 `.pt` 文件（output/intermediate/gradients；不入 Git）
- 环境：Python 3.12.3, PyTorch 2.7.0+cu128, CUDA 12.8, RTX 5090, Driver 595.71.05
- 绑定实验对象：ascend-torch4ms main `6b63d1ccbcced07fcc52842dc298d4093b186e2d`
- 注意：部分 CUDA backward 路径有 PyTorch 非确定性 warning，已固定 seed=42 并保存实际张量

## 2026-06-15

### 会话启动

- 仓库已克隆
- 等待总指挥分配任务
- 当前状态：就绪，无进行中任务

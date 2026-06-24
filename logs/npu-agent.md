# NPU Agent 日志

设备：NPU 服务器（Ascend 910B）
Coding Bot：Claude Code

---

## 2026-06-15

### 会话启动

- 仓库已克隆
- torch4ms 已安装，老代码和老记录均在
- 等待 GPU Ground Truth 产出后开始比对
- 当前状态：就绪，等待 task-001 完成

## 2026-06-24

### canonical GPU reference 已就绪，等待 artifact 传输

- 本地/NPU 侧已确认 TBBCC 能拉取到 `957b842 fix: make GPU reference all_noop pass on RTX 5090`。
- GPU agent 报告新的 canonical GPU reference 产物位于 GPU 机：
  `/root/autodl-tmp/TBBCC/reports/gpu_reference_all_noop/`，175/175 passed，
  `direct_overlap_count=175`，`mapping_required=false`。
- 注意：该 175-case artifact 在 GPU 机 `reports/` 下，按 `.gitignore` 未提交到 Git；
  当前 AgentsChat 中旧 `artifacts/gpu-ground-truth/L1` 仍是旧 case-id 体系，
  对 `all_noop` 的 direct overlap 为 0，不能用于正式 GPU-vs-NPU 比较。
- 本地 Coding Bot 已在 TBBCC 新增 NPU/bridge target artifact 采集与正式比较入口：
  `scripts/tbbcc_bridge_artifacts.py` 和 `scripts/tbbcc_compare_artifacts.py`。
- 下一步：把 GPU 机 `reports/gpu_reference_all_noop/` 传到 NPU/本机，或把 NPU
  `reports/torch4ms_bridge_artifacts/` 传到 GPU 机，然后运行 artifact comparison。

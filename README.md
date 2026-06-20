# HTLsAgentsChat

多设备 Agent 协同工作台 —— TorchBridgeBench 项目的分布式指挥中心。

## 一句话定位

我在 GPU 服务器、NPU 服务器和本地开发机上各部署了一个 Coding Agent（Claude Code）。这个仓库是它们的**共享黑板**：每个 Agent 在这里汇报进度、认领任务、传递中间产物，我（总指挥）在这里发号施令。

## 项目背景：TorchBridgeBench

**TorchBridgeBench** 是一个面向 PyTorch → MindSpore（华为昇腾 NPU）深度学习框架迁移的**多维度自动评估基准**。

### 核心命题

> 如何系统性地评估"把 PyTorch 模型迁移到 MindSpore 跑在 NPU 上"这个工程任务的质量和代价？

### 两篇论文

| | 论文 1（辅助） | 论文 2（主攻） |
|---|---|---|
| 主题 | 分层反馈修复 Agent | 多维度自动评估 Benchmark |
| 角色 | 桥接器内部的修理工 | 桥接器用户的标准化替身 + 评估框架 |
| 产出 | 执行/数值/梯度三层次反馈 + torch4ms 原型 | 四层指标体系 + 全自动 Pipeline + 标准化评测集 |

### 双赛道架构

- **赛道 A（拦截重定向，5 方案）**：torch4ms / torch-npu / mindtorch / mindtorch_v2 / mindnlp_patch —— PyTorch 代码运行时不变，底层算子被路由到非 PyTorch 后端
- **赛道 B（代码翻译，1+ 方案）**：LLM 直接翻译 PyTorch → MindSpore 源码

### 四层指标体系

| 层 | 指标 | 度量什么 |
|---|------|---------|
| 执行层 | OCR, MER | 代码能不能跑 |
| 数值层 | FNE, GC | 前向输出和反向梯度是否对齐 |
| 训练层 | TCA_ℓ, TCA_a | 训练曲线和最终精度是否收敛一致 |
| 过程层 | ME, AR | 桥接器给用户造成了多大负担、省了多少事 |

### 当前阶段

- **设计**：~90%（论文草稿、指标体系、执行规约、状态机、JSON Schema 全部就位）
- **编码**：~50%（核心框架有，effort/migrate@k/DTW 等关键模块是占位字段）
- **实验**：~0%（GPU/NPU 服务器就绪，对比实验尚未执行）

---

## 设备拓扑

```
┌─────────────────────────────────────────────────────────────┐
│                    总指挥 (Tele Huang)                       │
│              Vault: Bridgebench (Obsidian)                   │
│              本地 Coding Bot: Claude Code                     │
└──────────┬────────────────────────┬──────────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐    ┌──────────────────────┐
│   GPU 服务器       │    │   NPU 服务器 (Ascend)  │
│   PyTorch 原生     │    │   MindSpore + 桥接器    │
│                   │    │                       │
│   Coding Bot: ✅   │    │   Coding Bot: ✅       │
│   任务:            │    │   任务:                │
│   · 产出 Ground    │    │   · 运行桥接器测试      │
│     Truth 参考数据  │    │   · 与 GPU 结果比对     │
│   · 纯 PyTorch     │    │   · 报告 Tier-1/2/3    │
│     基准测试        │    │     通过率              │
└──────────────────┘    └──────────────────────┘
```

---

## 仓库结构

```
HTLsAgentsChat/
├── README.md              # 本文件 —— 项目总览
├── CLAUDE.md              # 本仓库自身的 Agent 指令
├── tasks/                 # 任务池 —— 总指挥发布任务，Agent 认领
│   ├── open/              # 待认领
│   ├── in-progress/       # 执行中
│   └── done/              # 已完成
├── artifacts/             # 跨设备共享的中间产物
│   ├── gpu-ground-truth/  # GPU 服务器产出的 PyTorch 参考数据
│   ├── npu-results/       # NPU 服务器产出的比对结果
│   └── reports/           # 汇总报告
├── logs/                  # 各 Agent 的运行日志
│   ├── gpu-agent.md
│   └── npu-agent.md
└── specs/                 # 共享规范
    ├── benchmark-cases.md # 基准测试用例规格
    ├── api-contract.md    # 数据格式约定（JSON Schema）
    └── bridge-list.md     # 被评测桥接器清单
```

---

## 开发工作流

### 总指挥发布任务

1. 在 `tasks/open/` 下创建任务文件，格式：
   ```markdown
   # task-001: GPU Ground Truth 产出
   assigned_to: gpu-agent
   priority: P0
   created: 2026-06-15
   ```
2. 任务内容包含明确的可验证产出物和路径约定

### Agent 工作循环

1. **读黑板**：Agent 启动后先 `git pull`，读 `tasks/open/` 看有没有自己的任务
2. **认领**：把任务文件从 `open/` 移到 `in-progress/`，标注开始时间
3. **执行**：按任务规格执行，产出放入 `artifacts/` 对应目录
4. **汇报**：更新任务文件状态，写入 `logs/<agent-name>.md`
5. **提交**：`git commit` + `git push`，让其他 Agent 可见

### 跨设备数据流

```
GPU 服务器                            NPU 服务器
─────────                            ─────────
python run_benchmark.py               git pull  # 拉取 GPU 产出
  --suite L1                          python run_benchmark.py
  --track pytorch-native                --suite L1
  --output artifacts/gpu-ground-truth/  --bridge torch4ms
                                        --reference artifacts/gpu-ground-truth/
git push                                --output artifacts/npu-results/
                                      git push

              ┌──────────────────────────┘
              ▼
         总指挥 / 本地 Agent
         读取两边结果，生成对比报告
         写入 artifacts/reports/
```

### 提交规范

- Commit message 格式：`<设备名>: <动作> —— <结果摘要>`
- 示例：`gpu-agent: L1 suite complete —— 58/60 passed, 2 failed (torch.svd, complex ops)`
- 大文件（>10MB 的张量数据）用 Git LFS 或放入 `artifacts/` 后通过 `.gitignore` 排除，改为在任务文件中记录路径

---

## 当前待办 (2026-06-15)

| 优先级 | 任务 | 负责人 | 状态 |
|--------|------|--------|------|
| **P0** | GPU 服务器产出 L1 Ground Truth（≥60 单算子 PyTorch 参考数据） | gpu-agent | ⬜ 待启动 |
| **P0** | NPU 服务器跑 torch4ms L1 比对（与 GPU Ground Truth 对比） | npu-agent | ⬜ 待启动 |
| P1 | 补齐 effort_adapt/effort_repair 真实计算（替代占位字段） | local-agent | ⬜ |
| P1 | migrate@k 饱和曲线 + 置信区间流水线封装 | local-agent | ⬜ |
| P1 | 自动分类对齐论文六类（当前代码分类 >6 类） | local-agent | ⬜ |
| P2 | DTW loss curve 实现（当前仅为通用 TASK_METRICS） | local-agent | ⬜ |

---

## 相关资源

- 项目 Wiki Vault：`C:\Users\huang\Documents\ObsVaults\Bridgebench\`
- 论文 LaTeX 草稿：`D:\Workspace\Mindspore\IEEE_Conference_Template__2_\torchbridgebench-ieee.tex`
- 核心代码库：`D:\Workspace\Mindspore\torchbridgebench\`
- CCplugin 代码库：（GPU/NPU 服务器上）

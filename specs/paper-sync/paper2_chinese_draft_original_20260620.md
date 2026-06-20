# TorchBridgeBench 论文草稿 — 中文译文

> 本文档是 `paper-draft-v2.tex` 的完整中文翻译，按章节对应。
> LaTeX 原稿位于 `ACM_Conference_Proceedings_Primary_Article_Template/paper-draft-v2.tex`。
> 红色中文注释 (`\cnnote` / `\cnfigure`) 指示了需要替换为我们项目实际图表的位置。

---

## 标题

**TorchBridgeBench：一个面向跨框架深度学习代码迁移的多维度自动评估 Benchmark**

## 摘要

深度学习框架迁移已成为工业部署中的常见需求，但现有的评估实践对框架特定语义仍然碎片化且缺乏系统性。CodeBLEU 和 pass@k 等代码翻译指标仅衡量语法相似性和单元测试通过率，无法检测仅在训练过程中才显现的数值和梯度级语义差异。

本文提出 TorchBridgeBench，一个面向 PyTorch→MindSpore 代码迁移的多维度、全自动评估框架。我们的框架引入：(1) 覆盖执行可行性、数值等价性和训练收敛对齐的三层指标体系；(2) 具备四阶段验证的全自动评估流水线；(3) 利用 LLM 修复 Agent 量化迁移代价和自动化率的 Agentic 评估环；(4) 覆盖模型多样性、训练组件多样性和受控失败模式的资产化 Benchmark。

我们对五种迁移方案进行横向对比，包括运行时拦截（torch4ms）、直接 API 映射（mindtorch、mindtorch_v2）、基于补丁的重写（mindnlp_patch）和硬件原生执行（torch-npu）。实验结果展示了 TorchBridgeBench 在多个维度上有效区分迁移质量的能力，且 Agentic 评估揭示了每种迁移范式自动化潜力的可操作洞见。

## 1. 引言

跨框架深度学习迁移已成为工业和学术界的常规需求，这源于部署约束、硬件适配要求和生态系统偏好。PyTorch→MindSpore 迁移是一个代表性的案例：源框架与目标框架在运行时模型（Eager vs. 图执行）、自动微分（AD）语义和算子覆盖上存在根本性差异——这使得迁移质量评估天然是多维度的。

现有的迁移评估主要遵循 CodeBLEU 和 pass@k 等通用代码翻译指标。这些指标衡量的是语法相似性和功能通过率，但对框架特定的语义等价性却是盲区。在实践中，一个迁移后的模型可能通过了所有单元测试，却在不经意间产生了系统性梯度偏差，导致训练不稳定或收敛失败。MindSpore 官方迁移指南已经认识到这一差距，建议进行手动的逐层前向验证和梯度检查——但这一过程劳动密集、标准薄弱，且在不同方法间不可复现。

近期在 DL 框架测试方面的工作取得了显著进展（Audee、∇Fuzz、NNSmith、SORT、ModelMeta），但这些工作聚焦于在特定框架中**发现 bug**。目前仍然缺失的是一个系统化、可量化、可复现的**兼容度度量**框架，能够对不同迁移方案进行公平的横向对比。

我们填补三个空白：

- **指标空白**：缺乏统一的指标体系来捕获跨框架迁移的执行、数值和训练级语义等价性。
- **工具空白**：现有的验证流程是手动的、不可扩展的，且在不同方法间缺乏一致性。
- **Benchmark 空白**：缺乏标准化的 Benchmark 支持跨迁移范式（运行时拦截、API 映射、补丁重写、LLM 翻译）的公平对比。

我们的贡献：

- 一个**三层评估指标体系**（执行层、数值层、训练层），在多个语义层级上捕获迁移质量。
- 一个集成算子级、模型级和训练级验证与结构化失败分类的**全自动评估流水线**。
- 一个**Agentic 评估环**，利用基于 LLM 的修复 Agent 通过蒙特卡洛采样量化迁移代价（ME）和自动化率（AR），实现动态 Benchmark 评估。
- 一个覆盖 10+ 模型架构、多种训练组件和受控失败模式的**资产化 Benchmark**，带有标准化的 ground-truth 标注。

## 2. 相关工作

### 2.1 DL 框架测试

先前的 DL 框架测试工作可分为模型级和 API 级方法。CRADLE、LEMON、Muffin 等模型级方法执行整个 DL 模型以检测跨后端或跨框架的不一致性。虽然这些方法在暴露集成级 bug 方面有效，但由于涉及大量 API，故障定位困难。FreeFuzz、DocTer、DeepREL 等 API 级方法使用自动生成的输入测试单个 API。然而，它们受到有效输入率低的困扰，且很大程度上忽略了会放大微小数值差异的 API 交互。

SORT 提出使用从热门模型中挖掘的频繁 API 交互模式进行子图级测试，实现了 100% 的有效输入生成率。ModelMeta 引入模型级蜕变测试，设计了四种基于结构的蜕变关系，检测到先前方法遗漏的资源使用和效率 bug。∇Fuzz 专门针对 DL 库中的自动微分，在 PyTorch、TensorFlow、JAX 和 OneFlow 中检测到 173 个 bug。

我们的工作在两个方面与上述不同：(1) 我们针对的是**跨框架迁移质量**，而非框架内部的 bug；(2) 我们提供了一个**量化的、多维度的评分体系**，而非二元的 pass/fail 测试。

### 2.2 DL Benchmarking 与可复现性

Deep500 提出了第一个模块化的 DL Benchmarking 基础设施，将评估分解为算子、网络、训练和分布式训练四个层级。MLPerf 提供以闭式和开放式模型竞赛为形式的工业标准性能 Benchmark。DAWNBench 聚焦于将 time-to-accuracy 作为性能-准确度的综合指标。近期关于跨后端兼容性的工作提出了一个配置优先的框架，采用三层验证（张量级、激活级、任务级）来度量 CPU、GPU 和编译后端之间的行为漂移。我们的工作在此方向上扩展，增加了 Agentic 评估，扩展了指标体系以包含过程导向指标（ME/AR），并构建了带有受控失败模式的迁移专用 Benchmark。

### 2.3 Agentic 软件工程

LLM 驱动的代码生成的最新进展（Codex、GPT-4）激发了对 Agentic 软件工程的兴趣。SWE-bench 以真实世界的 GitHub issue 评估 LLM Agent。在 DL 领域，LLM 已被应用于模型翻译和算子映射。我们的 Agentic 评估环建立在这些思想之上，但聚焦于**度量**而非**改善**迁移自动化——它在受控、可复现的条件下量化迁移代价中有多大比例可以卸载到 LLM Agent。

## 3. 系统架构与评估流水线

图 1 展示了 TorchBridgeBench 的整体架构（当前为 Cross-Backend Compatibility 框架的占位图，需替换为我们的系统架构总览图）。系统由五大组件构成：

**Adapter 层**：每个迁移方法通过 `AdapterSpec` 封装，定义了 preamble（导入、环境设置）、设备映射和算子到 API 的映射表。这一设计使得新迁移方法的集成只需编写单个 JSON 配置文件，无需修改评估核心。

**测试用例生成器**：我们实现了多级测试用例生成策略：(L1) 覆盖 dtype × shape × 参数组合的单算子测试；(L2) 按照 SORT 方法论从 49 个热门 PyTorch 模型中提取频繁子图的算子组合测试；(L3) 涵盖 CNN、视觉 Transformer、序列模型和检测/分割架构的模型级测试；(L4) 覆盖优化器/调度器/AMP 组合的训练环测试。

**三层验证引擎**：受 Cross-Backend 兼容性协议的启发，我们实现：
- **Tier-1（张量级）**：支持可配置 tolerance 扫描（$10^{-6}$ 到 $10^{-3}$）的元素级 allclose(atol, rtol)，外加 MAE 和 P95 误差统计。
- **Tier-2（激活级）**：轻量级前向 hook 捕获逐层输出，反向 hook 捕获梯度张量，实现**首次偏差层**的定位。
- **Tier-3（任务级）**：包括 Top-K 一致性、mAP/mIoU 和训练 loss 曲线 DTW 距离等任务级指标。

**Agentic 评估环**：一个将传统静态 pass/fail 判断替换为动态 LLM 驱动修复循环的新组件。当迁移尝试失败时，错误 traceback 和源代码被发送给 LLM Agent（配置冻结模型快照，temperature=0.0）进行修复。Agent 在沙盒环境中迭代修改代码，直至成功或达到最大轮数限制。我们采用蒙特卡洛采样（N=10 次独立试验）报告期望值和 95% 置信区间，以缓解 LLM 的非确定性。

**报告生成器**：输出结构化 JSON/Markdown 报告、CSV 指标矩阵、LaTeX 就绪的对比表、雷达图和失败分类可视化。

**【图表替换说明】** 系统架构总览图需包含：(1) 左侧：Benchmark 输入层；(2) 中部：四阶段 Pipeline；(3) 右侧：Agentic Loop 分支；(4) 底部：输出产物。建议用 TikZ 或 draw.io 绘制。

## 4. 指标体系

我们的指标体系按三个层级外加两个过程指标组织：

| 层级 | 指标 | 方向 | 描述 |
|------|------|------|------|
| 执行层 | OCR | ↑ | 算子覆盖率：已覆盖算子数 / 总算子数 |
| 执行层 | MER | ↑ | 模型执行率：完成前向推理的模型比例 |
| 数值层 | FNE | ↑ | 前向数值等价性：逐层 cosine 相似度均值 |
| 数值层 | GC | ↑ | 梯度一致性：逐层梯度 cosine 相似度均值 |
| 训练层 | TCA_ℓ | ↓ | 训练收敛对齐：loss 曲线 DTW 距离 |
| 训练层 | TCA_a | ↓ | 最终 accuracy 相对偏差 |
| 过程指标 | ME | ↓ | 迁移代价：α·交互轮数 + β·编辑距离 |
| 过程指标 | AR | ↑ | 自动化率：Agent 自动修复成功的失败案例比例 |

### 4.1 执行层

**算子覆盖率（OCR）**：OCR = |已执行算子集| / |总算子集|。

**模型执行率（MER）**：MER = 完成端到端前向的模型数 / 总模型数。执行失败自动分类为：`OperatorNotFound`、`TypeMismatch`、`DeviceMismatch`、`ShapeMismatch` 和 `RuntimeCrash`。

### 4.2 数值层

**前向数值等价性（FNE）**：在固定随机种子和同步输入下，计算各层展平输出的 cosine 相似度均值。FNE 衡量的是逐层对齐而非仅最终输出，能精确定位数值漂移首次出现的位置。

**梯度一致性（GC）**：对各层反向梯度计算 cosine 相似度。GC 捕获自动微分层面的差异（如 Dropout 方向语义、BatchNorm 图构建差异），这些差异在前向输出中可能不可见，但会导致训练发散。

**【图表替换说明】** 此处需替换为我们的梯度偏差累积示意图：展示算子序列逐层 GC 对比，高亮首次显著偏离的层位置。

### 4.3 训练层

**训练收敛对齐（TCA）**：TCA_ℓ 以 DTW 距离衡量 loss 曲线的整体对齐程度；TCA_a 以最终 accuracy 的相对偏差作为收敛结果汇总指标。

**【图表替换说明】** 此处需替换为我们的 Tolerance 扫描实验图：X 轴为 atol 档位，Y 轴为通过率，多条折线代表不同迁移方法。

### 4.4 Agentic 评估的过程指标

**自动化率（AR）**：AR = Agent 在 T_max 轮内修复成功的失败案例数 / 总失败案例数 × 100%。

**迁移代价（ME）**：ME = α·交互轮数 + β·编辑距离。α、β 为权重系数。

**蒙特卡洛评估**：每个测试用例在 N 次独立试验中评估（默认 N=10）。报告期望值 μ 和 95% 置信区间，若 σ² > ε 则自动增加 N。

我们还借鉴 pass@k 定义了 **migrate@k**：k 次独立 Agent 尝试中至少成功一次的概率。

**【图表替换说明】** Agentic Evaluation Loop 流程图是论文最大创新点之一，需包含完整的闭环流程和 Monte Carlo 聚合步骤。

## 5. Benchmark 构建

### 5.1 多样性维度

**【图表替换说明】** 此处需替换为 Benchmark 设计空间三维示意图（模型 × 组件 × 失败模式）。

**模型结构多样性**：涵盖标准 CNN（ResNet-18/50、VGG-16、MobileNetV2）、注意力架构（ViT-Base、Swin-T）、序列模型（LSTM、GRU）、检测/分割模型（YOLOv5n、UNet、DeepLabV3）和轻量网络（EfficientNet-B0、ShuffleNetV2）。

**训练组件多样性**：系统性地变化优化器（SGD+momentum、Adam、AdamW）× 学习率调度器（StepLR、CosineAnnealing）× 正则化策略（Dropout、weight decay）× 精度模式（FP32、AMP）。

**失败模式多样性**：构造三类标注样本：(1) 已知正确迁移；(2) 算子覆盖失败；(3) 梯度偏差案例。

### 5.2 与现有 Benchmark 的对比

对比表展示了 TorchBridgeBench 与 Deep500、MLPerf、Cross-Backend、SORT、ModelMeta 在算子级/模型级/训练级/数值级/梯度级/Agentic/跨框架七个维度上的功能对比。TorchBridgeBench 是所有系统中唯一在全部七个维度上提供支持的框架。

## 6. 实验设计

### 6.1 研究问题

- **RQ1（横向对比）**：不同迁移范式在完整指标矩阵下的表现如何？
- **RQ2（指标区分度）**：FNE 和 GC 是否能识别仅靠执行指标或最终 accuracy 无法发现的迁移质量差异？
- **RQ3（Agent 有效性）**：多大比例的迁移失败可以被 LLM Agent 自动修复？修复结果在多次试验中的一致性如何？
- **RQ4（流水线可靠性）**：自动评估流水线与人类专家标注的 ground truth 的一致性如何？

### 6.2 对比方法

- **运行时拦截**：torch4ms——基于 `__torch_dispatch__` 的兼容层
- **直接 API 映射**：mindtorch、mindtorch_v2——静态 API 级命名空间映射
- **补丁式重写**：mindnlp_patch——基于 AST 级代码变换和规则重写
- **硬件原生**：torch-npu——带 Ascend NPU 后端的 PyTorch，作为性能上界

### 6.3 实验协议

所有实验在固定随机种子（42）下运行，使用对齐的数据预处理和统一的训练预算（10 epoch，batch size 32，learning rate 0.001）。扫描四个 tolerance 档位（atol ∈ {1e-6, 1e-5, 1e-4, 1e-3}，rtol=1e-5）。Agentic 评估使用 DeepSeek-Chat，temperature=0.0，T_max=5 轮，每个测试用例 N=10 次独立试验。所有实验记录环境指纹以保证可复现性。

## 7. 结果（占位）

**【注意】** 所有结果表格和图表目前均为占位内容，数值为示意性 TBD，需在实际实验完成后替换。

- **表 2**：五种迁移方法 × 七项指标的完整对比矩阵
- **图 6**：Model × Migration Method 通过率热力图
- **表 3**：指标区分度消融实验（Accuracy Gap vs GC Gap）
- **图 7**：失败分类分布图
- **表 4**：Agentic 评估结果（AR / ME / migrate@3）
- **表 5**：流水线可靠性（自动评估 vs 人工标注的 precision/recall）
- **图 8**：案例分析图（逐层 FNE 折线 + loss 曲线对比）

## 8. 讨论

### 8.1 有效性威胁

**内部有效性**：Hook 插桩可能引入轻微运行时扰动，通过使用只读探针和验证扰动幅度低于测量精度来缓解。Agentic 评估引入 LLM 非确定性，通过蒙特卡洛采样、冻结模型快照和 temperature=0.0 来应对。

**外部有效性**：Benchmark 目前聚焦于 PyTorch→MindSpore 迁移。指标体系和流水线设计是框架无关的，但将结论推广到其他框架对需要额外的实证验证。

**构造有效性**：Cosine 相似度和 DTW 是语义等价性的近似，可能无法捕获所有形式的差异。通过报告一套指标而非单一汇总分数来缓解。

### 8.2 经验教训

需在完成实际实验后撰写，总结关键洞察：（1）哪些算子组合最常导致失败？（2）不同迁移方法的失败模式有何系统性差异？（3）LLM Agent 对哪些类型的错误修复最有效/最无效？（4）哪些配置对 benchmark 结论影响最大？

## 9. 结论与未来工作

TorchBridgeBench 将执行可行性、数值等价性和训练收敛对齐统一为一个指标体系，并通过 Agentic 评估对迁移代价和自动化率进行量化，捕获了现有代码指标和测试方法遗漏的迁移语义。配套的 Benchmark、评估流水线和 Agentic 环为大规模、公平、可复现的迁移方案对比提供了基础设施。

未来工作包括：(1) 将 Benchmark 扩展到更多框架对（PyTorch↔JAX、TensorFlow→PyTorch）；(2) 集成更复杂的 Agent 策略（多 Agent 辩论、检索增强修复）；(3) 构建社区贡献的带版本化测试用例的 Benchmark 仓库；(4) 将框架应用于 CI 流水线，对演化中的迁移工具进行回归测试。

---

## 附录：图表制作清单

以下是本论文最终需要自制的全部图表：

1. **系统架构总览图** — 五模块 pipeline + Agentic Loop 分支（TikZ / draw.io）
2. **梯度偏差累积示意图** — 算子序列逐层 GC 对比 + 首次偏差层标注（matplotlib / TikZ）
3. **Tolerance 扫描实验图** — 多方法 atol-pass_rate 折线图（matplotlib / seaborn）
4. **Agentic Evaluation Loop 流程图** — 完整闭环流程（TikZ）
5. **Benchmark 设计空间三维示意图** — 模型 × 组件 × 失败模式（matplotlib 3D / TikZ）
6. **Model × Migration Method 通过率热力图**（seaborn.heatmap）
7. **失败分类分布图** — 多方法分色堆叠图（matplotlib）
8. **案例分析图** — 逐层 FNE 折线 + loss 曲线对比（matplotlib）
9. **雷达图** — 每个迁移方法在 7 个指标上的雷达图（matplotlib radar chart）
10. **Comparison Matrix 表** — 5 方法 × 7 指标的完整对比表
11. **Benchmark 功能对比表** — 与现有 benchmark 的功能矩阵对比
12. **消融实验表** — Accuracy vs GC 区分度对比
13. **Agent 评估结果表** — AR / ME / migrate@k
14. **Pipeline 可靠性表** — 自动 vs 人工 precision/recall

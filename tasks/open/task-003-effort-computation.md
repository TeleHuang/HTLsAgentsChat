# task-003: 补齐 Effort 计算（替代占位字段）

**assigned_to**: local-agent
**priority**: P1
**created**: 2026-06-15
**depends_on**: 无

## 目标

当前 `torchbridgebenchCCplugin/scripts/` 中 effort 仍是占位字段。需要按执行规约实现真实的 ME = Effort_adapt + Effort_repair 计算。

## 参考文档

- `wiki/decisions/execution-spec.md` §3.6（ME 计算公式伪代码）
- `wiki/decisions/metrics-system.md` §1.4.1-1.4.2（ME/AR 定义）

## 具体改动

1. 在 `calc_effort(rounds, diff)` 中实现 `α × rounds + β × edit_distance`
2. α/β 先用临时硬编码值（如 α=1.0, β=0.5），标注为 `# TODO: calibrate via user study`
3. Effort_adapt 和 Effort_repair 分开累计
4. 在报告输出中展示 Effort_adapt / Effort_repair / ME total

## 验收标准

- [ ] `calc_effort` 返回非零实数
- [ ] 核心报告中 Effort_adapt、Effort_repair、ME total 三列有真实数值
- [ ] 代码注释标注 α/β 为临时值，指向用户调研标定任务

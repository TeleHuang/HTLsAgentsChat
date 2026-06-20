# HTLsAgentsChat — Agent 指令

你是 TorchBridgeBench 项目的分布式 Coding Agent 之一。你的总指挥是 Tele Huang。这个仓库是你的共享工作台。

## 身份

你在 README.md 中列出的设备之一上运行（GPU 服务器 / NPU 服务器 / 本地开发机）。你需要在每次启动时声明你所在的设备。

## 每次启动的标准流程

1. `git pull` —— 拉取最新状态
2. 读 `README.md` —— 了解项目背景
3. 读 `tasks/open/` —— 查看有没有分配给你的新任务
4. 读 `tasks/in-progress/` —— 查看有没有你之前未完成的任务
5. 读 `logs/<your-device>-agent.md` —— 回顾你自己上次的日志
6. 在 `logs/<your-device>-agent.md` 顶部追加本次启动记录
7. 如果你有进行中的任务，继续执行；如果有新任务，认领并开始；如果无事可做，向总指挥汇报当前设备状态

## 认领任务

1. 将任务文件从 `tasks/open/` 移动到 `tasks/in-progress/`
2. 在任务文件中记录认领时间和你的设备名
3. 在 `logs/<your-device>-agent.md` 中记录认领动作

## 执行任务

1. 严格按任务规格执行，不确定时在任务文件中追加 `## 疑问` 节，提交后等待总指挥回复
2. 所有产出物放入 `artifacts/` 对应目录
3. 大文件（>10MB）不要提交到 Git，在任务文件中记录服务器上的绝对路径
4. 每完成一个可验证的步骤就 commit + push，不要攒到最后

## 完成任务

1. 将任务文件从 `tasks/in-progress/` 移动到 `tasks/done/`
2. 在任务文件中记录完成时间、产出物列表、关键数字
3. 在 `logs/<your-device>-agent.md` 中记录完成摘要
4. Commit message 格式：`<设备名>: <任务ID> complete —— <关键数字>`

## 需要帮助时

在任务文件中追加 `## 阻塞` 节，描述遇到的问题。Commit + push。总指挥或其他 Agent 会来协助。

## 禁止事项

- 不要修改 `specs/` 目录下的共享规范文件，除非总指挥明确要求
- 不要修改其他 Agent 的日志文件
- 不要在没有总指挥确认的情况下修改核心代码库（`D:\Workspace\Mindspore\torchbridgebench\` 或服务器上的对应路径）
- 不要在 commit message 中使用 `Co-Authored-By` 或其他签名

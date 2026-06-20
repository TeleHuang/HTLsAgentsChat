# 基准测试用例规格

## L1 单算子（≥60 用例）

30 个核心 PyTorch 算子，每个 2 个变体（不同 dtype、形状或参数）。

### 算子分类

| 类别 | 算子 | 数量 | 已知风险 |
|------|------|------|---------|
| 卷积 | Conv1d, Conv2d, Conv3d, ConvTranspose2d, DepthwiseConv2d | 5×2 | |
| 归一化 | BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm, RMSNorm | 6×2 | BatchNorm momentum 语义差异 |
| 激活 | ReLU, GELU, SiLU, LeakyReLU, Sigmoid, Tanh, Softmax, Mish, Hardswish | 9×2 | |
| 线性/注意力 | Linear, matmul, bmm, einsum, scaled_dot_product_attention | 5×2 | |
| 池化 | MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d | 4×2 | |
| 正则化 | Dropout, Dropout2d | 2×2 | keep_prob vs p 语义反转 |
| 张量操作 | cat, reshape, permute, index_select, expand/repeat, scatter, gather | 7×2 | |
| 规约 | sum, mean, argmax, topk, sort | 5×2 | |
| 复数/RoPE | torch.polar, complex mult, view_as_real, RoPE, complex abs/angle | 5×2 | 复数类型支持缺口 |

### 每个用例的必要字段

```yaml
id: bench_v1.0.0/L1/conv2d/001
operator: torch.nn.Conv2d
dtype: float32
input_shape: [1, 3, 224, 224]
kwargs:
  in_channels: 3
  out_channels: 64
  kernel_size: 7
  stride: 2
  padding: 3
seed: 42
expected_output_shape: [1, 64, 112, 112]
known_risk: none  # none | semantic_diff | missing_op | numeric_unstable
```

## L2 频繁子图（≥40 用例）

15 个子图模式，每个 2-3 个变体。

| 类别 | 子图 | 变体数 |
|------|------|--------|
| CNN | Conv-BN-ReLU | 3 |
| CNN | Residual Block (含 bottleneck) | 3 |
| CNN | SE Attention | 2 |
| CNN | MobileNetV2 InvertedResidual | 2 |
| Transformer | Self-Attention (含 sdpa) | 3 |
| Transformer | Transformer Encoder Layer | 3 |
| Transformer | FFN (含 SwiGLU) | 3 |
| Backward | RMSNorm 梯度 | 2 |
| Backward | Gradient Chain (pow→mean→backward) | 2 |
| Backward | Frozen Parameters | 2 |
| Backward | no_grad Context | 2 |
| Backward | Embedding Gradient | 2 |
| Complex/RoPE | Real RoPE | 2 |
| Complex/RoPE | Complex RoPE | 2 |

## L3 完整模型（≥25 用例）

5 种架构 × 5 个配置变体（FP32, FP16/AMP, semantic-diff, ModelMeta SMR trap, width-reduced）。

| # | 模型 | 备注 |
|---|------|------|
| 1 | ResNet-18 | 手写，不依赖 torchvision |
| 2 | ViT-Tiny | 12 层，patch embedding |
| 3 | MobileNetV2 | 倒残差 + 深度可分离卷积 |
| 4 | YOLOv5n Backbone | CSP-Darknet + SiLU + SPP |
| 5 | UNet | 编码器-解码器 + 跳跃连接 |

## L4 训练环（≥40 用例）

| 子层 | 步数 | 用例数 | 检测目标 |
|------|------|--------|---------|
| L4-short | 2 步 | ≥25 | 基本训练环可执行性（5模型×3优化器×2精度） |
| L4-medium | 50 步 | ≥8 | BN running stats / AMP scale / 内存趋势 / 梯度裁剪 |
| L4-long | 完整 epoch | 3 | 优化器状态累积 / 收敛曲线 (ResNet18×30ep, ViT×50ep) |

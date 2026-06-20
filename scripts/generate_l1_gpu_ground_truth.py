#!/usr/bin/env python3
"""Generate PyTorch L1 operator ground truth artifacts for task-001."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


SEED = 42


@dataclass(frozen=True)
class Case:
    case_id: str
    operator: str
    dtype: torch.dtype
    input_shape: list[int] | list[list[int]]
    run: Callable[[torch.device, torch.dtype], dict[str, Any]]
    known_risk: str = "none"
    requires_grad: bool = True


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def rel(path: Path) -> str:
    return path.as_posix()


def randn(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype, requires_grad: bool = True) -> torch.Tensor:
    t = torch.randn(shape, device=device, dtype=dtype)
    if requires_grad and dtype.is_floating_point:
        t.requires_grad_(True)
    return t


def randint(shape: tuple[int, ...], high: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, high, shape, device=device, dtype=torch.long)


def finish_tensor(name: str, tensor: torch.Tensor, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "tensor": tensor.detach().cpu(),
        "shape": list(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        **(extras or {}),
    }


def backward(output: torch.Tensor) -> None:
    if output.dtype.is_floating_point or output.dtype.is_complex:
        target = output.real.sum() if output.dtype.is_complex else output.sum()
        target.backward()


def tensor_grads(named_tensors: list[tuple[str, torch.Tensor]]) -> list[dict[str, Any]]:
    grads: list[dict[str, Any]] = []
    for name, tensor in named_tensors:
        grad = tensor.grad
        if grad is not None:
            grads.append(finish_tensor(name, grad))
    return grads


def module_param_grads(module: torch.nn.Module) -> list[dict[str, Any]]:
    grads: list[dict[str, Any]] = []
    for name, param in module.named_parameters():
        if param.grad is not None:
            grads.append(finish_tensor(name, param.grad))
    return grads


def module_case(
    case_id: str,
    operator: str,
    dtype: torch.dtype,
    input_shape: tuple[int, ...],
    factory: Callable[[], torch.nn.Module],
    known_risk: str = "none",
) -> Case:
    def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
        set_seed()
        module = factory().to(device=device, dtype=run_dtype)
        module.train()
        x = randn(input_shape, device, run_dtype)
        output = module(x)
        backward(output)
        return {
            "output": finish_tensor("output", output),
            "activations": [finish_tensor("module_output", output)],
            "gradients": tensor_grads([("input", x)]) + module_param_grads(module),
        }

    return Case(case_id, operator, dtype, list(input_shape), run, known_risk)


def unary_case(case_id: str, operator: str, dtype: torch.dtype, shape: tuple[int, ...], fn: Callable[[torch.Tensor], torch.Tensor]) -> Case:
    def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
        set_seed()
        x = randn(shape, device, run_dtype)
        output = fn(x)
        backward(output)
        return {
            "output": finish_tensor("output", output),
            "activations": [finish_tensor("operator_output", output)],
            "gradients": tensor_grads([("input", x)]),
        }

    return Case(case_id, operator, dtype, list(shape), run)


def no_grad_case(
    case_id: str,
    operator: str,
    dtype: torch.dtype,
    input_shape: list[int] | list[list[int]],
    fn: Callable[[torch.device, torch.dtype], torch.Tensor],
    known_risk: str = "none",
) -> Case:
    def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
        set_seed()
        output = fn(device, run_dtype)
        return {
            "output": finish_tensor("output", output),
            "activations": [finish_tensor("operator_output", output)],
            "gradients": [],
        }

    return Case(case_id, operator, dtype, input_shape, run, known_risk, requires_grad=False)


def build_cases() -> list[Case]:
    cases: list[Case] = []
    add = cases.append

    add(module_case("bench_v1.0.0/L1/conv1d/001", "torch.nn.Conv1d", torch.float32, (2, 3, 64), lambda: torch.nn.Conv1d(3, 8, 3, padding=1)))
    add(module_case("bench_v1.0.0/L1/conv1d/002", "torch.nn.Conv1d", torch.float64, (1, 4, 31), lambda: torch.nn.Conv1d(4, 6, 5, stride=2, padding=2)))
    add(module_case("bench_v1.0.0/L1/conv2d/001", "torch.nn.Conv2d", torch.float32, (1, 3, 64, 64), lambda: torch.nn.Conv2d(3, 16, 7, stride=2, padding=3)))
    add(module_case("bench_v1.0.0/L1/conv2d/002", "torch.nn.Conv2d", torch.float64, (2, 4, 32, 32), lambda: torch.nn.Conv2d(4, 8, 3, padding=1, dilation=1)))
    add(module_case("bench_v1.0.0/L1/conv3d/001", "torch.nn.Conv3d", torch.float32, (1, 2, 8, 16, 16), lambda: torch.nn.Conv3d(2, 5, 3, padding=1)))
    add(module_case("bench_v1.0.0/L1/conv3d/002", "torch.nn.Conv3d", torch.float64, (1, 3, 6, 10, 10), lambda: torch.nn.Conv3d(3, 4, (3, 3, 1), padding=(1, 1, 0))))
    add(module_case("bench_v1.0.0/L1/conv_transpose2d/001", "torch.nn.ConvTranspose2d", torch.float32, (1, 4, 16, 16), lambda: torch.nn.ConvTranspose2d(4, 6, 4, stride=2, padding=1)))
    add(module_case("bench_v1.0.0/L1/conv_transpose2d/002", "torch.nn.ConvTranspose2d", torch.float64, (2, 3, 12, 12), lambda: torch.nn.ConvTranspose2d(3, 5, 3, padding=1)))
    add(module_case("bench_v1.0.0/L1/depthwise_conv2d/001", "torch.nn.Conv2d(groups=in_channels)", torch.float32, (1, 8, 32, 32), lambda: torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)))
    add(module_case("bench_v1.0.0/L1/depthwise_conv2d/002", "torch.nn.Conv2d(groups=in_channels)", torch.float64, (1, 6, 24, 24), lambda: torch.nn.Conv2d(6, 12, 3, padding=1, groups=6)))

    add(module_case("bench_v1.0.0/L1/batchnorm1d/001", "torch.nn.BatchNorm1d", torch.float32, (4, 8, 16), lambda: torch.nn.BatchNorm1d(8), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/batchnorm1d/002", "torch.nn.BatchNorm1d", torch.float64, (6, 10), lambda: torch.nn.BatchNorm1d(10), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/batchnorm2d/001", "torch.nn.BatchNorm2d", torch.float32, (2, 8, 16, 16), lambda: torch.nn.BatchNorm2d(8), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/batchnorm2d/002", "torch.nn.BatchNorm2d", torch.float64, (2, 5, 12, 12), lambda: torch.nn.BatchNorm2d(5), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/layernorm/001", "torch.nn.LayerNorm", torch.float32, (3, 12, 16), lambda: torch.nn.LayerNorm(16)))
    add(module_case("bench_v1.0.0/L1/layernorm/002", "torch.nn.LayerNorm", torch.float64, (2, 4, 8, 8), lambda: torch.nn.LayerNorm((8, 8))))
    add(module_case("bench_v1.0.0/L1/groupnorm/001", "torch.nn.GroupNorm", torch.float32, (2, 8, 16, 16), lambda: torch.nn.GroupNorm(4, 8)))
    add(module_case("bench_v1.0.0/L1/groupnorm/002", "torch.nn.GroupNorm", torch.float64, (2, 12, 8, 8), lambda: torch.nn.GroupNorm(3, 12)))
    add(module_case("bench_v1.0.0/L1/instancenorm/001", "torch.nn.InstanceNorm2d", torch.float32, (2, 6, 16, 16), lambda: torch.nn.InstanceNorm2d(6, affine=True)))
    add(module_case("bench_v1.0.0/L1/instancenorm/002", "torch.nn.InstanceNorm1d", torch.float64, (2, 5, 12), lambda: torch.nn.InstanceNorm1d(5, affine=True)))
    add(module_case("bench_v1.0.0/L1/rmsnorm/001", "torch.nn.RMSNorm", torch.float32, (2, 8, 16), lambda: torch.nn.RMSNorm(16)))
    add(module_case("bench_v1.0.0/L1/rmsnorm/002", "torch.nn.RMSNorm", torch.float64, (2, 6, 10), lambda: torch.nn.RMSNorm(10)))

    activations = [
        ("relu", "torch.nn.functional.relu", F.relu),
        ("gelu", "torch.nn.functional.gelu", F.gelu),
        ("silu", "torch.nn.functional.silu", F.silu),
        ("leaky_relu", "torch.nn.functional.leaky_relu", lambda x: F.leaky_relu(x, 0.2)),
        ("sigmoid", "torch.sigmoid", torch.sigmoid),
        ("tanh", "torch.tanh", torch.tanh),
        ("softmax", "torch.nn.functional.softmax", lambda x: F.softmax(x, dim=-1)),
        ("mish", "torch.nn.functional.mish", F.mish),
        ("hardswish", "torch.nn.functional.hardswish", F.hardswish),
    ]
    for name, op, fn in activations:
        add(unary_case(f"bench_v1.0.0/L1/{name}/001", op, torch.float32, (4, 16), fn))
        add(unary_case(f"bench_v1.0.0/L1/{name}/002", op, torch.float64, (2, 3, 8), fn))

    add(module_case("bench_v1.0.0/L1/linear/001", "torch.nn.Linear", torch.float32, (4, 16), lambda: torch.nn.Linear(16, 32)))
    add(module_case("bench_v1.0.0/L1/linear/002", "torch.nn.Linear", torch.float64, (2, 5, 12), lambda: torch.nn.Linear(12, 7)))

    def matmul_run(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> Callable[[torch.device, torch.dtype], dict[str, Any]]:
        def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
            set_seed()
            a = randn(shape_a, device, run_dtype)
            b = randn(shape_b, device, run_dtype)
            output = torch.matmul(a, b)
            backward(output)
            return {"output": finish_tensor("output", output), "activations": [finish_tensor("matmul_output", output)], "gradients": tensor_grads([("a", a), ("b", b)])}

        return run

    add(Case("bench_v1.0.0/L1/matmul/001", "torch.matmul", torch.float32, [[8, 16], [16, 4]], matmul_run((8, 16), (16, 4))))
    add(Case("bench_v1.0.0/L1/matmul/002", "torch.matmul", torch.float64, [[2, 8, 12], [2, 12, 5]], matmul_run((2, 8, 12), (2, 12, 5))))
    add(Case("bench_v1.0.0/L1/bmm/001", "torch.bmm", torch.float32, [[3, 8, 16], [3, 16, 4]], matmul_run((3, 8, 16), (3, 16, 4))))
    add(Case("bench_v1.0.0/L1/bmm/002", "torch.bmm", torch.float64, [[2, 6, 10], [2, 10, 7]], matmul_run((2, 6, 10), (2, 10, 7))))

    def einsum_run(pattern: str, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> Callable[[torch.device, torch.dtype], dict[str, Any]]:
        def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
            set_seed()
            a = randn(shape_a, device, run_dtype)
            b = randn(shape_b, device, run_dtype)
            output = torch.einsum(pattern, a, b)
            backward(output)
            return {"output": finish_tensor("output", output), "activations": [finish_tensor("einsum_output", output)], "gradients": tensor_grads([("a", a), ("b", b)])}

        return run

    add(Case("bench_v1.0.0/L1/einsum/001", "torch.einsum", torch.float32, [[2, 3, 4], [2, 4, 5]], einsum_run("bij,bjk->bik", (2, 3, 4), (2, 4, 5))))
    add(Case("bench_v1.0.0/L1/einsum/002", "torch.einsum", torch.float64, [[3, 4], [4, 5]], einsum_run("ij,jk->ik", (3, 4), (4, 5))))

    def sdpa_run(shape: tuple[int, int, int, int]) -> Callable[[torch.device, torch.dtype], dict[str, Any]]:
        def run(device: torch.device, run_dtype: torch.dtype) -> dict[str, Any]:
            set_seed()
            q = randn(shape, device, run_dtype)
            k = randn(shape, device, run_dtype)
            v = randn(shape, device, run_dtype)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            backward(output)
            return {"output": finish_tensor("output", output), "activations": [finish_tensor("sdpa_output", output)], "gradients": tensor_grads([("q", q), ("k", k), ("v", v)])}

        return run

    add(Case("bench_v1.0.0/L1/scaled_dot_product_attention/001", "torch.nn.functional.scaled_dot_product_attention", torch.float32, [2, 2, 8, 16], sdpa_run((2, 2, 8, 16))))
    add(Case("bench_v1.0.0/L1/scaled_dot_product_attention/002", "torch.nn.functional.scaled_dot_product_attention", torch.float64, [1, 4, 6, 8], sdpa_run((1, 4, 6, 8))))

    add(module_case("bench_v1.0.0/L1/maxpool2d/001", "torch.nn.MaxPool2d", torch.float32, (2, 3, 32, 32), lambda: torch.nn.MaxPool2d(2)))
    add(module_case("bench_v1.0.0/L1/maxpool2d/002", "torch.nn.MaxPool2d", torch.float64, (1, 4, 17, 17), lambda: torch.nn.MaxPool2d(3, stride=2, padding=1)))
    add(module_case("bench_v1.0.0/L1/avgpool2d/001", "torch.nn.AvgPool2d", torch.float32, (2, 3, 32, 32), lambda: torch.nn.AvgPool2d(2)))
    add(module_case("bench_v1.0.0/L1/avgpool2d/002", "torch.nn.AvgPool2d", torch.float64, (1, 4, 17, 17), lambda: torch.nn.AvgPool2d(3, stride=2, padding=1)))
    add(module_case("bench_v1.0.0/L1/adaptive_avgpool2d/001", "torch.nn.AdaptiveAvgPool2d", torch.float32, (2, 3, 31, 29), lambda: torch.nn.AdaptiveAvgPool2d((7, 7))))
    add(module_case("bench_v1.0.0/L1/adaptive_avgpool2d/002", "torch.nn.AdaptiveAvgPool2d", torch.float64, (1, 4, 13, 15), lambda: torch.nn.AdaptiveAvgPool2d((1, 1))))
    add(module_case("bench_v1.0.0/L1/adaptive_maxpool2d/001", "torch.nn.AdaptiveMaxPool2d", torch.float32, (2, 3, 31, 29), lambda: torch.nn.AdaptiveMaxPool2d((7, 7))))
    add(module_case("bench_v1.0.0/L1/adaptive_maxpool2d/002", "torch.nn.AdaptiveMaxPool2d", torch.float64, (1, 4, 13, 15), lambda: torch.nn.AdaptiveMaxPool2d((1, 1))))

    add(module_case("bench_v1.0.0/L1/dropout/001", "torch.nn.Dropout", torch.float32, (4, 16), lambda: torch.nn.Dropout(p=0.25), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/dropout/002", "torch.nn.Dropout", torch.float64, (2, 3, 8), lambda: torch.nn.Dropout(p=0.5), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/dropout2d/001", "torch.nn.Dropout2d", torch.float32, (2, 4, 8, 8), lambda: torch.nn.Dropout2d(p=0.25), "semantic_diff"))
    add(module_case("bench_v1.0.0/L1/dropout2d/002", "torch.nn.Dropout2d", torch.float64, (1, 3, 6, 6), lambda: torch.nn.Dropout2d(p=0.5), "semantic_diff"))

    add(no_grad_case("bench_v1.0.0/L1/cat/001", "torch.cat", torch.float32, [[2, 3], [2, 5]], lambda d, t: torch.cat([torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 5), device=d, dtype=t)], dim=1)))
    add(no_grad_case("bench_v1.0.0/L1/cat/002", "torch.cat", torch.float64, [[1, 2, 4], [3, 2, 4]], lambda d, t: torch.cat([torch.randn((1, 2, 4), device=d, dtype=t), torch.randn((3, 2, 4), device=d, dtype=t)], dim=0)))
    add(unary_case("bench_v1.0.0/L1/reshape/001", "torch.reshape", torch.float32, (2, 3, 4), lambda x: torch.reshape(x, (6, 4))))
    add(unary_case("bench_v1.0.0/L1/reshape/002", "torch.reshape", torch.float64, (2, 3, 4), lambda x: torch.reshape(x, (2, 12))))
    add(unary_case("bench_v1.0.0/L1/permute/001", "torch.permute", torch.float32, (2, 3, 4), lambda x: torch.permute(x, (0, 2, 1))))
    add(unary_case("bench_v1.0.0/L1/permute/002", "torch.permute", torch.float64, (2, 3, 4, 5), lambda x: torch.permute(x, (0, 2, 3, 1))))
    add(no_grad_case("bench_v1.0.0/L1/index_select/001", "torch.index_select", torch.float32, [5, 7], lambda d, t: torch.index_select(torch.randn((5, 7), device=d, dtype=t), 0, torch.tensor([0, 2, 4], device=d))))
    add(no_grad_case("bench_v1.0.0/L1/index_select/002", "torch.index_select", torch.float64, [4, 6, 8], lambda d, t: torch.index_select(torch.randn((4, 6, 8), device=d, dtype=t), 1, torch.tensor([1, 3, 5], device=d))))
    add(unary_case("bench_v1.0.0/L1/expand_repeat/001", "torch.Tensor.expand", torch.float32, (1, 3, 1), lambda x: x.expand(4, 3, 5)))
    add(unary_case("bench_v1.0.0/L1/expand_repeat/002", "torch.Tensor.repeat", torch.float64, (2, 1, 3), lambda x: x.repeat(1, 4, 2)))
    add(no_grad_case("bench_v1.0.0/L1/scatter/001", "torch.Tensor.scatter", torch.float32, [3, 5], lambda d, t: torch.zeros((3, 5), device=d, dtype=t).scatter(1, torch.tensor([[0, 2], [1, 3], [2, 4]], device=d), torch.randn((3, 2), device=d, dtype=t))))
    add(no_grad_case("bench_v1.0.0/L1/scatter/002", "torch.Tensor.scatter_add", torch.float64, [2, 4], lambda d, t: torch.zeros((2, 4), device=d, dtype=t).scatter_add(1, torch.tensor([[0, 1, 1], [2, 3, 0]], device=d), torch.randn((2, 3), device=d, dtype=t))))
    add(no_grad_case("bench_v1.0.0/L1/gather/001", "torch.gather", torch.float32, [3, 5], lambda d, t: torch.gather(torch.randn((3, 5), device=d, dtype=t), 1, torch.tensor([[0, 2, 4], [1, 3, 0], [2, 4, 1]], device=d))))
    add(no_grad_case("bench_v1.0.0/L1/gather/002", "torch.gather", torch.float64, [2, 4, 6], lambda d, t: torch.gather(torch.randn((2, 4, 6), device=d, dtype=t), 2, torch.tensor([[[0, 1], [2, 3], [4, 5], [1, 0]], [[5, 4], [3, 2], [1, 0], [2, 4]]], device=d))))

    add(unary_case("bench_v1.0.0/L1/sum/001", "torch.sum", torch.float32, (3, 4, 5), lambda x: x.sum(dim=1)))
    add(unary_case("bench_v1.0.0/L1/sum/002", "torch.sum", torch.float64, (2, 3, 4), lambda x: x.sum(dim=(1, 2), keepdim=True)))
    add(unary_case("bench_v1.0.0/L1/mean/001", "torch.mean", torch.float32, (3, 4, 5), lambda x: x.mean(dim=1)))
    add(unary_case("bench_v1.0.0/L1/mean/002", "torch.mean", torch.float64, (2, 3, 4), lambda x: x.mean(dim=(1, 2), keepdim=True)))
    add(no_grad_case("bench_v1.0.0/L1/argmax/001", "torch.argmax", torch.float32, [3, 4, 5], lambda d, t: torch.argmax(torch.randn((3, 4, 5), device=d, dtype=t), dim=1)))
    add(no_grad_case("bench_v1.0.0/L1/argmax/002", "torch.argmax", torch.float64, [2, 3, 4], lambda d, t: torch.argmax(torch.randn((2, 3, 4), device=d, dtype=t), dim=-1, keepdim=True)))
    add(no_grad_case("bench_v1.0.0/L1/topk/001", "torch.topk", torch.float32, [3, 8], lambda d, t: torch.topk(torch.randn((3, 8), device=d, dtype=t), k=3, dim=1).values))
    add(no_grad_case("bench_v1.0.0/L1/topk/002", "torch.topk", torch.float64, [2, 4, 6], lambda d, t: torch.topk(torch.randn((2, 4, 6), device=d, dtype=t), k=2, dim=-1).values))
    add(no_grad_case("bench_v1.0.0/L1/sort/001", "torch.sort", torch.float32, [3, 8], lambda d, t: torch.sort(torch.randn((3, 8), device=d, dtype=t), dim=1).values))
    add(no_grad_case("bench_v1.0.0/L1/sort/002", "torch.sort", torch.float64, [2, 4, 6], lambda d, t: torch.sort(torch.randn((2, 4, 6), device=d, dtype=t), dim=-1, descending=True).values))

    add(no_grad_case("bench_v1.0.0/L1/polar/001", "torch.polar", torch.float32, [[4, 5], [4, 5]], lambda d, t: torch.polar(torch.rand((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t)), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/polar/002", "torch.polar", torch.float64, [[2, 3], [2, 3]], lambda d, t: torch.polar(torch.rand((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t)), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/complex_mult/001", "complex multiplication", torch.float32, [[4, 5], [4, 5]], lambda d, t: torch.complex(torch.randn((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t)) * torch.complex(torch.randn((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t)), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/complex_mult/002", "complex multiplication", torch.float64, [[2, 3], [2, 3]], lambda d, t: torch.complex(torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t)) * torch.complex(torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t)), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/view_as_real/001", "torch.view_as_real", torch.float32, [4, 5], lambda d, t: torch.view_as_real(torch.complex(torch.randn((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t))), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/view_as_real/002", "torch.view_as_real", torch.float64, [2, 3], lambda d, t: torch.view_as_real(torch.complex(torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t))), "missing_op"))

    def rope(device: torch.device, run_dtype: torch.dtype, shape: tuple[int, int, int]) -> torch.Tensor:
        x = torch.randn(shape, device=device, dtype=run_dtype)
        seq, dim = shape[-2], shape[-1]
        freqs = torch.arange(0, dim, 2, device=device, dtype=run_dtype) / dim
        positions = torch.arange(seq, device=device, dtype=run_dtype)
        angles = torch.outer(positions, 1.0 / (10000 ** freqs))
        cos = angles.cos().unsqueeze(0)
        sin = angles.sin().unsqueeze(0)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    add(no_grad_case("bench_v1.0.0/L1/rope/001", "RoPE", torch.float32, [2, 8, 16], lambda d, t: rope(d, t, (2, 8, 16)), "numeric_unstable"))
    add(no_grad_case("bench_v1.0.0/L1/rope/002", "RoPE", torch.float64, [1, 6, 12], lambda d, t: rope(d, t, (1, 6, 12)), "numeric_unstable"))
    add(no_grad_case("bench_v1.0.0/L1/complex_abs_angle/001", "torch.abs/torch.angle", torch.float32, [4, 5], lambda d, t: torch.stack((torch.abs(torch.complex(torch.randn((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t))), torch.angle(torch.complex(torch.randn((4, 5), device=d, dtype=t), torch.randn((4, 5), device=d, dtype=t)))), dim=-1), "missing_op"))
    add(no_grad_case("bench_v1.0.0/L1/complex_abs_angle/002", "torch.abs/torch.angle", torch.float64, [2, 3], lambda d, t: torch.stack((torch.abs(torch.complex(torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t))), torch.angle(torch.complex(torch.randn((2, 3), device=d, dtype=t), torch.randn((2, 3), device=d, dtype=t)))), dim=-1), "missing_op"))

    return cases


def environment(timestamp: str, bridge_commit: str | None, bridge_url: str | None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "driver": _nvidia_driver(),
        "platform": platform.platform(),
        "timestamp": timestamp,
        "target_bridge": {
            "name": "ascend-torch4ms",
            "branch": "main",
            "commit": bridge_commit,
            "url": bridge_url,
        },
    }


def _nvidia_driver() -> str | None:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.splitlines()[0].strip()
    except Exception:
        return None


def safe_name(case_id: str) -> str:
    _, level, op, idx = case_id.split("/")
    return f"{op}_{idx}"


def save_case(case: Case, result: dict[str, Any], out_dir: Path, env: dict[str, Any]) -> dict[str, Any]:
    name = safe_name(case.case_id)
    output_path = out_dir / f"{name}_output.pt"
    activation_path = out_dir / f"{name}_intermediate.pt"
    gradients_path = out_dir / f"{name}_gradients.pt"

    torch.save(result["output"]["tensor"], output_path)
    torch.save({item["name"]: item["tensor"] for item in result["activations"]}, activation_path)
    torch.save({item["name"]: item["tensor"] for item in result["gradients"]}, gradients_path)

    meta = {
        "test_case_id": case.case_id,
        "level": "L1",
        "operator": case.operator,
        "input_shape": case.input_shape,
        "dtype": dtype_name(case.dtype),
        "seed": SEED,
        "known_risk": case.known_risk,
        "output": {
            "tensor_shape": result["output"]["shape"],
            "dtype": result["output"]["dtype"],
            "tensor_binary_path": rel(output_path),
        },
        "intermediate": {
            "binary_path": rel(activation_path),
            "activations": [
                {"layer": item["name"], "shape": item["shape"], "dtype": item["dtype"], "binary_path": rel(activation_path)}
                for item in result["activations"]
            ],
        },
        "gradients": {
            "binary_path": rel(gradients_path),
            "layers": [
                {"name": item["name"], "shape": item["shape"], "dtype": item["dtype"], "binary_path": rel(gradients_path)}
                for item in result["gradients"]
            ],
        },
        "environment": env,
    }
    meta_path = out_dir / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return {
        "test_case_id": case.case_id,
        "operator": case.operator,
        "status": "passed",
        "json_path": rel(meta_path),
        "output_path": rel(output_path),
        "output_shape": result["output"]["shape"],
        "gradient_count": len(result["gradients"]),
        "known_risk": case.known_risk,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/gpu-ground-truth/L1")
    parser.add_argument("--bridge-url", default=None)
    parser.add_argument("--bridge-commit", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for GPU ground truth generation")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    env = environment(timestamp, args.bridge_commit, args.bridge_url)

    device = torch.device("cuda")
    cases = build_cases()
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for case in cases:
        try:
            result = case.run(device, case.dtype)
            torch.cuda.synchronize()
            passed.append(save_case(case, result, out_dir, env))
        except Exception as exc:
            failed.append(
                {
                    "test_case_id": case.case_id,
                    "operator": case.operator,
                    "dtype": dtype_name(case.dtype),
                    "known_risk": case.known_risk,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary = {
        "suite": "L1",
        "seed": SEED,
        "total_cases": len(cases),
        "passed": len(passed),
        "failed": len(failed),
        "success_rate": len(passed) / len(cases) if cases else math.nan,
        "environment": env,
        "passed_cases": passed,
        "failed_cases": failed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ["suite", "total_cases", "passed", "failed", "success_rate"]}, indent=2))
    if failed:
        print(json.dumps(failed, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

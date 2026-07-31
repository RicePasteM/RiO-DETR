#!/usr/bin/env python3
"""Reproducible TensorRT 10 latency benchmark with GPU-resident I/O.

The reported latency covers only TensorRT execution. Inputs and outputs are
allocated once on the GPU; preprocessing, host/device copies, NMS, and other
host-side postprocessing are excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import tensorrt as trt
import torch


TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}
if hasattr(trt, "bfloat16"):
    TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more TensorRT engines with CUDA events."
    )
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="May be repeated; for example RiO-N=/tmp/rio_n.engine",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--require-fp16-io",
        action="store_true",
        help="Fail if any floating-point input or output is not FP16.",
    )
    parser.add_argument(
        "--include-hostname",
        action="store_true",
        help="Include the machine hostname in the JSON report.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def nvidia_smi_device() -> str:
    """Return the physical GPU identifier selected by CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        current = torch.cuda.current_device()
        if current < len(devices):
            return devices[current]
    return str(torch.cuda.current_device())


def gpu_snapshot() -> dict[str, str]:
    fields = (
        "name,driver_version,pstate,clocks.current.sm,clocks.current.memory,"
        "temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu"
    )
    command = [
        "nvidia-smi",
        f"--id={nvidia_smi_device()}",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(command, text=True).strip().split(", ")
    keys = fields.split(",")
    return dict(zip(keys, values))


def engine_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--engine must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError(f"--engine label must not be empty: {value!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append((label, path))
    return specs


def shape_tuple(dims: trt.Dims) -> tuple[int, ...]:
    return tuple(int(dim) for dim in dims)


def validate_fp16_io(engine: trt.ICudaEngine) -> None:
    floating_types = {trt.float16, trt.float32}
    if hasattr(trt, "bfloat16"):
        floating_types.add(trt.bfloat16)
    invalid = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        dtype = engine.get_tensor_dtype(name)
        if dtype in floating_types and dtype != trt.float16:
            invalid.append(f"{name}={dtype}")
    if invalid:
        raise RuntimeError(
            "Expected FP16 floating-point I/O, but found " + ", ".join(invalid)
        )


def allocate_io(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, object]]]:
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    metadata: list[dict[str, object]] = []

    # Input shapes must be fixed before output shapes can be queried.
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        dtype = engine.get_tensor_dtype(name)
        if dtype not in TRT_TO_TORCH:
            raise TypeError(f"Unsupported TensorRT dtype for {name}: {dtype}")
        shape = shape_tuple(engine.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            _, opt_shape, _ = engine.get_tensor_profile_shape(name, 0)
            shape = shape_tuple(opt_shape)
            if not context.set_input_shape(name, shape):
                raise RuntimeError(f"Could not set input shape for {name}: {shape}")
        tensor = torch.empty(shape, dtype=TRT_TO_TORCH[dtype], device="cuda")
        context.set_tensor_address(name, tensor.data_ptr())
        inputs.append(tensor)
        metadata.append(
            {
                "name": name,
                "mode": "input",
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )

    insufficient = context.infer_shapes()
    if insufficient:
        raise RuntimeError(
            f"TensorRT could not infer shapes; insufficient tensors: {insufficient}"
        )

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        dtype = engine.get_tensor_dtype(name)
        if dtype not in TRT_TO_TORCH:
            raise TypeError(f"Unsupported TensorRT dtype for {name}: {dtype}")
        shape = shape_tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved output shape for {name}: {shape}")
        tensor = torch.empty(shape, dtype=TRT_TO_TORCH[dtype], device="cuda")
        context.set_tensor_address(name, tensor.data_ptr())
        outputs.append(tensor)
        metadata.append(
            {
                "name": name,
                "mode": "output",
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )

    return inputs, outputs, metadata


def benchmark_one(
    label: str,
    path: Path,
    warmup: int,
    runs: int,
    trials: int,
    use_cuda_graph: bool,
    require_fp16_io: bool,
) -> dict[str, object]:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Could not create execution context for {path}")
    if require_fp16_io:
        validate_fp16_io(engine)

    stream = torch.cuda.Stream()
    inputs, outputs, io_metadata = allocate_io(engine, context)

    def enqueue() -> None:
        if not context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError(f"TensorRT enqueue failed for {path}")

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            enqueue()
    stream.synchronize()

    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            enqueue()
        stream.synchronize()

    latencies_ms: list[float] = []
    snapshots: list[dict[str, str]] = []
    for _ in range(trials):
        snapshots.append(gpu_snapshot())
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(runs):
                if graph is None:
                    enqueue()
                else:
                    graph.replay()
            end.record(stream)
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end) / runs)
        snapshots.append(gpu_snapshot())

    mean_ms = statistics.fmean(latencies_ms)
    result = {
        "label": label,
        "engine": str(path),
        "engine_bytes": path.stat().st_size,
        "io": io_metadata,
        "warmup": warmup,
        "runs_per_trial": runs,
        "trials": trials,
        "cuda_graph": use_cuda_graph,
        "required_fp16_io": require_fp16_io,
        "trial_latency_ms": latencies_ms,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": statistics.median(latencies_ms),
        "stdev_latency_ms": (
            statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        ),
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "fps_from_mean": 1000.0 / mean_ms,
        "gpu_snapshots": snapshots,
    }
    del inputs, outputs, context, engine, runtime
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    specs = engine_specs(args.engine)
    report = {
        "created_at_unix": time.time(),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "gpu_initial": gpu_snapshot(),
        "results": [],
    }
    if args.include_hostname:
        report["hostname"] = os.uname().nodename

    for label, path in specs:
        result = benchmark_one(
            label,
            path,
            args.warmup,
            args.runs,
            args.trials,
            args.cuda_graph,
            args.require_fp16_io,
        )
        report["results"].append(result)
        print(
            f"{label:16s} mean={result['mean_latency_ms']:.4f} ms "
            f"median={result['median_latency_ms']:.4f} ms "
            f"range=[{result['min_latency_ms']:.4f}, "
            f"{result['max_latency_ms']:.4f}] ms "
            f"FPS={result['fps_from_mean']:.1f}",
            flush=True,
        )

    report["gpu_final"] = gpu_snapshot()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

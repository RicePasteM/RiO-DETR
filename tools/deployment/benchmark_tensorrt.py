
import argparse
import os
import torch
import tensorrt as trt
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark RT-DETRv4 TensorRT model with raw TensorRT')
    parser.add_argument(
        '--engine',
        required=True,
        help='Path to TensorRT engine file (e.g., model.engine)')
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Inference device')
    parser.add_argument(
        '--warmup',
        type=int,
        default=50,
        help='Number of warmup runs')
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1000,
        help='Number of benchmark runs')
    parser.add_argument(
        '--use-cuda-graph',
        action='store_true',
        help='Use CUDA Graph for inference optimization')
    return parser.parse_args()

def main():
    args = parse_args()

    engine_file = args.engine
    if not os.path.exists(engine_file):
        print(f"Error: TensorRT engine not found at {engine_file}")
        return
    if args.device != 'cuda':
        print("Error: raw TensorRT benchmark only supports cuda device")
        return

    # Load TensorRT plugins
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')

    print(f"Loading engine from {engine_file}...")
    with open(engine_file, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        print("Error: Failed to deserialize TensorRT engine")
        return

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    inputs = []
    outputs = []
    bindings = []

    # Check TensorRT version capability
    is_trt_10 = hasattr(engine, 'get_tensor_mode')

    print("Allocating memory...")
    if is_trt_10:
        num_io = engine.num_io_tensors
        for i in range(num_io):
            tensor_name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tensor_name)
            dtype = engine.get_tensor_dtype(tensor_name)

            # Determine shape
            if mode == trt.TensorIOMode.INPUT:
                # Use optimization profile 0, index 1 (OPT)
                profile = engine.get_tensor_profile_shape(tensor_name, 0)
                shape = tuple(profile[1])
                print(f"Input {tensor_name}: using opt shape {shape}")
            else:
                # Output shape might depend on input, need to set input shape first to query output shape
                # However, for static output or independent output, we can try context.get_tensor_shape
                # If dynamic, we might need to run inference once to get shape or use max possible
                # For simplicity, we set input shapes first then query
                pass

            torch_dtype = torch.float32
            if dtype == trt.float16:
                torch_dtype = torch.float16
            elif dtype == trt.int32:
                torch_dtype = torch.int32
            elif dtype == trt.int8:
                torch_dtype = torch.int8
            elif dtype == trt.bool:
                torch_dtype = torch.bool

            if mode == trt.TensorIOMode.INPUT:
                tensor = torch.randn(shape, dtype=torch_dtype, device='cuda')
                context.set_input_shape(tensor_name, tuple(tensor.shape))
                context.set_tensor_address(tensor_name, int(tensor.data_ptr()))
                inputs.append(tensor)
            else:
                # After setting input shape, we can get output shape
                shape = tuple(context.get_tensor_shape(tensor_name))
                print(f"Output {tensor_name}: shape {shape}")
                tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
                context.set_tensor_address(tensor_name, int(tensor.data_ptr()))
                outputs.append(tensor)
    else:
        num_bindings = engine.num_bindings
        bindings = [None] * num_bindings
        for i in range(num_bindings):
            tensor_name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)

            if is_input:
                profile = engine.get_profile_shape(0, i)
                shape = tuple(profile[1])
                print(f"Input {tensor_name}: using opt shape {shape}")
            else:
                pass # Set later

            torch_dtype = torch.float32
            if dtype == trt.float16:
                torch_dtype = torch.float16
            elif dtype == trt.int32:
                torch_dtype = torch.int32
            elif dtype == trt.int8:
                torch_dtype = torch.int8
            elif dtype == trt.bool:
                torch_dtype = torch.bool

            if is_input:
                tensor = torch.randn(shape, dtype=torch_dtype, device='cuda')
                context.set_binding_shape(i, tuple(tensor.shape))
                bindings[i] = tensor.data_ptr()
                inputs.append(tensor)
            else:
                shape = tuple(context.get_binding_shape(i))
                print(f"Output {tensor_name}: shape {shape}")
                tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
                bindings[i] = tensor.data_ptr()
                outputs.append(tensor)

    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        if is_trt_10:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_async_v2(bindings, stream.cuda_stream)
    stream.synchronize()

    # CUDA Graph Capture
    cuda_graph = None
    if args.use_cuda_graph:
        try:
            print("Capturing CUDA Graph...")
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph, stream=stream):
                if is_trt_10:
                    context.execute_async_v3(stream_handle=stream.cuda_stream)
                else:
                    context.execute_async_v2(bindings, stream.cuda_stream)
            print("CUDA Graph captured successfully.")
        except Exception as e:
            print(f"Warning: Failed to capture CUDA Graph: {e}")
            print("Falling back to standard execution.")
            cuda_graph = None

    n_runs = args.num_runs
    print(f"Testing speed ({n_runs} runs)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(stream)
    for _ in range(n_runs):
        if cuda_graph is not None:
            cuda_graph.replay()
        elif is_trt_10:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_async_v2(bindings, stream.cuda_stream)
    end_event.record(stream)
    end_event.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_runs
    fps = 1000.0 / avg_time_ms

    print("\n" + "=" * 40)
    print("Benchmark Results (cuda):")
    print(f"Engine: {engine_file}")
    if len(inputs) > 0:
        print(f"Input Shape: {inputs[0].shape}")
        print(f"Precision: {'FP16' if inputs[0].dtype == torch.float16 else 'FP32'}")
    if args.use_cuda_graph and cuda_graph is not None:
        print("Optimization: CUDA Graph Enabled")
    print("Optimization: Pure GPU Inference (No Host-to-Device Copy)")
    print(f"Average Latency: {avg_time_ms:.4f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("=" * 40 + "\n")

if __name__ == '__main__':
    main()

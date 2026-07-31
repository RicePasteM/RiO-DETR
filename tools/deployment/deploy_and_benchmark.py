
import os
import sys
import argparse
import torch
import torch.nn as nn
import tensorrt as trt
import time
import glob

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from engine.core import YAMLConfig
except ImportError:
    print("Error: Could not import engine.core.YAMLConfig. Make sure you are in the project root.")
    sys.exit(1)

def export_onnx(config_path, resume_path, onnx_file, imgsz=640, simplify=True):
    print(f"\n[1/3] Exporting ONNX from {resume_path} with imgsz={imgsz}...")

    # Load Config
    cfg = YAMLConfig(config_path, resume=resume_path)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        print('Resume not provided. Initializing model with random weights...')

    # Wrapper Model for Deployment
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    # Dummy Input
    # Using 640x640 as default for most detection models, adjust if needed
    data = torch.rand(1, 3, imgsz, imgsz)
    size = torch.tensor([[imgsz, imgsz]])

    # Dry run
    _ = model(data, size)

    # Export
    torch.onnx.export(
        model,
        (data, size),
        onnx_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=None, # Static shape for TensorRT usually preferred for benchmark
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    print(f"ONNX exported to {onnx_file}")

    if simplify:
        try:
            import onnx
            import onnxsim
            print("Simplifying ONNX...")
            onnx_model = onnx.load(onnx_file)
            input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape}
            onnx_model_simplify, check = onnxsim.simplify(onnx_model, test_input_shapes=input_shapes)
            onnx.save(onnx_model_simplify, onnx_file)
            print("ONNX simplified successfully.")
        except ImportError:
            print("onnxsim not installed, skipping simplification.")
        except Exception as e:
            print(f"Simplification failed: {e}")

def build_engine(onnx_file_path, engine_file_path, fp16=True, imgsz=640, level=5):
    print(f"\n[2/3] Building TensorRT Engine: {engine_file_path} (Optimization Level: {level})...")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH flag is required for ONNX models
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    try:
        config.builder_optimization_level = level
    except AttributeError:
        print(f"Warning: builder_optimization_level not supported in this TensorRT version. Using default.")

    # Memory Pool
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 4GB

    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file {onnx_file_path} not found.")
        return False

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Force FP16 IO if requested
    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 Mode and Forcing FP16 IO...")
            config.set_flag(trt.BuilderFlag.FP16)

            # Set Network Inputs to FP16
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                if tensor.dtype != trt.float32:
                    print(f"Skipping input '{tensor.name}' for FP16 (type: {tensor.dtype})")
                    continue
                print(f"Setting input '{tensor.name}' to FP16")
                tensor.dtype = trt.float16
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

            # Set Network Outputs to FP16
            for i in range(network.num_outputs):
                tensor = network.get_output(i)
                if tensor.dtype != trt.float32:
                    print(f"Skipping output '{tensor.name}' for FP16 (type: {tensor.dtype})")
                    continue
                print(f"Setting output '{tensor.name}' to FP16")
                tensor.dtype = trt.float16
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        else:
            print("Warning: Platform does not support fast FP16. Using FP32.")

    # Optimization Profile for Dynamic Shapes (if any)
    # Even for static export, it's good practice to set this if dims are -1
    profile = builder.create_optimization_profile()
    found_dynamic = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        dims = inp.shape
        if -1 in dims:
            found_dynamic = True
            min_shape = []
            opt_shape = []
            max_shape = []
            for idx, d in enumerate(dims):
                if d == -1:
                    # Assume [1, 3, imgsz, imgsz] or [1, 2] typical
                    if idx == 0: val = 1
                    elif idx == 1 and inp.name == 'images': val = 3
                    elif idx >= 2 and inp.name == 'images': val = imgsz
                    elif inp.name == 'orig_target_sizes': val = 2 # usually [1, 2] or [N, 2]
                    else: val = 1
                    min_shape.append(val)
                    opt_shape.append(val)
                    max_shape.append(val)
                else:
                    min_shape.append(d)
                    opt_shape.append(d)
                    max_shape.append(d)
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

    if found_dynamic:
        config.add_optimization_profile(profile)

    print("Building serialized network...")
    try:
        plan = builder.build_serialized_network(network, config)
        if plan:
            with open(engine_file_path, 'wb') as f:
                f.write(plan)
            print(f"Engine built successfully: {engine_file_path}")
            return True
        else:
            print("Failed to build engine (result is None).")
            return False
    except Exception as e:
        print(f"Error building engine: {e}")
        return False

def benchmark_engine(engine_file, num_runs=1000, warmup=50, use_cuda_graph=False):
    print(f"\n[3/3] Benchmarking: {engine_file}...")

    if not os.path.exists(engine_file):
        print("Engine file not found.")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    # trt.init_libnvinfer_plugins(logger, '') # Not strictly needed if not using plugins, but good practice

    with open(engine_file, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if not engine:
        print("Failed to deserialize engine.")
        return

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    inputs = []
    outputs = []
    bindings = []

    is_trt_10 = hasattr(engine, 'get_tensor_mode')

    # Allocate Memory
    if is_trt_10:
        num_io = engine.num_io_tensors
        for i in range(num_io):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            dtype = engine.get_tensor_dtype(name)

            # Map TRT dtype to Torch dtype
            if dtype == trt.float16: t_dtype = torch.float16
            elif dtype == trt.int32: t_dtype = torch.int32
            elif dtype == trt.int8: t_dtype = torch.int8
            elif dtype == trt.bool: t_dtype = torch.bool
            else: t_dtype = torch.float32

            if mode == trt.TensorIOMode.INPUT:
                # Get shape (assuming static or opt profile 0)
                shape = tuple(engine.get_tensor_profile_shape(name, 0)[1])
                # context.set_input_shape(name, shape) # Not needed if static, but good for dynamic
                tensor = torch.randn(shape, dtype=t_dtype, device='cuda')
                context.set_input_shape(name, tuple(tensor.shape))
                context.set_tensor_address(name, int(tensor.data_ptr()))
                inputs.append(tensor)
                print(f"  Input: {name}, Shape: {shape}, Dtype: {t_dtype}")
            else:
                # Infer output shape
                shape = tuple(context.get_tensor_shape(name))
                tensor = torch.empty(shape, dtype=t_dtype, device='cuda')
                context.set_tensor_address(name, int(tensor.data_ptr()))
                outputs.append(tensor)
                print(f"  Output: {name}, Shape: {shape}, Dtype: {t_dtype}")

    else:
        # TRT 8.x fallback
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
             # Map TRT dtype to Torch dtype
            if dtype == trt.float16: t_dtype = torch.float16
            elif dtype == trt.int32: t_dtype = torch.int32
            elif dtype == trt.int8: t_dtype = torch.int8
            elif dtype == trt.bool: t_dtype = torch.bool
            else: t_dtype = torch.float32

            if engine.binding_is_input(i):
                shape = tuple(engine.get_profile_shape(0, i)[1])
                tensor = torch.randn(shape, dtype=t_dtype, device='cuda')
                context.set_binding_shape(i, tuple(tensor.shape))
                bindings.append(int(tensor.data_ptr()))
                inputs.append(tensor)
            else:
                shape = tuple(context.get_binding_shape(i))
                tensor = torch.empty(shape, dtype=t_dtype, device='cuda')
                bindings.append(int(tensor.data_ptr()))
                outputs.append(tensor)

    # Warmup
    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        if is_trt_10:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_async_v2(bindings, stream.cuda_stream)
    stream.synchronize()

    # CUDA Graph
    graph = None
    if use_cuda_graph:
        print("Capturing CUDA Graph...")
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if is_trt_10:
                    context.execute_async_v3(stream_handle=stream.cuda_stream)
                else:
                    context.execute_async_v2(bindings, stream.cuda_stream)
            print("CUDA Graph captured.")
        except Exception as e:
            print(f"CUDA Graph capture failed: {e}")

    # Test
    print(f"Running benchmark ({num_runs} runs)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(num_runs):
        if graph:
            graph.replay()
        elif is_trt_10:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_async_v2(bindings, stream.cuda_stream)
    end.record(stream)
    end.synchronize()

    avg_time = start.elapsed_time(end) / num_runs
    fps = 1000.0 / avg_time

    print("\n" + "="*40)
    print(f"Result for {engine_file}")
    print(f"Precision: {'FP16' if inputs[0].dtype == torch.float16 else 'FP32'}")
    print(f"Latency: {avg_time:.4f} ms")
    print(f"FPS: {fps:.2f}")
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Auto Deploy and Benchmark")
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config')
    parser.add_argument('--resume', '-r', default=None, help='Path to .pth checkpoint')
    parser.add_argument('--output', '-o', default='model_fp16.engine', help='Output engine path')
    parser.add_argument('--imgsz', '-s', type=int, default=640, help='Input image size')
    parser.add_argument('--level', '-l', type=int, default=5, help='TensorRT optimization level (0-5), default 5')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--cuda-graph', action='store_true', help='Enable CUDA Graph for benchmark')

    args = parser.parse_args()

    # Paths
    onnx_file = args.output.replace('.engine', '.onnx')
    engine_file = args.output

    # 1. Export ONNX
    export_onnx(args.config, args.resume, onnx_file, imgsz=args.imgsz)

    # 2. Build Engine (Forced FP16 IO by default unless --no-fp16)
    use_fp16 = not args.no_fp16
    if build_engine(onnx_file, engine_file, fp16=use_fp16, imgsz=args.imgsz, level=args.level):
        # 3. Benchmark
        benchmark_engine(engine_file, use_cuda_graph=args.cuda_graph)
    else:
        print("Engine build failed, skipping benchmark.")

if __name__ == '__main__':
    main()

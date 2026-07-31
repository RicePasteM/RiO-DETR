import tensorrt as trt
import os
import argparse

def build_engine(onnx_file_path, engine_file_path, fp16=False):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Set memory pool (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file {onnx_file_path} not found.")
        return

    print(f"Loading ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Check inputs and setup optimization profile if needed
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    profile = builder.create_optimization_profile()
    is_dynamic = False

    print("Input tensors:")
    for inp in inputs:
        print(f"  Name: {inp.name}, Shape: {inp.shape}, Dtype: {inp.dtype}")
        dims = inp.shape
        # Check for dynamic dimensions (-1)
        if -1 in dims:
            is_dynamic = True
            min_shape = []
            opt_shape = []
            max_shape = []

            for i, d in enumerate(dims):
                if d == -1:
                    # Heuristic: batch size usually dim 0
                    if i == 0:
                        min_shape.append(1)
                        opt_shape.append(1)
                        max_shape.append(1) # Allow up to batch 1 for static requirement
                    # Height/Width
                    elif i >= 2:
                        min_shape.append(800)
                        opt_shape.append(800)
                        max_shape.append(800)
                    else:
                        # Fallback for other dims like channels if dynamic (unlikely for images)
                        min_shape.append(1)
                        opt_shape.append(1)
                        max_shape.append(1)
                else:
                    min_shape.append(d)
                    opt_shape.append(d)
                    max_shape.append(d)

            print(f"  Dynamic shape detected for {inp.name}. Setting profile:")
            print(f"    Min: {min_shape}")
            print(f"    Opt: {opt_shape}")
            print(f"    Max: {max_shape}")
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)

    if is_dynamic:
        config.add_optimization_profile(profile)
    else:
        print("Model is static, no optimization profile needed.")

    if fp16:
        if builder.platform_has_fast_fp16:
            print("Enabling FP16 precision.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: Platform does not support fast FP16. Skipping FP16.")

    print("Building engine... this may take a while.")
    try:
        plan = builder.build_serialized_network(network, config)
        if plan:
            with open(engine_file_path, 'wb') as f:
                f.write(plan)
            print(f"Engine saved to {engine_file_path}")
        else:
            print("Failed to build engine.")
    except Exception as e:
        print(f"Error building engine: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", required=True, help="Path to ONNX file")
    parser.add_argument("--saveEngine", required=True, help="Path to save Engine file")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")

    args = parser.parse_args()

    build_engine(args.onnx, args.saveEngine, args.fp16)

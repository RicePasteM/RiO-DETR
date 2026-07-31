"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('Resume not provided. Initializing model with random weights for speed testing...')

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

    data = torch.rand(1, 3, 1024, 1024)
    size = torch.tensor([[1024, 1024]])
    _ = model(data, size)

    # dynamic_axes = {
    #     'images': {0: 'N', },
    #     'orig_target_sizes': {0: 'N'}
    # }

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    opset_version = 16
    # torch 2.4+ forces opset 18 for some ops, so we must use 18 or higher
    try:
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.4"):
            opset_version = 18
    except (ImportError, Exception):
        # Fallback if packaging is not installed
        try:
            v_str = torch.__version__.split('+')[0]
            v_parts = v_str.split('.')
            if int(v_parts[0]) > 2 or (int(v_parts[0]) == 2 and int(v_parts[1]) >= 4):
                opset_version = 18
        except:
            pass

    print(f'Exporting with opset_version={opset_version}...')

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=None,
        opset_version=opset_version,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        try:
            import onnx
            onnx_model = onnx.load(output_file)
            onnx.checker.check_model(onnx_model)
            print('Check export onnx model done...')
        except Exception as e:
            print(f'Check export onnx model failed: {e}')

    if args.simplify:
        try:
            import onnx
            import onnxsim
            dynamic = False
            # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
            input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
            onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
            onnx.save(onnx_model_simplify, output_file)
            print(f'Simplify onnx model {check}...')
        except Exception as e:
            print(f'Simplify onnx model failed: {e}')
            print('But the original onnx model is saved.')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/rtv2obb/trod_hgnetv2_s_diorr.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--check',  action='store_true', default=True,)
    parser.add_argument('--simplify',  action='store_true', default=True,)
    args = parser.parse_args()
    main(args)

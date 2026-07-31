"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

try:
    from dota_auto_eval import DOTAEvaluator
except ImportError:
    DOTAEvaluator = None

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    # Auto Evaluation Task Management
    if cfg.yaml_cfg.get('evaluator', {}).get('type') == 'DotaAutoEvaluator' and DOTAEvaluator is not None:
        if dist_utils.is_main_process():
            eval_cfg = cfg.yaml_cfg['evaluator']

            # Ensure output_dir is set
            if cfg.output_dir is None:
                cfg.output_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

            if not os.path.exists(cfg.output_dir):
                os.makedirs(cfg.output_dir, exist_ok=True)

            evaluator = DOTAEvaluator(
                base_url=eval_cfg.get('backend_url', 'http://dota-auto-eval.codesocean.top'),
                api_key=eval_cfg.get('api_key', ''),
                print_func=print
            )

            # Check if we are resuming or starting fresh
            task_id_file = os.path.join(cfg.output_dir, 'task_id.txt')
            last_epoch_file = os.path.join(cfg.output_dir, 'last_eval_epoch.txt')

            # If resuming (args.resume or resume config or files exist)
            is_resume = args.resume or cfg.resume

            if not is_resume:
                # Create new task
                # Read config file content as description
                try:
                    with open(args.config, 'r') as f:
                        config_content = f.read()
                except:
                    config_content = "No config content available"

                training_task_name = os.path.splitext(os.path.basename(args.config))[0] + '_' + time.strftime('%Y%m%d%H%M%S')

                try:
                    task_result = evaluator.create_training_task(
                        name=training_task_name,
                        description=config_content,
                        server_id=eval_cfg.get('server_id', 1)
                    )
                    print(f"创建的训练任务ID: {task_result['task_id']}")

                    os.environ['DOTA_AUTO_EVAL_TASK_ID'] = str(task_result['task_id'])
                    os.environ['work_dir'] = cfg.output_dir

                    with open(task_id_file, 'w') as f:
                        f.write(str(task_result['task_id']))
                    with open(last_epoch_file, 'w') as f:
                        f.write('0')
                except Exception as e:
                    print(f"创建训练任务失败: {e}")
                    # Fallback or exit? For now, print error but maybe continue
            else:
                # Resume task
                try:
                    if os.path.exists(task_id_file) and os.path.exists(last_epoch_file):
                        with open(task_id_file, 'r') as f:
                            task_id = int(f.read().strip())
                        with open(last_epoch_file, 'r') as f:
                            last_eval_epoch = int(f.read().strip())

                        print(f"恢复训练任务ID: {task_id}，上次评估轮数: {last_eval_epoch}")
                        os.environ['DOTA_AUTO_EVAL_TASK_ID'] = str(task_id)
                        os.environ['work_dir'] = cfg.output_dir
                    else:
                        print("Warning: Resume requested but task_id.txt or last_eval_epoch.txt not found.")
                except Exception as e:
                    print(f"恢复训练任务失败: {e}")

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

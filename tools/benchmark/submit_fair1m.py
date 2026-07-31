
import os
import sys
import argparse
import torch
import numpy as np
import cv2
import glob
import zipfile
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig
from engine.solver import TASKS
from engine.data.dataset import DOTADataset
from engine.data import DataLoader
from engine.data.transforms import Compose
from engine.misc import dist_utils

# FAIR1M Classes
FAIR1M_CLASSES = (
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321',
    'A220', 'A330', 'A350', 'C919', 'ARJ21', 'other-airplane',
    'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship',
    'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck',
    'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Truck_Tractor', 'Excavator', 'other-vehicle',
    'Baseball_Field', 'Basketball_Court', 'Football_Field', 'Tennis_Court', 'Roundabout', 'Intersection', 'Bridge'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Submit FAIR1M Task')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to test images')
    parser.add_argument('--output-dir', type=str, default='submission_output', help='Output directory for XMLs and Zip')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (single-GPU)')
    parser.add_argument('--batch-size', type=int, default=None, help='Inference batch size per GPU. Default: from config val_dataloader total_batch_size')
    return parser.parse_args()

def rbox2poly(rbox):
    """
    Convert rbox (cx, cy, w, h, angle) to poly (x1, y1, x2, y2, x3, y3, x4, y4)
    angle is in radian.
    """
    cx, cy, w, h, a = rbox

    # cv2.boxPoints expects degrees
    # Note: verify the angle definition.
    # RT-DETR OBB usually uses radian.
    # cv2.minAreaRect returns angle in degrees.
    # If a is radian, we can compute corners manually.

    cos = np.cos(a)
    sin = np.sin(a)

    wx, wy = w / 2 * cos, w / 2 * sin
    hx, hy = -h / 2 * sin, h / 2 * cos

    p1 = (cx - wx - hx, cy - wy - hy)
    p2 = (cx + wx - hx, cy + wy - hy)
    p3 = (cx + wx + hx, cy + wy + hy)
    p4 = (cx - wx + hx, cy - wy + hy)

    return np.array([p1, p2, p3, p4], dtype=np.float32)

def generate_xml(img_name, boxes, scores, labels, output_path):
    root = ET.Element('annotation')

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'filename').text = img_name
    ET.SubElement(source, 'origin').text = 'GF2/GF3'

    research = ET.SubElement(root, 'research')
    ET.SubElement(research, 'version').text = '1.0'
    ET.SubElement(research, 'provider').text = 'Company/School of team'
    ET.SubElement(research, 'author').text = 'Team name'
    ET.SubElement(research, 'pluginname').text = 'FAIR1M'
    ET.SubElement(research, 'pluginclass').text = 'object detection'
    ET.SubElement(research, 'time').text = '2021-03'

    objects = ET.SubElement(root, 'objects')

    for box, score, label in zip(boxes, scores, labels):
        # Filter low score if needed, but usually PostProcessor handles top-k

        # box is (x1, y1, x2, y2, x3, y3, x4, y4) or (cx, cy, w, h, a)
        # We converted to poly already in main loop

        obj = ET.SubElement(objects, 'object')
        ET.SubElement(obj, 'coordinate').text = 'pixel'
        ET.SubElement(obj, 'type').text = 'rectangle'
        ET.SubElement(obj, 'description').text = 'None'

        possibleresult = ET.SubElement(obj, 'possibleresult')
        label_idx = int(label)
        if label_idx < len(FAIR1M_CLASSES):
            cat_name = FAIR1M_CLASSES[label_idx]
        else:
            cat_name = f'unknown_{label_idx}'

        ET.SubElement(possibleresult, 'name').text = cat_name
        ET.SubElement(possibleresult, 'probability').text = f"{score:.6f}"

        points = ET.SubElement(obj, 'points')
        # box is np array of shape (4, 2)
        for i in range(4):
            pt = box[i]
            ET.SubElement(points, 'point').text = f"{pt[0]:.6f},{pt[1]:.6f}"
        # Add first point again to close the loop
        pt = box[0]
        ET.SubElement(points, 'point').text = f"{pt[0]:.6f},{pt[1]:.6f}"

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def main():
    args = parse_args()

    # 0. Optional: init distributed (multi-GPU via torchrun)
    dist_initialized = dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=None)
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    is_main = dist_utils.is_main_process()

    # 1. Config
    if is_main:
        print(f"Loading config from {args.config}...")
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Override validation dataset paths to point to test dir
    if 'val_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = args.test_dir
        cfg.yaml_cfg['val_dataloader']['dataset']['ann_folder'] = args.test_dir
        cfg.yaml_cfg['val_dataloader']['shuffle'] = False
        cfg.yaml_cfg['val_dataloader']['drop_last'] = False
        if args.batch_size is not None:
            # total_batch_size is split across GPUs by build_dataloader
            cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.batch_size * world_size

        transforms_cfg = cfg.yaml_cfg['val_dataloader']['dataset'].get('transforms', {})
        ops = transforms_cfg.get('ops', None)
        if not ops:
            if is_main:
                print("Warning: val_dataloader transforms ops is empty. Adding default Resize and Normalize.")
            eval_size = cfg.yaml_cfg.get('eval_spatial_size', [1024, 1024])
            cfg.yaml_cfg['val_dataloader']['dataset']['transforms'] = {
                'type': 'Compose',
                'ops': [
                    {'type': 'ResizeOBB', 'size': eval_size},
                    {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True},
                    {'type': 'ConvertOBB', 'normalize': True}
                ]
            }

    # 2. Solver & Model
    if is_main:
        print("Building model...")
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver._setup()
    raw_model = solver.model.module if hasattr(solver.model, 'module') else solver.model
    if dist_initialized:
        device = solver.device
        model = solver.model
    else:
        device = torch.device(args.device)
        model = raw_model.to(device)

    # Load checkpoint
    if args.resume:
        if is_main:
            print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        raw_model.load_state_dict(new_state_dict)

    model.eval()

    # 3. Data Loader (with optional DistributedSampler for multi-GPU)
    if is_main:
        print("Building data loader...")
    val_loader = cfg.build_dataloader('val_dataloader')
    if dist_initialized:
        val_loader = dist_utils.warp_loader(val_loader, shuffle=False)

    # 4. PostProcessor
    postprocessor = solver.postprocessor

    # 5. Output Dir (all ranks write to same dir; each rank writes different images)
    xml_dir = os.path.join(args.output_dir, 'test')
    os.makedirs(xml_dir, exist_ok=True)
    if dist_initialized:
        torch.distributed.barrier()

    # 6. Inference (multi-GPU: each rank only sees its share of batches via DistributedSampler)
    n_batches = len(val_loader)
    if is_main:
        print(f"Starting inference: world_size={world_size}, batch_size={val_loader.batch_size} per GPU, ~{n_batches * val_loader.batch_size} images/rank, {len(val_loader.dataset)} total.")

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(val_loader, disable=not is_main)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            for j, result in enumerate(results):
                if 'idx' in targets[j]:
                    idx = targets[j]['idx'].item()
                    img_id = val_loader.dataset.img_ids[idx]
                else:
                    if is_main:
                        print("Error: 'idx' not found in target. Skipping.")
                    continue

                ext = '.tif'
                for e in ['.tif', '.tiff', '.png', '.jpg', '.bmp']:
                    if os.path.exists(os.path.join(args.test_dir, img_id + e)):
                        ext = e
                        break
                img_name = f"{img_id}{ext}"

                boxes = result['boxes'].cpu().numpy()
                scores = result['scores'].cpu().numpy()
                labels = result['labels'].cpu().numpy()

                polys = []
                for box in boxes:
                    poly = rbox2poly(box)
                    polys.append(poly)

                xml_path = os.path.join(xml_dir, f"{img_id}.xml")
                generate_xml(img_name, polys, scores, labels, xml_path)

    # 7. Zip (only rank 0; ensure all ranks have written)
    if dist_initialized:
        torch.distributed.barrier()
    if is_main:
        print("Zipping results...")
        zip_path = os.path.join(args.output_dir, 'test.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for xml_file in sorted(glob.glob(os.path.join(xml_dir, '*.xml'))):
                zf.write(xml_file, os.path.join('test', os.path.basename(xml_file)))
        print(f"Submission file generated at {zip_path}")

if __name__ == '__main__':
    main()

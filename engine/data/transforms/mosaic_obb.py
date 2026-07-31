
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register
from .transforms_obb import ResizeOBB, RandomAffineOBB

@register()
class MosaicOBB(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images for OBB.
    Combines four randomly selected images into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        super().__init__()
        self.resize = ResizeOBB(size=output_size, max_size=max_size)
        self.probability = probability
        # Use RandomAffineOBB instead of T.RandomAffine
        self.affine_transform = RandomAffineOBB(degrees=rotation_range, translate=translation_range,
                                               scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size

        # Resize main image
        res = self.resize(image, target)
        if isinstance(res, tuple):
            image, target = res[0], res[1]

        resized_images, resized_targets = [image], [target]
        h, w = get_size_func(resized_images[0])
        if isinstance(h, torch.Tensor): h = h.item()
        if isinstance(w, torch.Tensor): w = w.item()
        max_height, max_width = h, w

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # Load item from dataset
            img_i, target_i = dataset.load_item(idx)

            # Resize
            res_i = self.resize(img_i, target_i)
            if isinstance(res_i, tuple):
                img_i, target_i = res_i[0], res_i[1]

            height, width = get_size_func(img_i)

            # Ensure height/width are int
            if isinstance(height, torch.Tensor): height = height.item()
            if isinstance(width, torch.Tensor): width = width.item()

            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(img_i)
            resized_targets.append(target_i)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        # Resize main image
        res = self.resize(image, target)
        if isinstance(res, tuple):
            image, target = res[0], res[1]

        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)

        sample_indices = random.choices(range(len(cache)), k=3)

        # Clone samples from cache
        mosaic_samples = []
        for idx in sample_indices:
            cached_item = cache[idx]
            # Deep copy labels
            labels_copy = self._clone(cached_item["labels"])
            img_copy = cached_item["img"] # Tensor is not cloned by default assignment, but we will rotate it which creates new tensor
            if isinstance(img_copy, torch.Tensor):
                img_copy = img_copy.clone()
            elif hasattr(img_copy, 'copy'):
                img_copy = img_copy.copy()

            mosaic_samples.append(dict(img=img_copy, labels=labels_copy))

        # Add main image to samples
        # Clone main image/target before rotate just in case
        main_target_copy = self._clone(target)
        main_img_copy = image.clone() if isinstance(image, torch.Tensor) else image.copy()

        mosaic_samples = [dict(img=main_img_copy, labels=main_target_copy)] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]

        # Create merged image
        img0 = mosaic_samples[0]["img"]
        if isinstance(img0, torch.Tensor):
            # Tensor image: (C, H, W)
            c = img0.shape[0]
            merged_image = torch.full((c, max_height * 2, max_width * 2), 114/255.0, dtype=img0.dtype) # Default fill
        else:
            # PIL Image
            merged_image = Image.new(mode=img0.mode, size=(max_width * 2, max_height * 2), color=0)

        # offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]) # (4, 2)

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            # Paste image
            dx, dy = placement_offsets[i]
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
                merged_image[:, dy:dy+h, dx:dx+w] = img
            else:
                merged_image.paste(img, tuple(placement_offsets[i]))

            # Shift boxes (cx, cy)
            # boxes: (N, 5) -> cx, cy, w, h, a
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                target['boxes'][:, 0] += dx
                target['boxes'][:, 1] += dy

            mosaic_target.append(target)

        merged_target = {}
        if len(mosaic_target) > 0:
            for key in mosaic_target[0]:
                if isinstance(mosaic_target[0][key], torch.Tensor):
                    merged_target[key] = torch.cat([target[key] for target in mosaic_target])
                else:
                    pass

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""

        # Ensure max_height/max_width are int
        if isinstance(max_height, torch.Tensor): max_height = int(max_height.item())
        if isinstance(max_width, torch.Tensor): max_width = int(max_width.item())

        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]

        img0 = images[0]
        if isinstance(img0, torch.Tensor):
            c = img0.shape[0]
            merged_image = torch.full((c, max_height * 2, max_width * 2), 114/255.0, dtype=img0.dtype)
        else:
            merged_image = Image.new(mode=img0.mode, size=(max_width * 2, max_height * 2), color=0)

        mosaic_target = []
        for i, img in enumerate(images):
            dx, dy = placement_offsets[i]

            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
                merged_image[:, dy:dy+h, dx:dx+w] = img
            else:
                merged_image.paste(img, tuple(placement_offsets[i]))

        # Merge targets
        merged_target = {}
        if len(targets) > 0:
            for key in targets[0]:
                if key == 'boxes':
                    # Shift boxes
                    shifted_boxes = []
                    for i, target in enumerate(targets):
                        dx, dy = placement_offsets[i]
                        boxes = target['boxes'].clone()
                        if boxes.shape[0] > 0:
                            boxes[:, 0] += dx
                            boxes[:, 1] += dy
                        shifted_boxes.append(boxes)
                    merged_target[key] = torch.cat(shifted_boxes, dim=0)
                elif isinstance(targets[0][key], torch.Tensor):
                    merged_target[key] = torch.cat([target[key] for target in targets], dim=0)

        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target, dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Apply affine transformations
        # Affine expects (img, target)
        res = self.affine_transform(mosaic_image, mosaic_target)
        if isinstance(res, tuple):
            mosaic_image, mosaic_target = res[0], res[1]
        else:
            mosaic_image = res

        return mosaic_image, mosaic_target, dataset

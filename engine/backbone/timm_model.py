"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#0583
"""
import torch

from .utils import IntermediateLayerGetter
from ..core import register


@register()
class TimmModel(torch.nn.Module):
    def __init__(self, \
        name,
        return_layers,
        pretrained=False,
        exportable=True,
        features_only=True,
        **kwargs) -> None:

        super().__init__()

        import timm
        model = timm.create_model(
            name,
            pretrained=pretrained,
            exportable=exportable,
            features_only=features_only,
            **kwargs
        )

        assert set(return_layers).issubset(model.feature_info.module_name()), \
            f'return_layers should be a subset of {model.feature_info.module_name()}'

        self.model = IntermediateLayerGetter(model, return_layers)

        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs


if __name__ == '__main__':

    model = TimmModel(name='resnet34', return_layers=['layer2', 'layer3'])
    data = torch.rand(1, 3, 640, 640)
    outputs = model(data)

    for output in outputs:
        print(output.shape)

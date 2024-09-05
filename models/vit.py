import timm
from utils.configurable import configurable
from models.build import MODELS_REGISTRY


def _cfg_to_vit(args):
    return {"num_classes": args.n_classes}


@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def vit(num_classes=200):
    return timm.create_model(
        "vit_small_patch16_224", pretrained=False, num_classes=num_classes
    )

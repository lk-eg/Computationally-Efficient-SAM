from torchvision.models import densenet121
from utils.configurable import configurable
from models.build import MODELS_REGISTRY


def _cfg_to_vit(args):
    return {"num_classes": args.n_classes}


@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_vit)
def densenet(num_classes=200):
    model = densenet121(pretrained=False, num_classes=200)
    return model

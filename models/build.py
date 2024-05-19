from utils.register import Registry 
import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_REGISTRY = Registry("Models")

def build_model(args):
    model = MODELS_REGISTRY.get(args.model)(args)
    model = model.to(device)
    return model
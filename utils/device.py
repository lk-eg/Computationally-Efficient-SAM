import torch
import platform

if platform.system() == "Darwin":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def onHPC():
    if platform.system() == "Linux":
        return True
    else:
        return False

import torch
import platform
import distro

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


def onServer():
    if distro.name() == "CentOS Linux":
        return True
    else:
        return False


def dataset_directory():
    if distro.name() == "CentOS Linux":
        return "/home/laltun/datasets"
    elif distro.name() == "Darwin":
        return "~/sam/datasets"
    else:
        return "/cluster/home/laltun/datasets"

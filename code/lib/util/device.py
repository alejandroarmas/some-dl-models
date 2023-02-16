import torch


def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # high-performance training on GPU for MacOS devices with Metal programming framework
        return torch.device("mps")

    return torch.device("cpu")

import torch


class BaseInference:
    def __init__(self, device: str = torch.device("cpu")):
        self.device = device
        pass

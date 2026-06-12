import torch.nn as nn
from .hybrid_vit import HybridQCNNViT


class ClassicalViT(HybridQCNNViT):
    def __init__(self, config):
        config.use_quantum = False
        super().__init__(config)

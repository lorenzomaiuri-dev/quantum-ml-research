import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.qcnn import QuantumPatchEmbedding


class HybridQCNNViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.use_quantum:
            self.patch_embed = QuantumPatchEmbedding(
                patch_dim=config.patch_dim,
                embed_dim=config.embed_dim,
                n_qlayers=config.n_qlayers,
            )
        else:
            self.patch_embed = nn.Linear(config.patch_dim, config.embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, config.seq_len, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_head,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.n_classes)

    def forward(self, x, labels=None):
        B, C, H, W = x.shape

        p = self.config.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, p * p * C)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.pos_drop(torch.cat((cls_tokens, x), dim=1) + self.pos_embedding)

        x = self.transformer(x)
        logits = self.head(self.norm(x[:, 0]))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

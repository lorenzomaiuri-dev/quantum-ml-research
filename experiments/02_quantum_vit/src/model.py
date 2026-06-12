import torch
import torch.nn as nn
from torch.nn import functional as F

from src.quantum_layers import QuantumLinear


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each
    patch into a vector of size embed_dim.

    For a 28×28 image with patch_size=7:
        4×4 grid = 16 patches, each 7×7×C → Linear → embed_dim
    """

    def __init__(self, config):
        super().__init__()
        # Conv2d with kernel=stride=patch_size acts as patch extraction + projection
        self.proj = nn.Conv2d(
            config.n_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//p, W//p)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """
    Self-attention with support for both quantum and classical Q/K/V projections.

    Quantum path:  Each head has its own VQC triplet (Q, K, V).
                   Input is split by head, processed through per-head VQCs.
    Classical path: Standard nn.Linear projects all heads at once.

    Note: ViT uses bidirectional attention — no causal mask.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.embed_dim // config.n_head
        self.use_quantum = config.use_quantum

        if config.use_quantum:
            # Each head gets its own Q/K/V quantum circuits
            self.q_heads = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "q": QuantumLinear(
                                self.head_size, config.n_qlayers, config.q_device
                            ),
                            "k": QuantumLinear(
                                self.head_size, config.n_qlayers, config.q_device
                            ),
                            "v": QuantumLinear(
                                self.head_size, config.n_qlayers, config.q_device
                            ),
                        }
                    )
                    for _ in range(config.n_head)
                ]
            )
        else:
            self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        if self.use_quantum:
            # Split input by head: (B, T, embed_dim) → list of (B, T, head_size)
            x_heads = x.chunk(self.n_head, dim=-1)

            q_list, k_list, v_list = [], [], []
            for i, x_h in enumerate(x_heads):
                q_list.append(self.q_heads[i]["q"](x_h))
                k_list.append(self.q_heads[i]["k"](x_h))
                v_list.append(self.q_heads[i]["v"](x_h))

            # Stack heads: (B, n_head, T, head_size)
            q = torch.stack(q_list, dim=1)
            k = torch.stack(k_list, dim=1)
            v = torch.stack(v_list, dim=1)
        else:
            # Classical: single linear projects all heads at once
            q = (
                self.query(x)
                .view(B, T, self.n_head, self.head_size)
                .transpose(1, 2)
            )
            k = (
                self.key(x)
                .view(B, T, self.n_head, self.head_size)
                .transpose(1, 2)
            )
            v = (
                self.value(x)
                .view(B, T, self.n_head, self.head_size)
                .transpose(1, 2)
            )

        # Scaled dot-product attention (bidirectional, no mask)
        attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (classical)."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attention → Residual → LN → FFN → Residual."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class QuantumViT(nn.Module):
    """
    Vision Transformer with optional quantum-enhanced attention.

    Architecture:
        Image → Patch Embedding → [CLS] + Position → Transformer Blocks → Classify

    The CLS token aggregates information from all patches through attention
    and is used as the image representation for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.seq_len, config.embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(config.dropout)

        # Transformer encoder
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Classification head
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.n_classes)

    def forward(self, x, targets=None):
        """
        Args:
            x: (B, C, H, W) — batch of images
            targets: (B,) — class labels (optional, for loss computation)
        Returns:
            logits: (B, n_classes)
            loss: scalar or None
        """
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) → (B, n_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, n_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        x = self.blocks(x)

        # Take CLS token output → classify
        cls_out = self.ln_f(x[:, 0])
        logits = self.head(cls_out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

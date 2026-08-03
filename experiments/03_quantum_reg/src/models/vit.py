"""
Vision Transformer classifier — shared across all ablation models.

The transformer encoder is IDENTICAL for models A, B, and C.
Only the patch embedding differs. This ensures any difference
in generalization is attributable to the patch embedding alone.
"""

import hashlib

import torch
import torch.nn as nn

from src.models.patch_embeddings import build_patch_embedding


class AblationViT(nn.Module):
    """
    ViT for the ablation study.

    Architecture:
        Patch Embedding (varies by model_type) → [CLS] + PosEmbed
        → TransformerEncoder → LayerNorm → Linear → Classes

    Args:
        config: AblationConfig
        model_type: "vanilla" | "bounded_mlp" | "quantum_reg"
    """

    def __init__(self, config, model_type: str):
        super().__init__()
        self.config = config
        self.model_type = model_type

        # Patch embedding — the only component that changes
        self.patch_embed = build_patch_embedding(config, model_type)

        # Shared ViT components
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.seq_len, config.embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_head,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layer
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.n_classes)

    def forward(self, x, labels=None):
        """
        Args:
            x: (B, C, H, W) images
            labels: (B,) class indices, optional
        Returns:
            logits: (B, n_classes)
            loss: scalar or None
        """
        B = x.shape[0]

        # Extract patches: (B, C, H, W) → (B, n_patches, patch_dim)
        p = self.config.patch_size
        # unfold extracts patches as sliding windows with stride=patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H//p, W//p, p, p)
        x = x.contiguous().view(B, self.config.n_channels, -1, p, p)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, -1, self.config.patch_dim)  # (B, n_patches, patch_dim)

        # Patch embedding (MODEL-SPECIFIC)
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Prepend CLS + positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer encoder (SHARED)
        x = self.transformer(x)

        # Classify from CLS token
        logits = self.head(self.norm(x[:, 0]))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return logits, loss

    def count_params(self):
        """Returns total and per-component parameter counts."""
        patch_params = sum(p.numel() for p in self.patch_embed.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        other_params = (
            self.cls_token.numel()
            + self.pos_embed.numel()
            + sum(p.numel() for p in self.norm.parameters())
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "patch_embed": patch_params,
            "transformer": transformer_params,
            "classifier": head_params,
            "other": other_params,
        }


def match_shared_initialization(models: dict[str, AblationViT]) -> dict[str, object]:
    """Copy the compression layer and shared ViT body from vanilla to all variants.

    Equal seeds alone are insufficient because each model-specific embedding
    consumes a different number of random draws before the shared body is
    constructed. The returned digest identifies the exact common starting
    tensors used by one paired dataset/seed block.
    """
    required = {"vanilla", "bounded_mlp", "quantum_reg"}
    if set(models) != required:
        raise ValueError(f"expected exactly {sorted(required)}, got {sorted(models)}")

    reference_state = models["vanilla"].state_dict()
    target_states = {
        name: model.state_dict() for name, model in models.items() if name != "vanilla"
    }
    shared_keys = []
    digest = hashlib.sha256()

    with torch.no_grad():
        for key in sorted(reference_state):
            is_shared = key.startswith("patch_embed.compression.") or not key.startswith(
                "patch_embed."
            )
            if not is_shared:
                continue
            source = reference_state[key]
            if any(
                key not in state or state[key].shape != source.shape
                for state in target_states.values()
            ):
                raise ValueError(f"shared tensor mismatch for {key}")
            for state in target_states.values():
                state[key].copy_(source)
            tensor = source.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
            shared_keys.append(key)

    for name, state in target_states.items():
        models[name].load_state_dict(state)

    return {"shared_state_keys": shared_keys, "sha256": digest.hexdigest()}

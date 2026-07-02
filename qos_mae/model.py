from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class PatchMeta:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    patch_size: int
    num_patch_rows: int
    num_patch_cols: int

    @property
    def num_patches(self) -> int:
        return self.num_patch_rows * self.num_patch_cols

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size


def pad_to_patch(arr: np.ndarray, patch_size: int, fill_value: float | bool = 0) -> tuple[np.ndarray, PatchMeta]:
    h, w = arr.shape
    hp = int(np.ceil(h / patch_size) * patch_size)
    wp = int(np.ceil(w / patch_size) * patch_size)
    padded = np.full((hp, wp), fill_value, dtype=arr.dtype)
    padded[:h, :w] = arr
    meta = PatchMeta((h, w), (hp, wp), patch_size, hp // patch_size, wp // patch_size)
    return padded, meta


def patchify(arr: np.ndarray, patch_size: int, fill_value: float | bool = 0) -> tuple[np.ndarray, PatchMeta]:
    padded, meta = pad_to_patch(arr, patch_size, fill_value)
    p = patch_size
    hp, wp = meta.padded_shape
    patches = padded.reshape(hp // p, p, wp // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)
    return patches, meta


def unpatchify(patches: np.ndarray, meta: PatchMeta) -> np.ndarray:
    p = meta.patch_size
    hp, wp = meta.padded_shape
    arr = patches.reshape(meta.num_patch_rows, meta.num_patch_cols, p, p).transpose(0, 2, 1, 3).reshape(hp, wp)
    h, w = meta.original_shape
    return arr[:h, :w]


def patch_reputation(qos_rep: np.ndarray, valid_mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, PatchMeta]:
    rep_patches, meta = patchify(qos_rep.astype(np.float32), patch_size, fill_value=1.0)
    valid_patches, _ = patchify(valid_mask.astype(np.float32), patch_size, fill_value=0.0)
    denom = valid_patches.sum(axis=1)
    out = np.ones(rep_patches.shape[0], dtype=np.float32)
    nonzero = denom > 0
    out[nonzero] = (rep_patches[nonzero] * valid_patches[nonzero]).sum(axis=1) / denom[nonzero]
    return np.clip(out, 0.0, 1.0), meta


class QoSMAE(nn.Module):
    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        encoder_dim: int = 256,
        decoder_dim: int = 128,
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        encoder_heads: int = 8,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        base_mask_rate: float = 0.75,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.base_mask_rate = base_mask_rate
        self.beta = beta
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)
        self.encoder_pos = nn.Parameter(torch.zeros(1, num_patches, encoder_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=int(encoder_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.encoder_norm = nn.LayerNorm(encoder_dim)
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.encoder_pos, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def sample_mask(self, patch_reputation: torch.Tensor) -> torch.Tensor:
        patch_reputation = patch_reputation.detach().clamp(0.0, 1.0)
        prob = (self.base_mask_rate + self.beta * (1.0 - patch_reputation)).clamp(0.0, 1.0)
        mask = torch.bernoulli(prob).bool()
        if mask.all():
            idx = int(torch.argmax(patch_reputation).item())
            mask[idx] = False
        if (~mask).all():
            idx = int(torch.argmin(patch_reputation).item())
            mask[idx] = True
        return mask

    def forward(
        self,
        patches: torch.Tensor,
        patch_reputation: torch.Tensor,
        force_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
        if patches.shape[1] != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {patches.shape[1]}")
        bsz = patches.shape[0]
        x = self.patch_embed(patches) + self.encoder_pos
        mask = force_mask.to(patches.device).bool() if force_mask is not None else self.sample_mask(patch_reputation.to(patches.device))
        visible_idx = torch.nonzero(~mask, as_tuple=False).flatten()
        x_vis = x[:, visible_idx, :]
        enc = self.encoder_norm(self.encoder(x_vis))
        enc = self.enc_to_dec(enc)
        dec_tokens = self.mask_token.expand(bsz, self.num_patches, -1).clone()
        dec_tokens[:, visible_idx, :] = enc
        dec_tokens = dec_tokens + self.decoder_pos
        dec = self.decoder_norm(self.decoder(dec_tokens))
        pred = self.decoder_pred(dec)
        return pred.squeeze(0), mask

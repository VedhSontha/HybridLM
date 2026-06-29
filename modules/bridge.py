# modules/bridge.py
"""
Bridge module — aligns encoder (BERT) embeddings with decoder (LLaMA) space.
"""

import os
import torch
from torch import nn


class Bridge(nn.Module):
    """
    Simple linear projection + normalization bridge.
    Maps encoder hidden states (e.g., 768) → decoder hidden states (e.g., 4096).
    """

    def __init__(self, encoder_dim=768, decoder_dim=4096, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, seq_len, encoder_dim)
        Returns:
            torch.Tensor of shape (batch_size, seq_len, decoder_dim)
        """
        return self.projection(x)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "bridge.pt"))

    def load_pretrained(self, path: str):
        file_path = os.path.join(path, "bridge.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No bridge checkpoint found at: {file_path}")
        self.load_state_dict(torch.load(file_path, map_location="cpu"))

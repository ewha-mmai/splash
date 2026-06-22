import torch
import torch.nn as nn
from typing import Optional


class TactileProjector(nn.Module):
    """
    Tactile → LLM token projector (modality adapter)

    Input:
        tactile features [B, N_patch, D_tactile]
    Output:
        LLM tokens [B, N_patch, D_llm]
    """

    def __init__(
        self,
        in_dim: int,
        llm_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        hidden_dim = hidden_dim or llm_dim

        layers = [nn.Linear(in_dim, hidden_dim)]

        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.GELU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, llm_dim))

        self.proj = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, N_patch, D_tactile]

        Returns:
            tokens: [B, N_patch, D_llm]
        """
        return self.proj(feats)
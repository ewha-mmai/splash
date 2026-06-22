import torch
import torch.nn as nn
import timm
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TactileEncoder(nn.Module):
    """
    This module defines the Tactile Encoder based on a Vision Transformer (ViT-Tiny).
    It uses ImageNet-1K pretrained weights by default via timm.

    Input:
        tactile image [B, 3, 224, 224]
    Output:
        tactile features [B, N_patch, D_tactile]
    """

    def __init__(
        self, 
        model_name: str = 'vit_tiny_patch16_224', 
    ):
        """
        Args:
            model_name (str): Name of the timm model (default: ViT-Tiny).
            pretrained_path (str, optional): Path to the TVL checkpoint file.
        """
        super().__init__()
        
        logger.info(f"Initializing TactileEncoder with backbone: {model_name}")

        with torch.device("cpu"):
            self.encoder = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0
            )
        
        self.embed_dim = self.encoder.num_features


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]

        Returns:
            feats: [B, N_patch, D_tactile]
                   (ViT-Tiny: N_patch=196, D_tactile=192)
        """
        feats = self.encoder.forward_features(x)
        
        feats = feats[:, 1:, :]
        
        return feats
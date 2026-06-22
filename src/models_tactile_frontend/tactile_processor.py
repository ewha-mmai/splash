"""
Tactile Image Processor
-----------------------
This module provides the preprocessing logic for tactile images (e.g., GelSight data)
before feeding them into the Tactile Encoder (ViT-Tiny).

It follows standard ImageNet normalization and resizing protocols required by 
Vision Transformers.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Union, Optional

class TactileProcessor:
    """
    A processor class for preparing tactile images for the ViT-based encoder.
    
    Attributes:
        size (int): The target resolution for resizing (default: 224).
        transform (transforms.Compose): The composition of image transformations.
    """

    def __init__(self, size: int = 224):
        """
        Initializes the TactileProcessor with standard ImageNet normalization.

        Args:
            size (int): Input resolution for the Vision Transformer. Default is 224.
        """
        self.size = size
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __call__(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> torch.Tensor:
        """
        Processes a list of image paths or PIL Image objects into a batch tensor.

        Args:
            images (Union[str, Image.Image, List]): A single image or a list of images.
                Each item can be a file path (str) or a PIL.Image object.

        Returns:
            torch.Tensor: A batch of preprocessed images with shape [Batch, 3, H, W].
        """
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert('RGB')
                except Exception as e:
                    print(f"[Error] Failed to load image: {img}. Details: {e}")
                    img = Image.new('RGB', (self.size, self.size), (0, 0, 0))
            
            if not isinstance(img, Image.Image):
                 if isinstance(img, np.ndarray):
                     img = Image.fromarray(img.astype('uint8')).convert('RGB')
                 else:
                     raise TypeError(f"Unsupported image type: {type(img)}")

            tensor_img = self.transform(img)
            processed_images.append(tensor_img)
        
        return torch.stack(processed_images)

    def __repr__(self):
        return f"TactileProcessor(size={self.size}, normalization=ImageNet)"
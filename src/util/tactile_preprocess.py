"""Tactile image preprocessing utilities used by SPLASH.

Adapted from the TVL tactile preprocessing utilities. This module keeps only
the image/tactile transforms needed by SPLASH and its baseline wrappers.
"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


RGB_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
RGB_STD = np.array([0.26862954, 0.26130258, 0.27577711])

TAC_MEAN = np.array([0.29174602047139075, 0.2971325588927249, 0.2910404549605639])
TAC_STD = np.array([0.18764469044810236, 0.19467651810273057, 0.21871583397361818])

TAC_BG_FP = str(Path(__file__).resolve().parent / "data" / "tac_background.png")
TAC_MEAN_BG = np.array([-0.00809318389762342, -0.01887447008747725, -0.018430588238856332])
TAC_STD_BG = np.array([0.04535400223885517, 0.044029170444552575, 0.05332520729596308])

TO_TENSOR = transforms.Compose([transforms.ToTensor()])

RGB_AUGMENTS = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=(0.8, 1.1),
                    contrast=(0.7, 1.3),
                    saturation=0.2,
                    hue=0.0,
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.5, 1))], p=0.5),
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ]
)

RGB_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ]
)


def to_pil(img: torch.Tensor) -> Image.Image:
    img = np.moveaxis(img.numpy() * 255, 0, -1)
    return Image.fromarray(img.astype(np.uint8))


def unnormalize_fn(mean: tuple, std: tuple) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            transforms.ToPILImage(),
        ]
    )


class RandomDiscreteRotation(nn.Module):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, random.choice(self.angles))


TAC_AUGMENTS = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=0.2,
                    hue=0.05,
                )
            ],
            p=0.8,
        ),
        RandomDiscreteRotation([0, 90]),
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=TAC_MEAN, std=TAC_STD),
    ]
)

TAC_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=TAC_MEAN, std=TAC_STD),
    ]
)

TAC_WBG = transforms.Compose(
    [
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=TAC_MEAN, std=TAC_STD),
    ]
)


def tac_padding(tac: Union[torch.Tensor, Image.Image]):
    if isinstance(tac, Image.Image):
        tac_w, tac_h = tac.size
    else:
        tac_h, tac_w = tac.shape[1:]
    hpad = int(np.clip(max(tac_h, tac_w) - tac_h, 0, np.inf) / 2)
    wpad = int(np.clip(max(tac_h, tac_w) - tac_w, 0, np.inf) / 2)
    tac = TF.pad(tac, [wpad, hpad])
    return TF.rotate(tac, 90)


class BackgroundOps(torch.nn.Module):
    def __init__(self, background_fp: str, op: str = "subtract", padding: bool = True) -> None:
        super().__init__()
        self.background_fp = background_fp
        self.background = TO_TENSOR(Image.open(background_fp))
        if padding:
            self.background = tac_padding(self.background)
        assert op in ["subtract", "add"], "op must be either subtract or add"
        if op == "subtract":
            self.background = -self.background

    def forward(self, img: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = TO_TENSOR(img)
        return img + self.background.to(img.device)

    def __repr__(self):
        return f"{self.__class__.__name__}(background={self.background_fp})"


class SyncRandomBackgroundSubtract(torch.nn.Module):
    def __init__(self, transform: torch.nn.Module, background_fp: str, p: float = 0.8) -> None:
        super().__init__()
        self.background_fp = background_fp
        self.background = tac_padding(TO_TENSOR(Image.open(background_fp)))
        self.background = to_pil(self.background)
        self.p = p
        self.transform = transform
        assert isinstance(self.transform, transforms.ColorJitter), "transform must be a ColorJitter transform"

    def forward(self, img: Image.Image) -> torch.Tensor:
        if np.random.rand() < self.p:
            _, brightness, contrast, saturation, hue = self.transform.get_params(
                self.transform.brightness,
                self.transform.contrast,
                self.transform.saturation,
                self.transform.hue,
            )
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)

            background = TF.adjust_brightness(deepcopy(self.background), brightness)
            background = TF.adjust_contrast(background, contrast)
            background = TF.adjust_saturation(background, saturation)
            background = TF.adjust_hue(background, hue)
        else:
            background = deepcopy(self.background)
        return TO_TENSOR(img) - TO_TENSOR(background)

    def __repr__(self):
        return f"{self.__class__.__name__}(background={self.background_fp}, transform={self.transform})"


def tac_subtract_bg_sync_aug(fp: str, mean: tuple, std: tuple, p: float = 0.8) -> transforms.Compose:
    cj = transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(0.9, 1.1),
        saturation=0.2,
        hue=0.05,
    )
    return transforms.Compose(
        [
            SyncRandomBackgroundSubtract(transform=cj, background_fp=fp, p=p),
            transforms.RandomHorizontalFlip(),
            RandomDiscreteRotation([0, 90]),
            transforms.Resize(size=224),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def tac_subtract_bg_aug(fp: str, mean: tuple, std: tuple, color_jitter: bool = True) -> transforms.Compose:
    all_trs = [
        BackgroundOps(fp, op="subtract"),
        transforms.RandomHorizontalFlip(),
        RandomDiscreteRotation([0, 90]),
        transforms.Resize(size=224),
        transforms.Normalize(mean=mean, std=std),
    ]
    if color_jitter:
        all_trs = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=(0.9, 1.1),
                        contrast=(0.9, 1.1),
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.8,
            )
        ] + all_trs
    return transforms.Compose(all_trs)


def tac_subtract_bg(fp: str, mean: tuple, std: tuple) -> transforms.Compose:
    return transforms.Compose(
        [
            BackgroundOps(fp, op="subtract"),
            transforms.Resize(size=224),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


TAC_AUGMENTS_BG = tac_subtract_bg_aug(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG, color_jitter=False)
TAC_AUGMENTS_BG_CJ = tac_subtract_bg_sync_aug(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)
TAC_BG = tac_subtract_bg(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)


def unnormalize_fn_bg(fp: str, mean: tuple, std: tuple) -> transforms.Compose:
    background_ops = BackgroundOps(fp, op="add")
    h, w = background_ops.background.shape[1:]
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Resize(size=(h, w)),
            background_ops,
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            transforms.ToPILImage(),
        ]
    )


TAC_BG_UNDO = unnormalize_fn_bg(TAC_BG_FP, TAC_MEAN_BG, TAC_STD_BG)


def load_vision_data(
    path: str,
    rgb_size: list = [224, 224],
    im_scale_range: list = [0.12, 0.18],
    transform_rgb=RGB_PREPROCESS,
    dataset_version: str = "v1",
    randomize_crop: bool = False,
    randomize_range: float = 0.05,
    device: str = None,
):
    assert dataset_version in ["v1", "v2"]
    rgb = Image.open(path)
    if rgb.mode != "RGB":
        rgb = rgb.convert("RGB")
    rgb_w, rgb_h = rgb.size

    if dataset_version == "v1":
        rgb = TF.center_crop(rgb, np.ceil(np.sqrt(2) * im_scale_range[1] * max(rgb_h, rgb_w)))
    elif dataset_version == "v2":
        crop_height = int(np.ceil(np.sqrt(2) * im_scale_range[1] * max(rgb_h, rgb_w)))
        crop_width = crop_height

        if randomize_crop:
            top_random_range = int(randomize_range * crop_height)
            left_random_range = int(randomize_range * crop_width)
            start_pos = 0 if "data3" not in path else 200
            top = np.random.randint(start_pos, top_random_range + start_pos)
            left = np.random.randint(-left_random_range // 2, left_random_range // 2) + (rgb_w - crop_width) // 2
        else:
            top = 0 if "data3" not in path else 200
            left = (rgb_w - crop_width) // 2
        rgb = TF.crop(rgb, top, left, crop_height, crop_width)
    if transform_rgb is not None:
        rgb = transform_rgb(rgb)
    if device is not None:
        rgb = rgb.to(device)
    return rgb


def load_tactile_data(
    path: str,
    transform_tac=TAC_PREPROCESS,
    device: str = None,
):
    tac = Image.open(path)
    tac = tac_padding(tac)
    if transform_tac is not None:
        tac = transform_tac(tac)
    if device is not None:
        tac = tac.to(device)
    return tac

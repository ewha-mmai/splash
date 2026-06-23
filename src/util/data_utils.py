import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from src.util import tactile_preprocess as tacvis

BICUBIC = InterpolationMode.BICUBIC

vision_transform = transforms.Compose([

    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC),
    
])

TACTILE_TOKEN_LEN = 196

def load_vision_image(path: str) -> Image.Image:
    """
    Loads an image from the given path and converts it to RGB.
    If the path does not exist, returns a black image (336x336).
    """
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (224, 224), (0, 0, 0))

    
    return img

def load_tactile_data(path: str, augment: bool = False):
    """
    Loads tactile data from a .pt or image file.
    
    Args:
        path (str): Path to the tactile data file.
        augment (bool): If True, applies data augmentation (TAC_AUGMENTS).
                        If False, applies standard processing (TAC_WBG).
    
    Returns:
        tuple: (tactile_pixel_values, tactile_grid_thw)
    """
    tactile_pixel_values = None
    tactile_grid_thw = torch.tensor([1, 14, 14], dtype=torch.long)
    
    transform_tac = tacvis.TAC_AUGMENTS if augment else tacvis.TAC_WBG

    if path and os.path.exists(path):
        tactile_pixel_values = tacvis.load_tactile_data(
            path, transform_tac=transform_tac
        )
        
    return tactile_pixel_values, tactile_grid_thw

def inject_tactile_tokens(
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor, 
    tokenizer
):
    """
    Injects tactile tokens into the input sequence.
    Sequence: <|tactile_start|> + <|tactile_pad|> * 196 + <|tactile_end|>
    Location: Immediately before the <|vision_start|> token.

    Args:
        input_ids (torch.Tensor): 1D Tensor of input IDs.
        attention_mask (torch.Tensor): 1D Tensor of attention mask.
        tokenizer: The tokenizer object used to retrieve special token IDs.
    
    Returns:
        tuple: (new_input_ids, new_attention_mask)
    """
    
    try:
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        tactile_pad_id = tokenizer.convert_tokens_to_ids("<|tactile_pad|>")
        tactile_start_id = tokenizer.convert_tokens_to_ids("<|tactile_start|>")
        tactile_end_id = tokenizer.convert_tokens_to_ids("<|tactile_end|>")
    except AttributeError:
        print("[Error] Tokenizer definition missing for tactile tokens.")
        return input_ids, attention_mask

    tac_tokens = [tactile_start_id] + [tactile_pad_id] * TACTILE_TOKEN_LEN + [tactile_end_id]
    
    tac_tensor = torch.tensor(tac_tokens, dtype=input_ids.dtype).to(input_ids.device)
    tac_mask = torch.ones_like(tac_tensor).to(attention_mask.device)

    matches = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    
    if len(matches) > 0:
        insert_idx = matches[0].item()
        
        new_input_ids = torch.cat([
            input_ids[:insert_idx],
            tac_tensor,
            input_ids[insert_idx:]
        ], dim=0)
        
        new_attention_mask = torch.cat([
            attention_mask[:insert_idx],
            tac_mask,
            attention_mask[insert_idx:]
        ], dim=0)
    else:
        new_input_ids = torch.cat([tac_tensor, input_ids], dim=0)
        new_attention_mask = torch.cat([tac_mask, attention_mask], dim=0)
    
    return new_input_ids, new_attention_mask
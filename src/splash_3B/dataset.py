import torch
import yaml
import os
import json
import copy
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from src.util.data_utils import load_vision_image, load_tactile_data, inject_tactile_tokens
from tvl.tvl_enc import tacvis

logger = logging.getLogger(__name__)


class TactileBaseDataset(Dataset):
    def __init__(
        self,
        config_path: str,
        qwen_path: str,
        processor=None,
        tokenizer=None,
        max_words: int = 512,
        augment_tactile: bool = False,
    ):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(qwen_path, use_fast=True)
        self.processor = processor or AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer
        self.processor.image_processor.min_pixels = 224 * 224
        self.processor.image_processor.max_pixels = 224 * 224
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")

        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        self.tactile_pad_id = self.tokenizer.convert_tokens_to_ids("<|tactile_pad|>")
        self.tactile_start_id = self.tokenizer.convert_tokens_to_ids("<|tactile_start|>")
        self.tactile_end_id = self.tokenizer.convert_tokens_to_ids("<|tactile_end|>")
        self.tactile_token_len = 196
        self.ignore_index = -100

        self.augment_tactile = augment_tactile
        self.tactile_process = tacvis.TAC_AUGMENTS if augment_tactile else tacvis.TAC_WBG
        self.max_words = max_words
        self.data = [] 

        print(f"[Dataset] Loading config: {config_path}")
        with open(config_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self):
        return len(self.data)
        

class PretrainDataset(TactileBaseDataset):
    def __init__(self, config_path, qwen_path, processor=None, tokenizer=None, **kwargs):
        super().__init__(config_path, qwen_path, processor, tokenizer, **kwargs)
        self._load_data()

    def _load_data(self):
        if "META" in self.cfg:
            if "train" in self.cfg and "META" in self.cfg["train"]:
                meta_list = self.cfg["train"]["META"]
            elif isinstance(self.cfg["META"], list):
                meta_list = self.cfg["META"]
            else:
                 meta_list = []
        elif "train" in self.cfg:
             meta_list = self.cfg["train"]["META"]
        else:
            meta_list = [self.cfg.get("dataset", {}).get("meta_path", "")]

        for meta_path in meta_list:
            if not os.path.exists(meta_path): continue
            folder = os.path.dirname(meta_path)

            if meta_path.endswith('.json'):
                with open(meta_path, 'r') as f:
                    raw_data = json.load(f)
                
                for item in raw_data:
                    img_path = item.get("url") or item.get("image")
                    tac_path = item.get("tactile")
                    caption = item.get("caption") or item.get("text")

                    vid_id = "unknown"
                    if item.get("video_id"):
                        vid_id = item.get("video_id")
                    elif tac_path:
                        top_folder = tac_path.split('/')[0] 
                        if '-' in top_folder:
                            vid_id = top_folder.split('-')[0]
                        else:
                            vid_id = top_folder
                    
                    self.data.append({
                        "image": os.path.join(folder, img_path) if img_path else None,
                        "tactile": os.path.join(folder, tac_path) if tac_path else None,
                        "caption": str(caption),
                        "video_id": vid_id
                    })
            
            else:
                df = pd.read_csv(meta_path)
                for _, row in df.iterrows():
                    vid_id = "unknown"
                    if "label" in row: vid_id = str(row["label"])
                    elif "material" in row: vid_id = str(row["material"])
                    elif "class" in row: vid_id = str(row["class"])

                    if vid_id == "unknown" and "url" in row:
                        try:
                            filename = os.path.basename(row["url"])
                            parts = filename.split('_')
                            if len(parts) >= 2: vid_id = parts[1]
                            else: vid_id = filename
                        except: pass

                    self.data.append({
                        "image": os.path.join(folder, row["url"]),
                        "tactile": os.path.join(folder, row["tactile"]) if "tactile" in row and not pd.isna(row["tactile"]) else None,
                        "caption": str(row["caption"]),
                        "video_id": vid_id
                    })
        print(f"[Pretrain] Loaded {len(self.data)} samples.")

    def __getitem__(self, idx):
        item = self.data[idx]
        image = load_vision_image(item["image"])
        caption = item["caption"]

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the tactile sensation."}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        ]
        
        text_full = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(images=image, text=[text_full], return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5:
             pixel_values = pixel_values.squeeze(0)
        elif pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)

        image_grid_thw = inputs["image_grid_thw"].squeeze(0)

        pixel_values_tactile, tactile_grid_thw = load_tactile_data(
            item["tactile"], 
            augment=(self.tactile_process == tacvis.TAC_AUGMENTS)
        )
        input_ids, attention_mask = inject_tactile_tokens(input_ids, attention_mask, self.tokenizer)

        labels = input_ids.clone()
        
        start_indices = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]

        if len(start_indices) > 0:
            last_im_start = start_indices[-1].item()
            
            mask_len = last_im_start + 3
            
            if mask_len < len(labels):
                labels[:mask_len] = self.ignore_index
            else:
                labels[:] = self.ignore_index
        else:
            labels[:] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_tactile": pixel_values_tactile,
            "tactile_grid_thw": tactile_grid_thw,
        }


class FinetuneDataset(TactileBaseDataset):
    
    _warning_count = 0
    _max_warnings = 10
    
    def __init__(self, config_path, qwen_path, processor=None, tokenizer=None, **kwargs):
        super().__init__(config_path, qwen_path, processor, tokenizer, **kwargs)
        self._load_metadata()

    def _extract_video_id(self, item: Dict) -> str:
        vid_id = item.get("video_id") or item.get("video")

        if not vid_id or vid_id == "unknown":
            tac_path = item.get("tactile", "")
            if tac_path:
                folder_name = os.path.basename(os.path.dirname(tac_path)) 
                if '-' in folder_name:
                    vid_id = folder_name.split('-')[0]
                else:
                    vid_id = folder_name
        
        if (not vid_id or vid_id == "unknown") and item.get("image"):
            img_path = item.get("image", "")
            vid_id = os.path.splitext(os.path.basename(img_path))[0]

        return vid_id if vid_id else "unknown"

    def _validate_conversations(self, conversations: List[Dict]) -> bool:
        """Assistant  """
        for turn in conversations:
            role = turn.get("from", turn.get("role"))
            content = turn.get("value", turn.get("content", ""))
            
            if role in ["gpt", "assistant"] and content.replace("<image>", "").strip():
                return True
        return False

    def _load_metadata(self):
        self.data = []

        """ JSON """
        meta_list = []
        if "META" in self.cfg:
            meta_list = self.cfg["META"] if isinstance(self.cfg["META"], list) else [self.cfg["META"]]
        elif "train" in self.cfg and "META" in self.cfg["train"]:
            meta_list = self.cfg["train"]["META"]
        
        logger.info(f"[Finetune] Found {len(meta_list)} metadata sources")

        total_loaded = 0
        total_skipped = 0

        for meta_path in meta_list:
            if not os.path.exists(meta_path): 
                logger.warning(f"File not found: {meta_path}")
                continue
            
            folder = os.path.dirname(meta_path)
            try:
                with open(meta_path, 'r') as f:
                    raw_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {meta_path}: {e}")
                continue

            count = 0
            skipped = 0

            for item in raw_data:
                if "conversations" not in item: 
                    continue
                
                if not self._validate_conversations(item["conversations"]):
                    skipped += 1
                    continue
                
                image = os.path.join(folder, item.get("image", ""))
                tactile = os.path.join(folder, item.get("tactile", ""))
                
                vid_id = self._extract_video_id(item)

                user_input = "Describe this."
                assistant_output = "I cannot describe it."
                
                for turn in item["conversations"]:
                    role = turn.get("from", turn.get("role"))
                    val = turn.get("value", turn.get("content", "")).replace("<image>", "").strip()
                    if role in ["human", "user"]:
                        user_input = val
                    elif role in ["gpt", "assistant"]:
                        assistant_output = val

                self.data.append({
                    "image": image,
                    "tactile": tactile,
                    "user_input": user_input,
                    "assistant_output": assistant_output,
                    "video_id": vid_id
                })
                count += 1
            
            logger.info(f"Loaded {count} samples from {os.path.basename(meta_path)} (skipped {skipped})")
            total_loaded += count
            total_skipped += skipped
        
        logger.info(f"[Finetune] Total: {total_loaded} samples loaded, {total_skipped} skipped")

    def __getitem__(self, idx):

        item = self.data[idx]

        image = load_vision_image(item["image"])
        pixel_values_tactile, tactile_grid_thw = load_tactile_data(
            item["tactile"], 
            augment=(self.tactile_process == tacvis.TAC_AUGMENTS)
        )

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["user_input"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": item["assistant_output"]}]}
        ]
        
        text_full = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(images=image, text=[text_full], return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        elif pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)

        image_grid_thw = inputs["image_grid_thw"].squeeze(0)

        input_ids, attention_mask = inject_tactile_tokens(input_ids, attention_mask, self.tokenizer)

        labels = input_ids.clone()
        
        start_indices = (input_ids == self.im_start_id).nonzero(as_tuple=True)[0]

        if len(start_indices) > 0:
            last_im_start = start_indices[-1].item()
            
            mask_len = last_im_start + 3
            
            if mask_len < len(labels):
                labels[:mask_len] = self.ignore_index
            else:
                labels[:] = self.ignore_index
        else:
            labels[:] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_tactile": pixel_values_tactile,
            "tactile_grid_thw": tactile_grid_thw,
        }


class DataCollatorForTactileDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [i["input_ids"] for i in instances], 
            batch_first=True, 
            padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [i["labels"] for i in instances], 
            batch_first=True, 
            padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [i["attention_mask"] for i in instances], 
            batch_first=True, 
            padding_value=0
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if instances[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat([i["pixel_values"] for i in instances], dim=0)
            batch["image_grid_thw"] = torch.stack([i["image_grid_thw"] for i in instances], dim=0)

        valid_sample = next((x for x in instances if x["pixel_values_tactile"] is not None), None)
        
        if valid_sample is not None:
            tactile_vals = []
            tactile_grids = []
            zero_tac = torch.zeros_like(valid_sample["pixel_values_tactile"])
            
            for i in instances:
                if i["pixel_values_tactile"] is None:
                    tactile_vals.append(zero_tac)
                    tactile_grids.append(valid_sample["tactile_grid_thw"])
                else:
                    tactile_vals.append(i["pixel_values_tactile"])
                    tactile_grids.append(i["tactile_grid_thw"])
            
            batch["pixel_values_tactile"] = torch.stack(tactile_vals)
            batch["tactile_grid_thw"] = torch.stack(tactile_grids)
        else:
            batch["pixel_values_tactile"] = torch.zeros(
                len(instances), 3, 224, 224, 
                dtype=instances[0]["pixel_values"].dtype
            )
            batch["tactile_grid_thw"] = torch.tensor([[1, 14, 14]] * len(instances), dtype=torch.long)

        return batch
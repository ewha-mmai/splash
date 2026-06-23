import torch
import yaml
import os
import json
import logging
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
from src.util.data_utils import load_vision_image, load_tactile_data
from src.util import tactile_preprocess as tacvis

logger = logging.getLogger(__name__)

def safe_load_tokenizer(qwen_path, passed_tokenizer=None):
    if passed_tokenizer is not None:
        return passed_tokenizer

    print(" Loading tokenizer (InternVL 2.5 safe version)...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            qwen_path, 
            trust_remote_code=True, 
            use_fast=False,
            fix_mistral_regex=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            qwen_path, 
            trust_remote_code=True, 
            use_fast=False
        )
    
    tactile_tokens = ["<tac>", "</tac>", "<TAC_CONTEXT>"]
    tokenizer.add_tokens(tactile_tokens, special_tokens=True)
    original_tokenize = tokenizer.tokenize

    def safe_tokenize(text, **kwargs):
        special_tokens = [
            "<TAC_CONTEXT>", "<tac>", "</tac>",
            "<IMG_CONTEXT>", "<image>", "</image>",
            "<tactile>", "</tactile>", "<|im_start|>", "<|im_end|>"
        ]
        for t in special_tokens:
            text = text.replace(t, f" {t} ")
        return original_tokenize(text, **kwargs)

    tokenizer.tokenize = safe_tokenize

    if getattr(tokenizer, "pad_token", None) is None:
        vocab = tokenizer.get_vocab()
        if "<|extra_0|>" in vocab:
            tokenizer.pad_token = "<|extra_0|>"
        elif "<unk>" in vocab:
            tokenizer.pad_token = "<unk>"
        else:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", "</s>")
            
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    
    return tokenizer

class TactileBaseDataset(Dataset):
    def __init__(self, config_path, qwen_path, tokenizer=None, max_words=512, augment_tactile=False):
        self.tokenizer = safe_load_tokenizer(qwen_path, passed_tokenizer=tokenizer)
        
        print(" Loading Image Processor (Forcing 448x448 for InternVL 2.5)...")
        self.vision_image_processor = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.tactile_image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

        class _ProcWrapper:
            def __init__(self, tokenizer, vision_processor, tactile_processor=None):
                self.tokenizer = tokenizer
                self.vision_processor = vision_processor
                self.tactile_processor = tactile_processor

            def __call__(self, text=None, images=None, tactile_image=None, return_tensors=None, padding=None):
                model_inputs = self.tokenizer(text, return_tensors="pt", padding=padding)

                if images is not None:
                    if isinstance(images, list):
                        img_list = [self.vision_processor(img) for img in images]
                        model_inputs["pixel_values"] = torch.stack(img_list, dim=0)
                    else:
                        model_inputs["pixel_values"] = self.vision_processor(images).unsqueeze(0)

                if tactile_image is not None and self.tactile_processor is not None:
                    if isinstance(tactile_image, list):
                        tac_list = [self.tactile_processor(img) for img in tactile_image]
                        model_inputs["pixel_values_tactile"] = torch.stack(tac_list, dim=0)
                    else:
                        model_inputs["pixel_values_tactile"] = self.tactile_processor(tactile_image).unsqueeze(0)

                return model_inputs

        self.processor = _ProcWrapper(
            tokenizer=self.tokenizer,
            vision_processor=self.vision_image_processor,
            tactile_processor=self.tactile_image_processor 
        )
                
        self.augment_tactile = augment_tactile
        self.tactile_process = tacvis.TAC_AUGMENTS if augment_tactile else tacvis.TAC_WBG
        self.max_words = max_words
        self.data = []

        with open(config_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self):
        return len(self.data)


class FinetuneDataset(TactileBaseDataset):
    _warning_count = 0
    _max_warnings = 10
    
    def __init__(self, config_path, qwen_path, tokenizer=None, max_words=512, augment_tactile=False):
        super().__init__(
            config_path=config_path,
            qwen_path=qwen_path,
            tokenizer=tokenizer,
            max_words=max_words,
            augment_tactile=augment_tactile
        )
        self._load_metadata()

    def _extract_video_id(self, item: Dict) -> str:
        vid_id = item.get("video_id") or item.get("video")
        if not vid_id or vid_id == "unknown":
            tac_path = item.get("tactile", "")
            if tac_path:
                folder_name = os.path.basename(os.path.dirname(tac_path)) 
                vid_id = folder_name.split('-')[0] if '-' in folder_name else folder_name
        
        if (not vid_id or vid_id == "unknown") and item.get("image"):
            vid_id = os.path.splitext(os.path.basename(item.get("image", "")))[0]

        return vid_id if vid_id else "unknown"

    def _validate_conversations(self, conversations: List[Dict]) -> bool:
        for turn in conversations:
            role = turn.get("from", turn.get("role"))
            content = turn.get("value", turn.get("content", ""))
            if role in ["gpt", "assistant"] and content.replace("<image>", "").strip():
                return True
        return False

    def _load_metadata(self):
        self.data = []
        meta_list = self.cfg.get("META", [])
        if not meta_list and "train" in self.cfg:
            meta_list = self.cfg["train"].get("META", [])
        if not isinstance(meta_list, list):
            meta_list = [meta_list]

        for meta_path in meta_list:
            if not os.path.exists(meta_path): continue
            folder = os.path.dirname(meta_path)
            try:
                with open(meta_path, 'r') as f:
                    raw_data = json.load(f)
            except Exception:
                continue

            for item in raw_data:
                if "conversations" not in item or not self._validate_conversations(item["conversations"]):
                    continue
                
                user_input, assistant_output = "Describe this.", "I cannot describe it."
                for turn in item["conversations"]:
                    role = turn.get("from", turn.get("role"))
                    val = turn.get("value", turn.get("content", "")).replace("<image>", "").strip()
                    if role in ["human", "user"]: user_input = val
                    elif role in ["gpt", "assistant"]: assistant_output = val

                self.data.append({
                    "image": os.path.join(folder, item.get("image", "")),
                    "tactile": os.path.join(folder, item.get("tactile", "")),
                    "user_input": user_input,
                    "assistant_output": assistant_output,
                    "video_id": self._extract_video_id(item)
                })

    def __getitem__(self, idx):
        item = self.data[idx]
        image = load_vision_image(item["image"])

        user_text = f"<image>\n<tactile>\n{item.get('user_input', '')}"
        assistant_text = item.get("assistant_output", "")

        chat_full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}],
            tokenize=False, add_generation_prompt=False
        )

        text_user_only = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True
        )

        pixel_values_tactile, tactile_grid_thw, num_tac = None, None, 0
        if item.get("tactile"):
            pixel_values_tactile, tactile_grid_thw = load_tactile_data(
                item["tactile"], augment=(self.tactile_process == tacvis.TAC_AUGMENTS)
            )
            try:
                tg_list = tactile_grid_thw.tolist() if isinstance(tactile_grid_thw, torch.Tensor) else tactile_grid_thw
                if len(tg_list) >= 2:
                    num_tac = int(tg_list[-2] * tg_list[-1])
            except:
                num_tac = 0
            max_tactile_tokens = 1024
            num_tac = min(num_tac, max_tactile_tokens)

        tac_start, tac_end, tac_context_token = "<tac>", "</tac>", "<TAC_CONTEXT>"
        if num_tac > 0:
            tac_tokens_str = tac_start + " " + " ".join([tac_context_token]*num_tac) + " " + tac_end
            chat_full = chat_full.replace("<tactile>", tac_tokens_str, 1)
            text_user_only = text_user_only.replace("<tactile>", tac_tokens_str, 1)
        else:
            chat_full = chat_full.replace("<tactile>\n", "").replace("<tactile>", "")
            text_user_only = text_user_only.replace("<tactile>\n", "").replace("<tactile>", "")

        num_img_tokens = 256
        if "<image>" in chat_full:
            img_tokens = "<IMG_CONTEXT>" * num_img_tokens
            chat_full = chat_full.replace("<image>", img_tokens, 1)
            text_user_only = text_user_only.replace("<image>", img_tokens, 1)

        model_inputs = self.processor(
            text=chat_full,
            images=image,
            tactile_image=pixel_values_tactile,
            return_tensors="pt",
            padding=False
        )
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)

        pixel_values = model_inputs["pixel_values"]
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        elif pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        image_grid_thw = model_inputs.get("image_grid_thw", None)
        if image_grid_thw is not None and image_grid_thw.dim() == 4:
            image_grid_thw = image_grid_thw.squeeze(0)

        user_inputs = self.tokenizer(text_user_only, add_special_tokens=False)
        user_len = len(user_inputs["input_ids"])
        labels = input_ids.clone()
        labels[:user_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_tactile": pixel_values_tactile,
            "tactile_grid_thw": tactile_grid_thw,
            "num_tactile_tokens": num_tac,
        }

class DataCollatorForTactileDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence([i["input_ids"] for i in instances], batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([i["labels"] for i in instances], batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence([i["attention_mask"] for i in instances], batch_first=True, padding_value=0)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if instances[0].get("pixel_values") is not None:
            batch["pixel_values"] = torch.cat([i["pixel_values"] for i in instances], dim=0)
            if instances[0].get("image_grid_thw") is not None:
                batch["image_grid_thw"] = torch.stack([i["image_grid_thw"] for i in instances], dim=0)

        valid_sample = next((x for x in instances if x["pixel_values_tactile"] is not None), None)
        if valid_sample is not None:
            tactile_vals, tactile_grids = [], []
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
            if instances[0].get("pixel_values") is not None:
                dtype = instances[0]["pixel_values"].dtype
            else:
                dtype = torch.float32
            batch["pixel_values_tactile"] = torch.zeros(len(instances), 3, 224, 224, dtype=dtype)
            batch["tactile_grid_thw"] = torch.tensor([[1, 14, 14]] * len(instances), dtype=torch.long)

        return batch
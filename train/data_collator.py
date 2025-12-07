"""
Custom data collator for Qwen3-VL grasp training.
Fixes Loss=0.0 by accurately calculating dynamic image token lengths.
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

@dataclass
class GraspDataCollator:
    processor: Any 
    tokenizer: PreTrainedTokenizer = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.processor.tokenizer
        
        # CRITICAL FIX: Force right-padding. 
        # If this is 'left', our masking indices will be wrong.
        self.tokenizer.padding_side = 'right'

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prompt_texts = []
        images = []
        
        for f in features:
            messages = f['messages']
            
            # 1. Extract Image
            user_content = messages[1]['content']
            img_obj = None
            for item in user_content:
                if item['type'] == 'image':
                    img_obj = item['image']
                    break
            if img_obj is None:
                raise ValueError("No image found!")
            images.append(img_obj)
            
            # 2. Get FULL text (User + Assistant)
            full_str = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            full_texts.append(full_str)

            # 3. Get PROMPT text (User Only)
            user_messages = messages[:-1]
            prompt_str = self.processor.apply_chat_template(
                user_messages, 
                tokenize=False, 
                add_generation_prompt=True 
            )
            prompt_texts.append(prompt_str)

        # 4. Process FULL Batch (This is what we train on)
        # padding=True will now use 'right' padding because of __post_init__
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors='pt',
        )

        # 5. Process PROMPT Batch (Just to measure lengths)
        # padding=False + return_tensors=None gives us lists of ids
        # We need this to get exact length of the prompt part per sample
        with torch.no_grad():
            prompt_batch = self.processor(
                text=prompt_texts,
                images=images,
                padding=False, 
                return_tensors=None # Returns python lists, avoids "expected sequence of length..." error
            )
            
        # 6. Create Labels
        labels = batch['input_ids'].clone()
        
        # Mask padding (standard)
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask User Prompts (The "Lens Method")
        for i in range(len(labels)):
            # Get exact length of the prompt (including 1000+ image tokens)
            if isinstance(prompt_batch['input_ids'], list):
                p_len = len(prompt_batch['input_ids'][i])
            else:
                p_len = prompt_batch['input_ids'][i].shape[0]
            
            # Mask the prompt part. The rest (the answer) remains unmasked.
            labels[i, :p_len] = -100

        batch['labels'] = labels
        return batch

def compute_grasp_metrics(eval_pred):
    return {}
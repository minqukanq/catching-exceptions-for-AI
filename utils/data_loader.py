import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, X, y, code_length, tokenizer):
        self.tokenizer = tokenizer
        self.X = X
        self.y = y
        
        self.data = []
        for code_token in X:
            code_ids = tokenizer.encode_plus(
              code_token,
              add_special_tokens=True,
              truncation=True,
              max_length=code_length,
              return_token_type_ids=False,
              padding="max_length",
              return_attention_mask=True,
              return_tensors='pt',
            )
            self.data.append(code_ids)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return dict(
            input_ids=self.data[idx]['input_ids'].flatten(),
            attention_mask=self.data[idx]["attention_mask"].flatten(),
            labels=torch.FloatTensor(self.y[idx])
        )
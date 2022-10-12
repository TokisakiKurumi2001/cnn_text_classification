from cnn_text_classification import dataset, tokenizer
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch

def collate_fn(examples):
    if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
        encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
    input = encoded_inputs['ba']
    tok = tokenizer.tokenize(input, padding='max_length', truncation=True, max_length=70, return_tensors='pt')
    
    labels = encoded_inputs['label']
    tok_labels = torch.tensor(labels)
    
    tok['labels'] = tok_labels
    return tok

split_dataset = dataset.class_encode_column('label')
split_dataset = split_dataset.train_test_split(test_size=0.3, stratify_by_column='label', seed=42)
test_valid_dataset = split_dataset.pop('test')
test_valid_dataset = test_valid_dataset.train_test_split(test_size=0.5, stratify_by_column='label', seed=42)
split_dataset['valid'] = test_valid_dataset.pop('train')
split_dataset['test'] = test_valid_dataset.pop('test')
train_dataloader = DataLoader(split_dataset['train'], batch_size=32, collate_fn=collate_fn, num_workers=24)
valid_dataloader = DataLoader(split_dataset['valid'], batch_size=32, collate_fn=collate_fn, num_workers=24)
test_dataloader = DataLoader(split_dataset['test'], batch_size=32, collate_fn=collate_fn, num_workers=24)
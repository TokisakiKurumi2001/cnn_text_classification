import torch
import torch.nn as nn
from cnn_text_classification import tokenizer
from typing import List
from functools import reduce

class CNNForSeqClassifier(nn.Module):
    def __init__(
        self, num_classes: int, embed_dim: int, kernel_sizes: List[int],
        num_channels: List[int], dropout: float
    ):
        super(CNNForSeqClassifier, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.word_embed = nn.Embedding(self.vocab_size, embed_dim)

        # Conv1D
        self.conv1d_layer = nn.ModuleList()
        for kernel_size, channel in zip(kernel_sizes, num_channels):
            self.conv1d_layer.append(
                nn.Conv1d(embed_dim, channel, kernel_size)
            )

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

        # projection to classification
        hidden_dim = reduce(lambda x, y: x + y, num_channels)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs):
        input_ids = inputs.input_ids
        we_output = self.word_embed(input_ids) # (B, S, E)
        we_output = we_output.permute(0, 2, 1) # (B, E, S)

        # after layer->pool->relu -> (B, Channel, 1)
        # squeeze -> (B, Channel)
        conv1d_res = []
        for layer in self.conv1d_layer:
            conv1d_res.append(
                torch.squeeze(self.relu(self.pool(layer(we_output))), dim=-1)
            )
        merge = torch.cat(conv1d_res, dim=1)
        output = self.tanh(self.dropout(self.proj(merge)))
        cls_output = self.out(output)
        return cls_output

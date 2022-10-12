import torch
import torch.nn as nn
from cnn_text_classification import tokenizer

class CNNForSeqClassifier(nn.Module):
    def __init__(
        self, num_classes
    ):
        super(CNNForSeqClassifier, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.word_embed = nn.Embedding(self.vocab_size, embed_dim)

        # projection to classification
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs):
        return None

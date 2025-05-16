import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        self.transformer = Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        captions_embed = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :]
        outputs = self.transformer(features.unsqueeze(0), captions_embed.permute(1, 0, 2))
        outputs = self.fc(outputs.permute(1, 0, 2))
        return outputs

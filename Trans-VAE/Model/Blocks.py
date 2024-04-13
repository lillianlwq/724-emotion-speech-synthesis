##input embedding

import torch
import torch.nn as nn
import math

from torchaudio.models.wav2vec2.components import SelfAttention

class InputEmbedding(nn.Module):
    def __init__(self, d_model, n_vocab):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = n_vocab
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        x = x.long()
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.seq_len = seq_len

        # create a matrix of shape
        pe = torch.zeros(self.seq_len, self.d_model)

        ##pe(pos, 2i) = sin(pos/(10000^d_model(2i)))
        ##pe(pos, 2i+1) = cos(pos/(10000^d_model(2i)))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # creating tensor
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to the positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to the positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, Seq_len, D_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardBlock, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) -> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % self.h == 0
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // self.h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.size(-1)
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # @ is matrix multiplication
        # apply the softmax
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -float('inf'))
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores  # second attention score is for visualization

    def forward(self, query, key, value, mask=None):
        query_ = self.w_q(query)
        key_ = self.w_k(key)
        value_ = self.w_v(value)

        query_ = query_.view(query_.shape[0], query_.shape[1], self.h, self.d_k).transpose(1, 2)
        key_ = key_.view(key_.shape[0], key_.shape[1], self.h, self.d_k).transpose(1, 2)
        value_ = value_.view(value_.shape[0], value_.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query_, key_, value_, mask, self.dropout)

        # (batch, h, seq_len, d_k) =>(batch, seq_len, h, d_k)->(Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model)
        return self.w_o(x)


class ResidualBlock(nn.Module):
    def __init__(self, dropout):
        super(ResidualBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList(
            [ResidualBlock(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual[1](x, lambda x: self.feed_forward)
        return x + self


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention: MultiHeadAttentionBlock,
                 cross_attention: MultiHeadAttentionBlock,
                 feed_forward: FeedForwardBlock,
                 dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList(ResidualBlock(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual[2](x, lambda x: self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch Seq_Len, d_model) -> (Batch, Seq_Len, Vocab Size)
        return torch.log_softmax(self.projection(x), dim=-1)

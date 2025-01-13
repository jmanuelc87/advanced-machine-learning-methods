import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq_len, dim_model)
        
        # Create a vector of shape (seq)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)) # (d_model / 2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('positional_encoding_table', pe)

    def forward(self, x):
        x = x + self.positional_encoding_table[:, :x.shape[1], :] # (batch, seq, d_model)
        return self.dropout(x)


class FeedForward(nn.Module):
    
    def __init__(self, dim_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_model, num_heads, dropout, masked=False):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.masked = masked
        assert self.dim_model % self.num_heads == 0, "dim_model is not divisible by num_heads"
        
        self.d_k = self.dim_model // self.num_heads
        self.w_q = nn.Linear(dim_model, dim_model, bias=False)
        self.w_k = nn.Linear(dim_model, dim_model, bias=False)
        self.w_v = nn.Linear(dim_model, dim_model, bias=False)
        self.w_o = nn.Linear(dim_model, dim_model, bias=False)
        self.dropout_a = nn.Dropout(p=dropout)
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(5000, 5000, dtype=torch.bool)).unsqueeze(0).unsqueeze(0))
    
    def scaled_dot_product_attention(self, q, k, v):
        # calculate the attention scores
        attn = q @ k.transpose(-2, -1) * (self.d_k ** -0.5)
        
        if self.masked:
            mask = ~self.causal_mask[:, :, :q.size(2), :q.size(2)]
            attn = torch.masked_fill(attn, mask, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        
        attn = self.dropout_a(attn)
        
        out = attn @ v
        
        return out, attn

    def forward(self, q, k, v):
        # Compute Linear
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # reshape
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)
        
        # compute attention
        x, _ = self.scaled_dot_product_attention(q, k, v)
        
        # combine
        x = self.combine(x)
        
        # Compute final Linear
        x = self.w_o(x)
        
        return x
        
    def split(self, tensor: torch.Tensor):
        # (batch, length, d_model) --> (batch, length, num_heads, d_k) --> (batch, num_heads, length, d_k)
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.d_k).transpose(1,2)
    
    def combine(self, tensor: torch.Tensor):
        # (batch, num_heads, length, d_k) --> (batch, length, num_heads, d_k) --> (batch, length, dim_model)
        batch, num_heads, length, d_k = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, length, self.num_heads * self.d_k)


class EncoderBlock(nn.Module):

    def __init__(self, dim_model, num_heads, hidden_dim, dropout):
        super().__init__()
        self.head_attn = MultiHeadAttention(dim_model, num_heads, dropout)
        self.ffn = FeedForward(dim_model, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout_a = nn.Dropout(p=dropout)
        self.dropout_b = nn.Dropout(p=dropout)

    def forward(self, x):
        sub1 = lambda a: self.head_attn(a,a,a)
        # perform the self-attention, linear, layer norm, and skip connections
        x = x + self.dropout_a(sub1(self.norm1(x)))
        x = x + self.dropout_b(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    
    def __init__(self, dim_model, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return self.norm(x)


class DecoderBlock(nn.Module):
    
    def __init__(self, dim_model, num_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attention_block = MultiHeadAttention(dim_model, num_heads, dropout, masked=True)
        self.cross_attention_block = MultiHeadAttention(dim_model, num_heads, dropout)
        self.ffn = FeedForward(dim_model, hidden_dim, dropout)
        
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout_a = nn.Dropout(p=dropout)
        self.dropout_b = nn.Dropout(p=dropout)
        self.dropout_c = nn.Dropout(p=dropout)
    
    def forward(self, x, encoder_output):
        sub1 = lambda a: self.self_attention_block(a,a,a)
        sub2 = lambda a: self.cross_attention_block(a, encoder_output, encoder_output)
        
        x = x + self.dropout_a(sub1(self.norm1(x)))
        x = x + self.dropout_b(sub2(self.norm2(x)))
        x = x + self.dropout_c(self.ffn(self.norm3(x)))
        
        return x


class Decoder(nn.Module):
    
    def __init__(self, dim_model, num_heads, hidden_dim, num_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(dim_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)
    
    def forward(self, x, encoder_output):
        for layer in self.blocks:
            x = layer(x, encoder_output)
        return self.norm(x)


class Seq2SeqTranslatorTransformer(nn.Module):
    
    def __init__(self, dim_model, num_heads, hidden_dim, max_len, enc_num_layers, dec_num_layers, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()
        self.encoder = Encoder(dim_model, num_heads, hidden_dim, enc_num_layers, dropout)
        self.decoder = Decoder(dim_model, num_heads, hidden_dim, dec_num_layers, dropout)

        self.src_emb = nn.Embedding(src_vocab_size, dim_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dim_model)

        self.src_pos = PositionalEncoding(dim_model, max_len)
        self.tgt_pos = PositionalEncoding(dim_model, max_len)

        self.proj = nn.Linear(dim_model, tgt_vocab_size)

        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src)
    
    def decode(self, encoder_output, tgt):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output)

    def forward(self, x, target):
        enc_out = self.encode(x)
        dec_out = self.decode(enc_out, target)
        out = self.proj(dec_out)
        return out
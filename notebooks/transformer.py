import torch
import torch.nn as nn
import torch.nn.functional as F




class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len, vocab_size, padding_idx):
        super().__init__()

        # same size with input matrix
        self.register_buffer('encoding_table', torch.zeros(max_seq_len, d_model))

        # generate the positions for each embedding vector
        self.register_buffer("pos", torch.arange(0, max_seq_len).float().unsqueeze(dim=1))

        # generate _2i term
        self.register_buffer("_2i", torch.arange(0, d_model, step=2).float())

        # Create the encoding matrix
        self.encoding_table[:, 0::2] = torch.sin(self.pos / (10000 ** (self._2i / d_model)))
        self.encoding_table[:, 1::2] = torch.cos(self.pos / (10000 ** (self._2i / d_model)))
        
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx)

    def forward(self, x):
        batch_size, seq_len = x.shape
        emb = self.tok_emb(x)
        pos = self.encoding_table[:seq_len, :]
        return emb + pos


class HeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, max_seq_len, dropout, mask=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.mask = mask
        
        self.q_attn = nn.Linear(d_model, d_model, bias=False)
        self.k_attn = nn.Linear(d_model, d_model, bias=False)
        self.v_attn = nn.Linear(d_model, d_model, bias=False)
        
        self.d_proj = nn.Linear(d_model, d_model, bias=False)
        self.a_dropout = nn.Dropout(p=dropout)
        self.b_dropout = nn.Dropout(p=dropout)
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, q, k, v):
        batch_size, length, c = q.shape

        # Execute the Linear layer for query, key and value
        q, k, v = self.q_attn(q), self.k_attn(k), self.v_attn(v)

        # Compute the matmul between Q and K_T
        att = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        # if masked
        if self.mask:
            att = att.masked_fill(self.bias[:length,:length] == 0, float('-inf'))

        # apply softmax
        att = F.softmax(att, dim=-1)
        att = self.a_dropout(att)

        # Compute the matmul between softmax result and V
        y = att @ v
                
        # Compute dropout and last linear layer
        y = self.b_dropout(self.d_proj(y))

        return y


class FeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, input):
        return self.net(input)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len, dropout, mask=False):
        super().__init__()
        self.heads = nn.ModuleList([ HeadAttention(d_model, num_heads, max_seq_len, dropout, mask=mask) for _ in range(num_heads) ])
        self.proj = nn.Linear(d_model * num_heads, d_model)        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        # concat the result of scaled dot product results
        out = torch.cat([ h(q, k, v) for h in self.heads ], dim=-1)
        # forward the result from concat
        out = self.dropout(self.proj(out))
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len, d_ff, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout, mask=False)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # perform the self-attention, linear, layer norm, and skip connections
        x = x + self.norm1(self.sa(x, x, x))
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len, d_ff, enc_vocab_size, padding_idx, num_layers, dropout):
        super().__init__()
        self.emb = PositionalEncoding(d_model, max_seq_len, enc_vocab_size, padding_idx)
        self.encoder_layers = nn.Sequential(*[ EncoderLayer(d_model, num_heads, max_seq_len, d_ff, dropout) for _ in range(num_layers) ])

    def forward(self, x):
        # create embeddings and forward to the encoder layers
        x = self.emb(x)
        x = self.encoder_layers(x)
        return x


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, max_seq_len, d_ff, dropout):
        super().__init__()
        self.mask_sa = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout, mask=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.sa = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout, mask=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder):
        # compute the masked self attention, norm layer, and dropout
        x = x + self.dropout1(self.norm1(self.mask_sa(x, x, x)))
        # compute the self attention, norm layer, linear, and dropout
        x = x + self.dropout2(self.norm2(self.sa(x, encoder, encoder)))
        x = x + self.dropout3(self.ffn(self.norm3(x)))

        return x


class Decoder(nn.Module):
    
    def __init__(self, d_model, num_heads, max_seq_len, d_ff, dec_vocab_size, padding_idx, num_layers, dropout):
        super().__init__()
        self.emb = PositionalEncoding(d_model, max_seq_len, dec_vocab_size, padding_idx)
        self.decoder = nn.ModuleList([ DecoderLayer(d_model, num_heads, max_seq_len, d_ff, dropout) for _ in range(num_layers) ])
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, dec_vocab_size)


    def forward(self, x, y):        
        # create embeddings, and compute the decoder layer with inputs from encoder and target values
        x = self.emb(x)
        for layer in self.decoder:
            x = layer(x, y)
        # compute the linear and norm layers
        output = self.linear(self.norm1(x))
        return output


class Transformer(nn.Module):
    
    def __init__(self, d_model, num_heads, max_seq_len, d_ff, enc_vocab_size, dec_vocab_size, src_pad_idx, trg_pad_idx, num_layers, dropout):
        super().__init__()
        
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            d_ff=d_ff,
            enc_vocab_size=enc_vocab_size,
            padding_idx=src_pad_idx,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            d_ff=d_ff,
            dec_vocab_size=dec_vocab_size,
            padding_idx=trg_pad_idx,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x, y):
        # encoder & decoder blocks
        enc = self.encoder(x)
        out = self.decoder(y, enc)
        return out
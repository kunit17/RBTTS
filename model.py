import torch
import torch.nn as nn
from torch.nn import functional as F





class FeedFoward(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), # expand 4x more neurons to give model more freedom to learn and expressiveness. (demonstrated by GPT)
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model), # projection
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, head_size)
        self.cross_attn = MultiHeadAttention(n_heads, head_size)  # Adding cross-attention
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Self-attention followed by add & norm
        x = x + self.self_attn(self.ln1(x), mask=mask)
        # Feedforward followed by add & norm
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, head_size, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, head_size)
        self.cross_attn = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, context):
        # Self-attention followed by add & norm
        x = x + self.self_attn(self.ln1(x))
        # Cross-attention with encoder output as context
        x = x + self.cross_attn(self.ln2(x), context)
        # Feedforward followed by add & norm
        x = x + self.ffwd(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, head_size, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.d_model = n_heads * head_size
        # Linear layers for query, key, and value
        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        # Output projection
        self.proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, d_model) for the query input.
            context: Tensor of shape (B, S, d_model) for the key-value input (optional).
                If None, performs self-attention.
        Returns:
            Tensor of shape (B, T, d_model) after applying multi-head attention.
        """
        # Ensure input x has a batch dimension
        if x.dim() == 2:  # If input is (T, d_model)
            x = x.unsqueeze(0)  # Add batch dimension -> (1, T, d_model)

        if context is not None and context.dim() == 2:  # If context is (S, d_model)
            context = context.unsqueeze(0)  # Add batch dimension -> (1, S, d_model)

        B, T, _ = x.shape
        context = x if context is None else context

        # Compute query, key, value
        q = self.query(x)  # (B, T, d_model)
        k = self.key(context)  # (B, S, d_model)
        v = self.value(context)  # (B, S, d_model)

        # Split into multiple heads and reshape
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)
        k = k.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, S, head_size)
        v = v.view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, S, head_size)

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, n_heads, T, S)

        if mask is not None:
            # Adjust mask dimensions and apply
            mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            ask = mask.expand(x.shape[0], 1, 1, x.shape[1])  # Match batch size and sequence length
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Compute attention weights
        # scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, n_heads, T, S)
        # print(f"x shape: {x.shape}")  # Should be [B, T, d_model]
        # if mask is not None:
        #     # Adjust mask dimensions to [B, 1, 1, T] for broadcasting
        #     print(f"mask shape: {mask.shape}")  # Should be [B, 1, 1, T]
        #     mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        #     mask = mask.expand(x.shape[0], 1, 1, x.shape[1])  # Match batch size and sequence length
        #     print(f"mask shape: {mask.shape}")  # Should be [B, 1, 1, T]
        

        weights = F.softmax(scores, dim=-1)  # Normalize scores
        weights = self.dropout(weights)

        # Compute the attention output
        attention_output = weights @ v  # (B, n_heads, T, head_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)

        # Apply final linear projection
        out = self.dropout(self.proj(attention_output))
        return out
    
class Transformer(nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(char_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        # self.mel_token_embedding = nn.Embedding(n_mels, d_model)  # Assuming  mel indices
        # self.mel_position_embedding = nn.Embedding(max_ts, d_model)  # Context length for mel spectrogram        
        self.encoder = Encoder(n_layers, d_model, n_heads, head_size, dropout_rate)
        self.decoder = Decoder(n_layers, d_model, n_heads, head_size, dropout_rate)
        self.trg_proj = nn.Linear(n_mels, d_model)
        self.ln_f = nn.LayerNorm(d_model) # Final layer norm before the head
        self.ln_head = nn.Linear(d_model, n_mels)  # Linear layer to project back to mel spectrogram size
        # self.ln_head2 = nn.Linear(d_model, n_mels)

    def forward(self, src_idx, mask, targets, pred):
        B, T = src_idx.shape
        #idx and target are both (B,T) tensors
        tok_embed = self.token_embedding_table(src_idx) #B,T,C
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #T,C
        x = tok_embed + pos_embed # B,T,C
        encoder_output = self.encoder(x, mask)

        # Embedding and positional encoding for target (decoder input)
        B_tgt, T_tgt = pred.shape
        # tgt_tok_embed = self.token_embedding_table(targets)
        # tgt_pos_embed = self.position_embedding_table(torch.arange(T_tgt, device=targets.device))
        # tgt = tgt_tok_embed + tgt_pos_embed
        pred = self.trg_proj(pred)
        decoder_output = self.decoder(pred, encoder_output)
        mel_output = self.ln_head(self.ln_f(decoder_output))

        targets = targets.unsqueeze(0)  # Shape: [1, 302, 128]
        
        print(mel_output.shape)
        if targets is None:
            loss = None
        else:
            loss = torch.nn.functional.l1_loss(mel_output, targets, size_average=None, reduce=None, reduction='mean')
        return mel_output, loss




    

        





import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class FeedForward(nn.Module):

    def __init__(self, d_model, dropout_rate):
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
        self.self_attn = MultiHeadAttention(n_heads, head_size, dropout_rate)
        self.ffwd = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_mask):
        # Self-attention followed by add & norm
        x = x + self.self_attn(self.ln1(x), mask=encoder_mask)
        # Feedforward followed by add & norm
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, head_size, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_mask=encoder_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, head_size, dropout_rate)
        self.cross_attn = MultiHeadAttention(n_heads, head_size, dropout_rate)
        self.ffwd = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, context, decoder_mask=None, encoder_mask=None):
        # Self-attention 
        x = x + self.self_attn(self.ln1(x), context=None, mask=decoder_mask) # (B, timesteps, D_model) 
        # Cross-attention with encoder output as context
        x = x + self.cross_attn(self.ln2(x), context=context, mask=encoder_mask) # (B, timesteps, D_model) context = (B, T, d_model)
        # Feedforward 
        x = x + self.ffwd(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, head_size, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, head_size, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, context, decoder_mask=None, encoder_mask=None):
        for layer in self.layers:
            x = layer(x, context, decoder_mask, encoder_mask) # x is (B, timesteps, D_model) context is ((B, S, d_model))
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model

        # Linear layers for query, key, and value
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Rotary positional embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=max_seq_len)

    def forward(self, x, context=None, mask=None, input_pos=None):
        """
        Args:
            x: Tensor of shape (B, T, d_model) for the query input.
            context: Tensor of shape (B, S, d_model) for the key-value input (optional).
                If None, performs self-attention.
            mask: Tensor of shape (B, T, S), masking for attention (optional).
            input_pos: Tensor of shape (T,) indicating the position of each token (optional).
        Returns:
            Tensor of shape (B, T, d_model) after applying multi-head attention.
        """
        B, T, _ = x.shape
        context = x if context is None else context
        S = context.shape[1]

        # Compute query, key, value
        q = self.query(x)  # (B, T, d_model)
        k = self.key(context)  # (B, S, d_model)
        v = self.value(context)  # (B, S, d_model)

        # Split into multiple heads and reshape
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)
        k = k.view(B, S, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, S, head_size)
        v = v.view(B, S, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, S, head_size)

        # Apply rotary positional embeddings to query and key
        q = self.rotary_pos_emb(q, input_pos=input_pos)
        k = self.rotary_pos_emb(k, input_pos=input_pos)

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, n_heads, T, S)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Normalize scores and apply dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute the attention output
        attention_output = weights @ v  # (B, n_heads, T, head_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)

        # Apply final linear projection
        out = self.dropout(self.proj(attention_output))
        return out

    
class Transformer(nn.Module):
    
    def __init__ (self, block_size, char_size, d_model, n_heads, dropout_rate, head_size, n_layers, n_mels):
        super().__init__()
        self.token_embedding_table = nn.Embedding(char_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)      
        self.encoder = Encoder(n_layers, d_model, n_heads, head_size, dropout_rate)
        self.decoder = Decoder(n_layers, d_model, n_heads, head_size, dropout_rate)
        self.trg_proj = nn.Linear(n_mels, d_model)
        self.ln_f = nn.LayerNorm(d_model) # Final layer norm before the head
        self.ln_head = nn.Linear(d_model, n_mels)  # Linear layer to project back to mel spectrogram size

    def forward(self, src_idx, encoder_mask, decoder_input, decoder_mask):
        B, T = src_idx.shape # (8 , max_input_from_batch)
        print(f'src shape is {B,T}')
        tok_embed = self.token_embedding_table(src_idx) #B, T, d_model
        # Positional embedding (fetch only the first T positions)
        pos_embed = self.position_embedding_table.weight[:T]  # Shape: (T, d_model)
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, T, d_model)
        x = tok_embed + pos_embed # B,T,d_model
        encoder_output = self.encoder(x, encoder_mask) #(B, T, d_model)
        pred = self.trg_proj(decoder_input) # (B , timesteps, d_model)
        decoder_output = self.decoder(pred, encoder_output, decoder_mask, encoder_mask)
        mel_output = self.ln_head(self.ln_f(decoder_output))
        
        return mel_output






    

        





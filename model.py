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
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout_rate)
        self.ffwd = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_mask):
        # Self-attention followed by add & norm
        x = x + self.self_attn(self.ln1(x), mask=encoder_mask)
        print(f'x in encoder is {x.shape}')
        # Feedforward followed by add & norm
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, encoder_mask):
        for layer in self.layers:
            print(f"encoder mask in forward of encoder is {encoder_mask.shape}")
            x = layer(x, encoder_mask=encoder_mask)
            print(f'x in encoder is {x.shape}')
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout_rate)
        self.cross_attn = MultiHeadAttention(n_heads, d_model, dropout_rate)
        self.ffwd = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, context, decoder_mask=None, encoder_mask=None):
        # Self-attention 
        x = x + self.self_attn(self.ln1(x), context=None, mask=decoder_mask) # (B, timesteps, D_model) 
        print(f'x in decoder is {x.shape}')
        # Cross-attention with encoder output as context
        x = x + self.cross_attn(self.ln2(x), context=context, mask=encoder_mask) # (B, timesteps, D_model) context = (B, T, d_model)
        # Feedforward 
        x = x + self.ffwd(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x, context, decoder_mask=None, encoder_mask=None):
        for layer in self.layers:
            x = layer(x, context, decoder_mask=decoder_mask, encoder_mask=encoder_mask) # x is (B, timesteps, D_model) context is ((B, S, d_model))
            print(f'x in decoder is {x.shape}, decoder mask is {decoder_mask.shape}, encoder mask is {encoder_mask.shape}')
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout_rate, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)  # Projects decoder output to query
        self.k_proj = nn.Linear(d_model, d_model)  # Projects encoder output to key
        self.v_proj = nn.Linear(d_model, d_model)  # Projects encoder output to value
        # Rotary positional embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=max_seq_len)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, context=None, mask=None):    
        B, T, _ = x.size()
        context = x if context is None else context
        S = context.shape[1]
        print('x shape is ' , x.shape)
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(context)  # (B, S, d_model)
        v = self.v_proj(context)  # (B, S, d_model)
        # Split into multiple heads and reshape
        q = q.view(B, T, self.n_heads, self.head_size)  # (B, T, n_heads, head_size)
        k = k.view(B, S, self.n_heads, self.head_size)  # (B, S, n_heads, head_size)
        # Apply rotary positional embeddings to query and key Args: x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
        q = self.rotary_pos_emb(q)
        k = self.rotary_pos_emb(k)     
        q = q.transpose(1, 2)  # (B, n_heads, T, head_size)
        k = k.transpose(1, 2)  # (B, n_heads, S, head_size)
        v = v.view(B, S, self.n_heads, self.head_size).transpose(1,2) # (B, n_heads, S, head_size)
        print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(3).expand(-1,self.n_heads, -1, T) # B,n_heads,T,T
        print(f'mask in attention is {mask[0,0,-5:-1,-5:-1]}, {mask.shape}')
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False) # Flash attention; no dropout
        print(f'attention is: {attention_output.shape}')
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        # Apply final linear projection
        out = self.dropout(self.proj(attention_output))
        return out

    
class Transformer(nn.Module): 
    
    def __init__ (self, block_size, char_size, d_model, n_heads, dropout_rate, n_layers, n_mels):
        super().__init__()
        self.token_embedding_table = nn.Embedding(char_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)      
        self.encoder = Encoder(n_layers, d_model, n_heads, dropout_rate)
        self.decoder = Decoder(n_layers, d_model, n_heads, dropout_rate)
        self.trg_proj = nn.Linear(n_mels, d_model)
        self.ln_f = nn.LayerNorm(d_model) # Final layer norm before the head
        self.ln_head = nn.Linear(d_model, n_mels)  # Linear layer to project back to mel spectrogram size; weight matrix is (n_mels, d_model); n_mels is out feature

    def forward(self, src_idx, encoder_mask, decoder_input, decoder_mask, batch_targets):
        B, T = src_idx.shape # (8 , max_input_from_batch)
        print(f'src shape is {B,T}')
        tok_embed = self.token_embedding_table(src_idx) #B, T, d_model
        # Positional embedding (fetch only the first T positions)
        pos_embed = self.position_embedding_table.weight[:T]  # Shape: (T, d_model)
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, T, d_model)
        x = tok_embed + pos_embed # B,T,d_model
        print(f'after embedding, before encoder: {x.shape}, encoder mask is {encoder_mask.shape}, decoder is {decoder_mask.shape}')
        encoder_output = self.encoder(x, encoder_mask) #(B, T, d_model)
        pred = self.trg_proj(decoder_input) # (B , timesteps, d_model)
        decoder_output = self.decoder(pred, encoder_output, decoder_mask, encoder_mask)
        mel_output = self.ln_head(self.ln_f(decoder_output))

        non_padded_tgt_loc = (batch_targets != -100)
        mel_output = torch.where(non_padded_tgt_loc, mel_output, torch.tensor(0.0))
        y = torch.where(non_padded_tgt_loc, batch_targets, torch.tensor(0.0))
        loss = torch.nn.functional.l1_loss(mel_output, y, reduction='sum') / non_padded_tgt_loc.float().sum()
        return mel_output, loss






    

        





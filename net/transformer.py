import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_sinusiud_encoding_table(n_pos, d_model):
    def cal_angle(pos, hid_idx):
        return pos / np.power(10000, 2*(hid_idx // 2) / d_model)
    
    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hid_j) for hid_j in range(d_model)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, hidden_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.hidden_ff = hidden_ff
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_ff, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dim):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, n_heads * dim, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * dim, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * dim, bias=False)
        self.fc = nn.Linear(n_heads*dim, d_model, bias=False)
    
    def forward(self, input_q, input_k, input_v, attn_mask):
        residual, batch_size = input_q, input_v.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dim)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.dim) 
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, dim, hidden_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, dim=dim, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, hidden_ff=hidden_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_fc_size, block_num, dim):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusiud_encoding_table(vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, dim=dim, hidden_ff=hidden_fc_size) for _ in range(block_num)])
    
    def forward(self, enc_inputs):
        word_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs)
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# class Transformer(nn.Module):
#     def __init__(self):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder()
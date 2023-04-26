import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda:{}'.format(2))
def kaiming_normal(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    nn.init.zeros_(layer.bias)

def xavier_init(layer):
    tanh_gain = nn.init.calculate_gain('tanh')
    nn.init.xavier_normal_(layer.weight, tanh_gain)
    nn.init.zeros_(layer.bias)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

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
            nn.Tanh(),
            nn.Linear(hidden_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda(2)(output + residual) # [batch_size, seq_len, d_model]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim) 
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
        
    def forward(self, input_q, input_k, input_v):
        residual, batch_size = input_q, input_v.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dim)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.dim) 
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).cuda(2)(output + residual), attn

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, dim, hidden_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, dim=dim, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, hidden_ff=hidden_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class TransformerEncoder(nn.Module):
    def __init__(self, state_dim, vocab_size, d_model, hidden_fc_size, block_num, dim, n_heads):
        super(TransformerEncoder, self).__init__()
        self.w_layers = nn.ModuleList()
        self.d_model = d_model
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusiud_encoding_table(vocab_size, d_model), freeze=True)
        # for _ in range(vocab_size):
        #     self.w_layers.append(nn.Linear(state_dim, d_model))
        #     kaiming_normal(self.w_layers[-1])
        self.linear = nn.Linear(state_dim, d_model)
        xavier_init(self.linear)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, dim=dim, hidden_ff=hidden_fc_size, n_heads=n_heads) for _ in range(block_num)])
        
    def forward(self, s):
        _, seq_len = s.shape[:2]
        # after_s = torch.zeros(batch_size, seq_len, self.d_model, dtype=torch.float32).to(device)
        enc_self_attns = []

        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         after_s[b, i, :] = self.w_layers[i](s[b, i, :])

        enc_outputs = F.tanh(self.linear(s))
        pos_emb = self.pos_emb(torch.LongTensor(range(seq_len)).to(device))
        enc_outputs = enc_outputs + pos_emb
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
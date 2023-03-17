import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

n_inf = -1e9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TemporalEmbedding(nn.Module):
    def __init__(self, dim, seq_len=100, dropout=0.):
        super(TemporalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len
        self.dim = dim
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10.0) / dim))
        self.div_term = div_term.to(device)

    def forward(self, x, time_seq):
        # time_seq: [bh_size, max_len], if embedding for time token, max_len = 1: [bh_size, 1]
        # x: [bh_size, max_len, h_dim]
        time_seq = time_seq.unsqueeze(2).expand([len(x), self.seq_len, self.dim])
        te = torch.zeros([len(x), self.seq_len, self.dim]).float().to(device)
        te[:, :, 0::2] = torch.sin(time_seq[:, :, 0::2] * self.div_term)
        te[:, :, 1::2] = torch.cos(time_seq[:, :, 1::2] * self.div_term)
        # x [seq_len, batch_size, x_dim]
        x = x + te
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, h_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=6, dim_head=64, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        h_dim = dim_head * n_heads
        self.dim = dim
        project_out = not (n_heads == 1 and dim_head == dim)
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(dim, h_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.scale = dim_head ** -0.5

    def forward(self, x, attn_mask):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).eq(0)
        # [b h n d] [b h d n] -> [b h n n]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores.masked_fill_(attn_mask, n_inf)
        attn = self.attend(scores)
        context = torch.matmul(attn, v)

        context = rearrange(context, 'b h n d -> b n (h d)')
        return self.out_layer(context)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, dim_fc, dim_head=64, depth=3, n_heads=6, dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.encoders = nn.ModuleList([])
        for _ in range(depth):
            self.encoders.append(nn.ModuleList([
                MultiHeadAttention(dim, n_heads=n_heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, h_dim=dim_fc, dropout=dropout)
            ]))

    def forward(self, x, attn_mask):
        for multi_attn, ff in self.encoders:
            x = self.norm(x)
            x = multi_attn(x, attn_mask) + x
            x = self.norm(x)
            x = ff(x) + x
        return x


class FeatureAttention(nn.Module):
    def __init__(self, dim, h_dim=5, dropout=0.):
        super(FeatureAttention, self).__init__()
        self.dim = dim
        self.to_qkv = nn.Linear(dim, h_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        project_out = (h_dim != dim)
        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.scale = h_dim ** -0.5

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v -> [b d n] * [b n d] -> [b d d]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn [b d d]
        attn = self.attend(scores)
        # context = [b d d] * [b n d] -> [b d n]
        context = torch.matmul(attn, v)
        context = self.out_layer(context)
        return rearrange(context, 'b d n -> b n d'), attn


class TemporalTransformer(nn.Module):
    def __init__(self, x_dim, dim, dim_fc, max_len, dim_head=64, depth=3, n_heads=6, pool='time', dropout=0.):
        super(TemporalTransformer, self).__init__()
        self.dim = dim
        self.pool = pool
        self.seq_len = max_len
        self.projection = nn.Linear(x_dim, dim, bias=False)
        self.feature_attn = FeatureAttention(self.seq_len, h_dim=self.seq_len, dropout=dropout)
        self.time_token = nn.Parameter(torch.randn(1, 1, dim))
        self.time_emb = TemporalEmbedding(dim, seq_len=max_len)
        self.time_emb2 = TemporalEmbedding(dim, seq_len=1)
        self.dropout = nn.Dropout(dropout)
        self.trans_encoder = TransformerEncoder(dim, dim_fc, dim_head, depth, n_heads, dropout)

    def forward(self, x, t, attn_mask, time_seq):
        # x -> [bh_size, seq_len, x_dim]
        # attn_mask -> [hb_size, seq_len]
        # time_seq -> [bh_size, seq_len]
        x, _ = self.feature_attn(x)
        x = self.projection(x)
        x = self.time_emb(x, time_seq)

        # x -> [bh_size, seq_len, dim]
        bh_size, n, _ = x.shape
        time_token = repeat(self.time_token, '() n d -> b n d', b=bh_size)
        time_token = self.time_emb2(time_token, t.unsqueeze(1))

        x = torch.cat([time_token, x], dim=1)
        x = self.dropout(x)
        time_mask = torch.ones([bh_size, 1]).float().to(device)
        attn_mask = torch.cat([time_mask, attn_mask], dim=1)
        attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.seq_len)
        x = self.trans_encoder(x, attn_mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x

    def get_attn_score(self, x):
        _, attn = self.feature_attn(x)
        return attn

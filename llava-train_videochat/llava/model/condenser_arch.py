import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


class TransformerCondenser(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert False, "This class is not implemented yet"

        self.stride = int(os.environ.get('EXTRA_HYPER_PARAMS_STRIDE', '4'))
        self.reshape_style = os.environ.get('EXTRA_HYPER_PARAMS_RESHAPE_STYLE', 'avgpool') # avgpool or truncation
        self.attn_style = os.environ.get('EXTRA_HYPER_PARAMS_ATTN_STYLE', 'encoder') # encoder or decoder
        self.num_layers = int(os.environ.get('EXTRA_HYPER_PARAMS_NUM_LAYERS', '2'))
        assert self.attn_style in ['encoder', 'decoder']
        assert self.reshape_style in ['avgpool', 'truncation']

        config.num_hidden_layers = self.num_layers
        config.is_causal = self.attn_style == 'decoder'
        self.model = Qwen2Model(config)
        if self.reshape_style == 'avgpool':
            self.avgpool = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride)
        # transformer already has weight initialization

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        y = self.model(
            input_ids = None,
            attention_mask = None,
            inputs_embeds = x,
            output_attentions = False,
            output_hidden_states = True
        )
        out_hidden = y.last_hidden_state
        if self.reshape_style == 'avgpool':
            out_hidden = out_hidden.permute(0, 2, 1)
            out_hidden = self.avgpool(out_hidden)
            out_hidden = out_hidden.permute(0, 2, 1)
        else:
            out_hidden = out_hidden[:, -out_hidden.shape[1] // self.stride:, :]
        return out_hidden




class TransformerEncoderLayerRotary(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead

        # 手动的投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # FeedForward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

        # Rotary Embedding
        self.rotary_emb = RotaryEmbedding(dim=d_model // nhead)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: [seq_len, batch_size, d_model]
        src_mask: [seq_len, seq_len] or broadcastable
        src_key_padding_mask: [batch_size, seq_len] or broadcastable
        """
        seq_len, batch_size, embed_dim = src.size()
        assert embed_dim == self.d_model, "src hidden dim != d_model"

        # ---- 1. 线性投影 ----
        q = self.q_proj(src)  # [seq_len, batch_size, d_model]
        k = self.k_proj(src)
        v = self.v_proj(src)

        # ---- 2. 多头拆分 [seq_len, batch_size, d_model] -> [batch_size, nhead, seq_len, head_dim] ----
        q = q.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)  # (b, h, s, hd)
        k = k.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)  # (b, h, s, hd)
        v = v.view(seq_len, batch_size, self.nhead, self.head_dim).permute(1, 2, 0, 3)  # (b, h, s, hd)

        # ---- 3. Rotary Embedding (只对 Q, K 旋转) ----
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        # 有需要也可以对 v 做同样操作，但常见做法只对 q/k

        # ---- 4. 注意力计算 ----
        # 4.1 QK^T / sqrt(d)
        # (b, h, s, hd) x (b, h, hd, s) => (b, h, s, s)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4.2 加 src_mask (shape 通常是 [seq_len, seq_len]，需要广播到 [1, 1, s, s] 或者类似)
        if src_mask is not None:
            # src_mask: [seq_len, seq_len]
            # attn_scores: [batch_size, nhead, seq_len, seq_len]
            attn_scores = attn_scores + src_mask.unsqueeze(0).unsqueeze(0)

        # 4.3 加 key_padding_mask (shape 通常是 [batch_size, seq_len])
        if src_key_padding_mask is not None:
            # src_key_padding_mask: [batch_size, seq_len]
            # 我们要把它变成 (batch_size, 1, 1, seq_len)，再广播到 (b, h, s, s)
            mask_expanded = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, s)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))

        # 4.4 softmax -> attn_probs
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 4.5 与 V 相乘
        # (b, h, s, s) x (b, h, s, hd) => (b, h, s, hd)
        out = torch.matmul(attn_probs, v)

        # ---- 5. 拼回原形 [batch_size, nhead, seq_len, head_dim] -> [seq_len, batch_size, d_model] ----
        out = out.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, embed_dim)

        # ---- 6. 残差 & LayerNorm (MHA 之后) ----
        src = src + self.dropout1(out)
        src = self.norm1(src)

        # ---- 7. Feed Forward ----
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class AvgPoolingCondenser(nn.Module):
    def __init__(self, hidden_size=4096, num_layers=1):
        super(AvgPoolingCondenser, self).__init__()
        stride = int(os.environ.get('EXTRA_PARAM_INNER_STRIDE', '4'))
        print('!!!!!!!!!! [Warning]', 'You are using AvgPoolingCondenser with stride', stride, 'This is considered an inner condenser')
        self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)
        self.num_layers = num_layers
        # extra num_layers - 1 MLP layers
        self.extra_mlps = nn.ModuleList([
            nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
            ) for _ in range(num_layers - 1)
        ])

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        if self.num_layers > 1:
            for mlp in self.extra_mlps:
                x = mlp(x)
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        x = self.pool(x)  # [batch_size, hidden_size, seq_len//4]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len//4, hidden_size]
        return x


class SelfAttentionCondenser(nn.Module):
    def __init__(self, hidden_size=4096, num_layers=1, position_embedding_type=None):
        super(SelfAttentionCondenser, self).__init__()
        if position_embedding_type == "rotary":
            self.layers = nn.ModuleList([
                TransformerEncoderLayerRotary(d_model=hidden_size, nhead=8)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
                for _ in range(num_layers)
            ])
        
        stride = int(os.environ.get('EXTRA_PARAM_OUTER_STRIDE', '4'))
        print('!!!!!!!!!! [Warning]', 'You are using SelfAttentionCondenser with stride', stride, 'This is considered an outer condenser')
        self.stride = stride
        self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)
        self.position_embedding_type = position_embedding_type
        self.hidden_size = hidden_size

        if position_embedding_type == "hier":
            self.segment_embeddings = nn.Parameter(torch.randn(4, hidden_size))
        
        self.forward_style = os.environ.get('EXTRA_PARAM_OUTER_CONDENSER_FORWARD_TYPE', 'None')

        self._init_weights()
        self.pos_encoding_cache = None  # Cache for positional encoding

    def _init_weights(self):
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _generate_positional_encoding(self, seq_len, hidden_size, device, dtype):
        if (
            self.pos_encoding_cache is not None 
            and self.pos_encoding_cache["seq_len"] == seq_len
            and self.pos_encoding_cache["device"] == device
            and self.pos_encoding_cache["dtype"] == dtype
        ):
            return self.pos_encoding_cache["pos_encoding"]

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, device=device, dtype=dtype) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(seq_len, hidden_size, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pos_encoding_cache = {
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "pos_encoding": pe
        }
        return pe

    def _apply_position_embedding(self, x):
        if self.position_embedding_type == 'rotary':
            return x

        # print('before pos embedding', x.shape, x.norm())
        seq_len, batch_size, hidden_size = x.size(0), x.size(1), self.hidden_size
        device, dtype = x.device, x.dtype

        pos_encoding = self._generate_positional_encoding(seq_len, hidden_size, device, dtype).unsqueeze(1).repeat(1, batch_size, 1)

        other_factor = 1 / 10

        if self.position_embedding_type == "raw":
            x += pos_encoding * other_factor

        elif self.position_embedding_type == "hier":
            x += pos_encoding * other_factor
            segment_len = seq_len // 4

            for i in range(4):
                start, end = i * segment_len, (i + 1) * segment_len
                x[start:end] += self.segment_embeddings[i].unsqueeze(0).unsqueeze(1) * other_factor
        # print('after pos embedding', x.shape, x.norm())
        return x

    def forward(self, x, wh=28):
        spatial_temporal = 'spatial_temporal' in self.forward_style

        if not spatial_temporal:
            x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]

            if self.position_embedding_type is not None and self.position_embedding_type != "rotary":
                x = self._apply_position_embedding(x)

            for layer in self.layers:
                x = layer(x)

            x = x.permute(1, 2, 0)  # [batch_size, hidden_size, seq_len]
            x = self.pool(x)  # [batch_size, hidden_size, seq_len//4]
            x = x.permute(0, 2, 1)  # [batch_size, seq_len//4, hidden_size]
        else:
            # x: [batch_size, seq_len, hidden_size]
            b, t, c = x.size()
            assert t % (wh * wh) == 0, f"seq_len {t} should be divisible by {wh * wh}"
            n_frames = t // (wh * wh)
            n_layers = len(self.layers)
            assert n_layers % 2 == 0, f"n_layers {n_layers} should be divisible by 2"
            
            '''
            Step 1: Spatial Condense
                use first half of layers, process each frame separately (batch_size * n_frames, wh * wh, hidden_size)
                avgpool (take average) of 4x4 patches to get 7x7 patches
                (batch_size * n_frames, 7 * 7, hidden_size)
            Step 2: Temporal Condense
                use second half of layers, process all frames together (batch_size, n_frames * 7 * 7, hidden_size)
                avgpool (take average) of n_frames frames to get 1 frame
                (batch_size, 7 * 7, hidden_size)
            '''
            # Reshape to group the frames; we get [b*n_frames, wh*wh, c]
            x = x.reshape(b * n_frames, wh * wh, c)  # shape => [b*n_frames, wh*wh, hidden_size]
            
            x = x.permute(1, 0, 2)

            # Apply the first half of the layers
            for layer in self.layers[: (n_layers // 2)]:
                x = layer(x)

            x = x.permute(1, 2, 0)  # => [b*n_frames, c, wh*wh]

            x = x.reshape(b * n_frames, c, wh, wh)   # [b*n_frames, c, 28, 28]
            x = F.avg_pool2d(x, kernel_size=4, stride=4)  # => [b*n_frames, c, 7, 7]
            x = x.reshape(b * n_frames, c, 7 * 7)    # => [b*n_frames, c, 49]
            x = x.permute(0, 2, 1)                # => [b*n_frames, 49, c]

            x = x.reshape(b, n_frames * 49, c)  # [b, n_frames*49, c]
            
            x = x.permute(1, 0, 2)  # => [n_frames*49, b, c]

            for layer in self.layers[(n_layers // 2):]:
                x = layer(x)

            # Permute back => [b, n_frames*49, c]
            x = x.permute(1, 0, 2)

            x = x.reshape(b, n_frames, 49, c)  # => [b, n_frames, 49, c]
            x = x.mean(dim=1)               # => [b, 49, c]
        
        return x

class RemoveFirstFrameCondenser(nn.Module):
    def __init__(self):
        super(RemoveFirstTokenCondenser, self).__init__()

    def forward(self, x):
        return x[:, 169:, :]

class IdentityCondenser(nn.Module):
    def __init__(self):
        super(IdentityCondenser, self).__init__()

    def forward(self, x):
        return x


# SelectTheNextHalf(prev_idx=idx_lists[_], next_idx=idx_lists[_+1])
class SelectTheNextHalf(nn.Module):
    def __init__(self, stride=None, prev_idx=None, next_idx=None):
        super(SelectTheNextHalf, self).__init__()
        self.stride = stride
        if stride is None:
            self.stride = int(os.environ.get('EXTRA_PARAM_INNER_STRIDE', '2'))
        if prev_idx is not None:
            self.stride = len(prev_idx)
            self.prev_idx = prev_idx
            self.next_idx = next_idx
        else:
            self.prev_idx = None
            self.next_idx = None
        if 'EXTRA_PARAM_ALWAYS_SUFFIX' in os.environ:
            assert self.prev_idx is not None, "prev_idx should not be None when EXTRA_PARAM_ALWAYS_SUFFIX is set"
            self.next_idx = self.prev_idx[-len(self.next_idx):]
            print(f'set always suffix: {self.prev_idx} -> {self.next_idx}')

    def forward(self, x):
        # for every self.stride tokens, select the last one
        # x: [batch_size, seq_len, hidden_size]
        b, t, c = x.size()
        assert t % self.stride == 0, f"seq_len {t} should be divisible by stride {self.stride}"
        x = x.reshape(b, t // self.stride, self.stride, c)
        if self.prev_idx is not None:
            mask = torch.zeros(self.stride, dtype=torch.bool, device=x.device)
            for i, v in enumerate(self.prev_idx):
                if v in self.next_idx:
                    mask[i] = True
            x = x[:, :, mask, :]
        else:
            x = x[:, :, -1, :]
        x = x.reshape(b, -1, c)
        return x


class SelectRowByRow(nn.Module):
    def __init__(self):
        super(SelectRowByRow, self).__init__()

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        b, t, c = x.size()
        assert t % 7 == 0, f"seq_len {t} should be divisible by 7"
        # for each adjacent 7 tokens 1~7, select 1 3 5 7
        # your code here
        x = x.view(b, t // 7, 7, c)
        x = x[:, :, [0, 2, 4, 6], :]
        x = x.view(b, -1, c)
        return x

class StupidPooling(nn.Module):
    def __init__(self):
        super(StupidPooling, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.stride = 13*14
        self.fake_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)
        return x

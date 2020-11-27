import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

import os
from fairseq import utils

from fairseq.modules import multihead_attention

DEBUG = False


def maybe_print(s):
    if DEBUG:
        print('lgl_attention.py:: ' + s)

GETMASK = bool(int(os.environ.get('GETMASK', 0)))

GPU = torch.device("cuda")
GPU0 = torch.device("cuda:0")


class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 9
    def __init__(
            self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
            gumbel_alpha=1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        if isinstance(gumbel_alpha, (int, float)):
            self._gumbel_alpha = gumbel_alpha
            self.const_gumbel_alpha = True
        else:
            assert isinstance(gumbel_alpha, str)
            # if gumbel_alpha == 'seq_len':
            self._gumbel_alpha = gumbel_alpha
            self.const_gumbel_alpha = False

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(
            # needs:
            # 2 for global q k
            # 1 for values
            # 2 for local left
            # 2 for local right
            # 2 for local
            # orders:
            # global q, k, v, left q, k, right q, k, local q,k,
            self.NUM_WEIGHTS * embed_dim, embed_dim
        ))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def gumbel_alpha(self, padding_mask, **kwargs):
        """
            padding_mask is 1 where padding idx is
        :param padding_mask:    [b, 1, 1, src_len]
        :param kwargs:
        :return:
        """
        if self.const_gumbel_alpha:
            return self._gumbel_alpha
        elif self._gumbel_alpha == 'src_len':
            if padding_mask is None:
                return 1.0
            else:
                gumbel_mask = (1.0 - padding_mask.float()).sum(dim=-1)
                g_size = gumbel_mask.size()
                bsz = g_size[0]
                gumbel_mask = gumbel_mask.expand(bsz, self.num_heads, 1).contiguous().view(bsz * self.num_heads, 1, 1)
                # [b * h, 1, 1]
                return gumbel_mask
        else:
            raise NotImplementedError(f'gumbel {self._gumbel_alpha} not impl')

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    # all
    def in_proj_all(self, query):
        return self._in_proj(query).chunk(self.NUM_WEIGHTS, dim=-1)

    # global
    def in_proj_global_qkv(self, query):
        return self._in_proj(query, start=0, end=3 * self.embed_dim).chunk(3, dim=-1)

    def in_proj_global_kv(self, key):
        return self._in_proj(key, start=self.embed_dim, end=3 * self.embed_dim).chunk(2, dim=-1)

    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)

    def in_proj_local_k(self, key):
        return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def maybe_add_att_padding_mask(self, attn_weights, exp_key_padding_mask, bsz, tgt_len, src_len):
        if exp_key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    # key_padding_mask.unsqueeze(1).unsqueeze(2),
                    exp_key_padding_mask,
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    exp_key_padding_mask,
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        return attn_weights

    def prepare_local_masking(self, q_left, k_left, q_right, k_right, padding_mask, bsz, **kwargs):
        # left_attn_weights = torch.bmm(q_left, k_left.transpose_(1, 2))
        # right_attn_weights = torch.bmm(q_right, k_right.transpose_(1, 2))
        left_attn_weights = torch.bmm(q_left, k_left.transpose(1, 2))
        right_attn_weights = torch.bmm(q_right, k_right.transpose(1, 2))

        # assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        # weights: [b * h, lq, lk]
        left_size = left_attn_weights.size()
        # bsz = bh // self.num_heads
        bh = left_size[0]
        tgt_len = left_size[1]
        src_len = left_size[2]

        # triu = torch.ones(src_len, src_len).float().cuda().triu_().unsqueeze_(0)
        triu = torch.ones(src_len, src_len, device=GPU, dtype=q_left.dtype).triu_().unsqueeze_(0)
        # triu = torch.ones(src_len, src_len, device=GPU, dtype=q_left.dtype).triu_().unsqueeze(0)
        # triu = torch.ones(src_len, src_len, device=GPU, dtype=q_left.dtype).triu().unsqueeze(0)
        # triu_t = triu.transpose(1, 2)

        # left softmax
        padded_left_weights = self.maybe_add_att_padding_mask(left_attn_weights, padding_mask, bsz, tgt_len, src_len)
        padded_right_weights = self.maybe_add_att_padding_mask(right_attn_weights, padding_mask, bsz, tgt_len, src_len)

        left_softmax = F.softmax(self.gumbel_alpha(padding_mask) * padded_left_weights, dim=-1)
        right_softmax = F.softmax(self.gumbel_alpha(padding_mask) * padded_right_weights, dim=-1)

        local_mask = self.compute_lrmask2localmask(left_softmax, right_softmax, triu)

        return local_mask

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        triu_t = triu.transpose(1, 2)
        left_mask = torch.matmul(left_softmax, triu)
        right_mask = torch.matmul(right_softmax, triu_t)
        bw_left_mask = torch.matmul(left_softmax, triu_t)
        bw_right_mask = torch.matmul(right_softmax, triu)

        fw_mask = left_mask * right_mask
        bw_mask = bw_left_mask * bw_right_mask
        local_mask = fw_mask + bw_mask
        return local_mask

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            # q, k, v = self.in_proj_qkv(query)
            q, k, v, q_left, k_left, q_right, k_right, q_local, k_local = self.in_proj_all(query)
        elif kv_same:
            # raise NotImplementedError(f'encoder-decoder attention not available')
            # encoder-decoder attention
            q = self.in_proj_global_q(query)
            q_left = self.in_proj_local_left_q(query)
            q_right = self.in_proj_local_right_q(query)
            q_local = self.in_proj_local_q(query)
            if key is None:
                assert value is None
                k = v = None
                k_left = k_right = k_local = None
                # raise NotImplementedError(f'kv_same but key is None')
            else:
                k, v = self.in_proj_global_kv(key)
                k_left = self.in_proj_local_left_k(key)
                k_right = self.in_proj_local_right_k(key)
                k_local = self.in_proj_local_k(key)
        else:
            # something else
            # raise NotImplementedError(f'This case should not be here!')
            q = self.in_proj_global_q(query)
            k = self.in_proj_global_k(key)
            v = self.in_proj_global_v(value)
            q_left = self.in_proj_local_left_q(query)
            k_left = self.in_proj_local_left_k(key)
            q_right = self.in_proj_local_right_q(query)
            k_right = self.in_proj_local_right_k(key)
            q_local = self.in_proj_local_q(query)
            k_local = self.in_proj_local_k(key)

        q *= self.scaling
        q_local *= self.scaling
        # q_left *= self.scaling
        # q_right *= self.scaling

        if self.bias_k is not None:
            raise NotImplementedError(f'bias_k not None not impl')
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        # maybe_print(f'sizes:'
        #       f'{q.size()} -'
        #       f'{k.size()} -'
        #       f'{v.size()} -'
        #       f'{q_left.size()} -'
        #       f'{k_left.size()} -'
        #       f'{q_right.size()} -'
        #       f'{k_right.size()} -'
        #       f'{q_local.size()} -'
        #       f'{k_local.size()} -'
        #       )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k_local is not None:
            k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k_left is not None:
            k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k_right is not None:
            k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if q_left is not None:
            q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if q_right is not None:
            q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # q: [b*h, l, d]
        # k: [b*h, l, d]
        # k_local: [b*h, l, d]
        # v: [b*h, l, d]

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            # raise NotImplementedError(f'saved_state not impl for cross or decoder attention yet!')
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_key_local' in saved_state:
                prev_key_local = saved_state['prev_key_local'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_local = prev_key_local
                else:
                    k_local = torch.cat((prev_key_local, k_local), dim=1)
            if 'prev_key_local_left' in saved_state:
                prev_key_local_left = saved_state['prev_key_local_left'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_left = prev_key_local_left
                else:
                    k_left = torch.cat((prev_key_local_left, k_left), dim=1)
            if 'prev_key_local_right' in saved_state:
                prev_key_local_right = saved_state['prev_key_local_right'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_right = prev_key_local_right
                else:
                    k_right = torch.cat((prev_key_local_right, k_right), dim=1)

            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local'] = k_local.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_left'] = k_left.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_right'] = k_right.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        assert src_len == k_local.size(1), f'{src_len} != {k_local.size(1)}'

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            raise NotImplementedError(f'add_zero_attn not impl')
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        # TODO: starting dot product
        exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) if key_padding_mask is not None else None

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        assert_size = [bsz * self.num_heads, tgt_len, src_len]
        assert list(global_attn_weights.size()) == assert_size, f'{global_attn_weights.size()} != {assert_size}'
        assert list(local_attn_weights.size()) == assert_size, f'{local_attn_weights.size()} != {assert_size}'

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, exp_key_padding_mask, bsz)

        masked_local_attn_weights = local_attn_weights * local_att_mask

        attn_weights = global_attn_weights + masked_local_attn_weights

        if attn_mask is not None:
            raise NotImplementedError(f'attn_mask is not None not impl')
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        attn_weights = self.maybe_add_att_padding_mask(attn_weights, exp_key_padding_mask, bsz, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            if GETMASK:
                print(f'---> get local_att_mask')
                attn_weights = local_att_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

# LearnableGlobalLocalMultiheadAttention
class LearnableGlobalLocalMultiheadAttentionV2(LearnableGlobalLocalMultiheadAttention):

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        return lgl_v2_compute_mask2localmask(left_softmax, right_softmax, triu)


class LearnableGlobalLocalExpConMultiheadAttention(LearnableGlobalLocalMultiheadAttention):
    """
    This is not appliable to cross-attention!
    """

    def prepare_local_masking(self, q_left, k_left, q_right, k_right, padding_mask, bsz, **kwargs):
        # left_attn_weights = torch.bmm(q_left, k_left.transpose_(1, 2))
        # right_attn_weights = torch.bmm(q_right, k_right.transpose_(1, 2))
        left_attn_weights = torch.bmm(q_left, k_left.transpose(1, 2))
        right_attn_weights = torch.bmm(q_right, k_right.transpose(1, 2))

        # assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        # weights: [b * h, lq, lk]
        left_size = left_attn_weights.size()
        # bsz = bh // self.num_heads
        bh = left_size[0]
        tgt_len = left_size[1]
        src_len = left_size[2]

        assert tgt_len == src_len, f'tgt_len = {tgt_len} != src_len != {src_len}, not applied to cross-att'

        triu = torch.ones(src_len, src_len, device=GPU, dtype=torch.float).triu_().unsqueeze_(0)
        eye = torch.eye(src_len, device=GPU, dtype=torch.float).unsqueeze_(0)

        # future_mask = (triu - eye) * -1e-9
        # TODO: masking of past and future
        future_mask = (triu - eye).byte()
        past_mask = future_mask.transpose(1, 2)

        gum_left_weights = self.gumbel_alpha(padding_mask) * left_attn_weights
        gum_right_weights = self.gumbel_alpha(padding_mask) * right_attn_weights

        left_attn_weights = gum_left_weights.masked_fill(future_mask, float('-inf'))
        right_attn_weights = gum_right_weights.masked_fill(past_mask, float('-inf'))
        # left_attn_weights = left_attn_weights.masked_fill(future_mask, float('-inf'))
        # right_attn_weights = right_attn_weights.masked_fill(past_mask, float('-inf'))

        padded_left_weights = self.maybe_add_att_padding_mask(left_attn_weights, padding_mask, bsz, tgt_len, src_len)
        padded_right_weights = self.maybe_add_att_padding_mask(right_attn_weights, padding_mask, bsz, tgt_len, src_len)

        # left_softmax = F.softmax(self.gumbel_alpha(padding_mask) * padded_left_weights, dim=-1)
        # right_softmax = F.softmax(self.gumbel_alpha(padding_mask) * padded_right_weights, dim=-1)
        left_softmax = F.softmax(padded_left_weights, dim=-1)
        right_softmax = F.softmax(padded_right_weights, dim=-1)

        left_softmax = left_softmax.masked_fill(torch.isnan(left_softmax), 0.0)
        right_softmax = right_softmax.masked_fill(torch.isnan(right_softmax), 0.0)

        # self._check_not_nan(left_softmax, padding_mask)
        # self._check_not_nan(right_softmax, padding_mask)

        local_mask = self.compute_lrmask2localmask(left_softmax, right_softmax, triu)

        return local_mask

    def _check_not_nan(self, softmax, padding_mask):
        if torch.isnan(softmax).any():
            print(f'softmax nan, {softmax.size()}')
            print(softmax)
            if padding_mask is not None:
                print(f'pad_mask: {padding_mask.size()}')
                print(padding_mask)
            print(f'-------------------------------------')
            raise ValueError(f'softmax is Nan')


class LearnableGlobalLocalExpConMultiheadAttentionV2(LearnableGlobalLocalExpConMultiheadAttention):

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        return lgl_v2_compute_mask2localmask(left_softmax, right_softmax, triu)


class LearnableGlobalHardLocalMultiheadAttention(LearnableGlobalLocalMultiheadAttention):
    NUM_WEIGHTS = 7

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 gumbel_alpha=1.0):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, gumbel_alpha)
        # needs:
        # 2 for global q k
        # 1 for values
        # 2 for local left
        # 2 for local right
        # orders:
        # global q, k, v, left q, k, right q, k, local q,k,

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    # all
    def in_proj_all(self, query):
        return self._in_proj(query).chunk(self.NUM_WEIGHTS, dim=-1)

    # global
    def in_proj_global_qkv(self, query):
        return self._in_proj(query, start=0, end=3 * self.embed_dim).chunk(3, dim=-1)

    def in_proj_global_kv(self, key):
        return self._in_proj(key, start=self.embed_dim, end=3 * self.embed_dim).chunk(2, dim=-1)

    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # def in_proj_local_q(self, query):
    #     return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)
    #
    # def in_proj_local_k(self, key):
    #     return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            # q, k, v = self.in_proj_qkv(query)
            # q, k, v, q_left, k_left, q_right, k_right, q_local, k_local = self.in_proj_all(query)
            q, k, v, q_left, k_left, q_right, k_right = self.in_proj_all(query)
        elif kv_same:
            # raise NotImplementedError(f'encoder-decoder attention not available')
            # encoder-decoder attention
            q = self.in_proj_global_q(query)
            q_left = self.in_proj_local_left_q(query)
            q_right = self.in_proj_local_right_q(query)
            # q_local = self.in_proj_local_q(query)
            if key is None:
                assert value is None
                k = v = None
                # k_left = k_right = k_local = None
                k_left = k_right = None
                # raise NotImplementedError(f'kv_same but key is None')
            else:
                k, v = self.in_proj_global_kv(key)
                k_left = self.in_proj_local_left_k(key)
                k_right = self.in_proj_local_right_k(key)
                # k_local = self.in_proj_local_k(key)
        else:
            # something else
            # raise NotImplementedError(f'This case should not be here!')
            q = self.in_proj_global_q(query)
            k = self.in_proj_global_k(key)
            v = self.in_proj_global_v(value)
            q_left = self.in_proj_local_left_q(query)
            k_left = self.in_proj_local_left_k(key)
            q_right = self.in_proj_local_right_q(query)
            k_right = self.in_proj_local_right_k(key)
            # q_local = self.in_proj_local_q(query)
            # k_local = self.in_proj_local_k(key)

        q *= self.scaling
        # q_local *= self.scaling
        # q_left *= self.scaling
        # q_right *= self.scaling

        if self.bias_k is not None:
            raise NotImplementedError(f'bias_k not None not impl')
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # if k_local is not None:
        #     k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k_left is not None:
            k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k_right is not None:
            k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if q_left is not None:
            q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if q_right is not None:
            q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # q: [b*h, l, d]
        # k: [b*h, l, d]
        # k_local: [b*h, l, d]
        # v: [b*h, l, d]

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            # raise NotImplementedError(f'saved_state not impl for cross or decoder attention yet!')
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            # if 'prev_key_local' in saved_state:
            #     prev_key_local = saved_state['prev_key_local'].view(bsz * self.num_heads, -1, self.head_dim)
            #     if static_kv:
            #         k_local = prev_key_local
            #     else:
            #         k_local = torch.cat((prev_key_local, k_local), dim=1)
            if 'prev_key_local_left' in saved_state:
                prev_key_local_left = saved_state['prev_key_local_left'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_left = prev_key_local_left
                else:
                    k_left = torch.cat((prev_key_local_left, k_left), dim=1)
            if 'prev_key_local_right' in saved_state:
                prev_key_local_right = saved_state['prev_key_local_right'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_right = prev_key_local_right
                else:
                    k_right = torch.cat((prev_key_local_right, k_right), dim=1)

            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            # saved_state['prev_key_local'] = k_local.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_left'] = k_left.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_right'] = k_right.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        # assert src_len == k_local.size(1), f'{src_len} != {k_local.size(1)}'

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            raise NotImplementedError(f'add_zero_attn not impl')
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        # TODO: starting dot product
        exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) if key_padding_mask is not None else None

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        # local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        assert_size = [bsz * self.num_heads, tgt_len, src_len]
        assert list(global_attn_weights.size()) == assert_size, f'{global_attn_weights.size()} != {assert_size}'
        # assert list(local_attn_weights.size()) == assert_size, f'{local_attn_weights.size()} != {assert_size}'

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, exp_key_padding_mask, bsz)

        # masked_local_attn_weights = local_attn_weights * local_att_mask

        # attn_weights = global_attn_weights + masked_local_attn_weights
        attn_weights = global_attn_weights
        # a = 1

        if attn_mask is not None:
            raise NotImplementedError(f'attn_mask is not None not impl')
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        attn_weights = self.maybe_add_att_padding_mask(attn_weights, exp_key_padding_mask, bsz, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_weights *= local_att_mask

        # summall = attn_weights.sum(dim=-1, keepdim=True)
        # ifzero = torch.any(summall.abs() < 1e-2)
        # # assert not ifzero, f"contain zeros in sum"
        # if ifzero:
        #     print(f'contain zeros in sum, max_sum={summall.max()}, min_sum={summall.min()}')
        #     # raise ValueError
        #     # attn_weights /= summall
        # # else:
        # attn_weights /= summall

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights


# LearnableGlobalHardLocalMultiheadAttention
class LearnableGlobalHardLocalAvgMultiheadAttention(LearnableGlobalHardLocalMultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            # q, k, v = self.in_proj_qkv(query)
            # q, k, v, q_left, k_left, q_right, k_right, q_local, k_local = self.in_proj_all(query)
            q, k, v, q_left, k_left, q_right, k_right = self.in_proj_all(query)
        elif kv_same:
            # raise NotImplementedError(f'encoder-decoder attention not available')
            # encoder-decoder attention
            q = self.in_proj_global_q(query)
            q_left = self.in_proj_local_left_q(query)
            q_right = self.in_proj_local_right_q(query)
            # q_local = self.in_proj_local_q(query)
            if key is None:
                assert value is None
                k = v = None
                k_left = k_right = k_local = None
                # raise NotImplementedError(f'kv_same but key is None')
            else:
                k, v = self.in_proj_global_kv(key)
                k_left = self.in_proj_local_left_k(key)
                k_right = self.in_proj_local_right_k(key)
                # k_local = self.in_proj_local_k(key)
        else:
            # something else
            # raise NotImplementedError(f'This case should not be here!')
            q = self.in_proj_global_q(query)
            k = self.in_proj_global_k(key)
            v = self.in_proj_global_v(value)
            q_left = self.in_proj_local_left_q(query)
            k_left = self.in_proj_local_left_k(key)
            q_right = self.in_proj_local_right_q(query)
            k_right = self.in_proj_local_right_k(key)
            # q_local = self.in_proj_local_q(query)
            # k_local = self.in_proj_local_k(key)

        q *= self.scaling
        # q_local *= self.scaling
        # q_left *= self.scaling
        # q_right *= self.scaling

        if self.bias_k is not None:

            # assert self.bias_v is not None
            # k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            # v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            # if attn_mask is not None:
            #     attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            # if key_padding_mask is not None:
            #     key_padding_mask = torch.cat(
            #         [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
            raise NotImplementedError(f'bias_k not None not impl')

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # if k_local is not None:
        #     k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if k_left is not None:
            k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k_right is not None:
            k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if q_left is not None:
            q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if q_right is not None:
            q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q: [b*h, l, d]
        # k: [b*h, l, d]
        # k_local: [b*h, l, d]
        # v: [b*h, l, d]

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            # raise NotImplementedError(f'saved_state not impl for cross or decoder attention yet!')
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            # if 'prev_key_local' in saved_state:
            #     prev_key_local = saved_state['prev_key_local'].view(bsz * self.num_heads, -1, self.head_dim)
            #     if static_kv:
            #         k_local = prev_key_local
            #     else:
            #         k_local = torch.cat((prev_key_local, k_local), dim=1)
            if 'prev_key_local_left' in saved_state:
                prev_key_local_left = saved_state['prev_key_local_left'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_left = prev_key_local_left
                else:
                    k_left = torch.cat((prev_key_local_left, k_left), dim=1)
            if 'prev_key_local_right' in saved_state:
                prev_key_local_right = saved_state['prev_key_local_right'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k_right = prev_key_local_right
                else:
                    k_right = torch.cat((prev_key_local_right, k_right), dim=1)

            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            # saved_state['prev_key_local'] = k_local.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_left'] = k_left.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_local_right'] = k_right.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        # assert src_len == k_local.size(1), f'{src_len} != {k_local.size(1)}'
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:

            # src_len += 1
            # k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            # v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            # if attn_mask is not None:
            #     attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            # if key_padding_mask is not None:
            #     key_padding_mask = torch.cat(
            #         [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
            raise NotImplementedError(f'add_zero_attn not impl')

        # TODO: starting dot product
        exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) if key_padding_mask is not None else None

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        # local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        assert_size = [bsz * self.num_heads, tgt_len, src_len]
        assert list(global_attn_weights.size()) == assert_size, f'{global_attn_weights.size()} != {assert_size}'
        # assert list(local_attn_weights.size()) == assert_size, f'{local_attn_weights.size()} != {assert_size}'

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, exp_key_padding_mask, bsz)

        # masked_local_attn_weights = local_attn_weights * local_att_mask

        # attn_weights = global_attn_weights + masked_local_attn_weights
        attn_weights = global_attn_weights
        # a = 1

        if attn_mask is not None:

            # attn_mask = attn_mask.unsqueeze(0)
            # if self.onnx_trace:
            #     attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            # attn_weights += attn_mask
            raise NotImplementedError(f'attn_mask is not None not impl')

        attn_weights = self.maybe_add_att_padding_mask(attn_weights, exp_key_padding_mask, bsz, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        attn_weights += local_att_mask
        attn_weights /= 2

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # summall = attn_weights.sum(dim=-1, keepdim=True)
        # ifzero = torch.any(summall.abs() < 1e-2)
        # # assert not ifzero, f"contain zeros in sum"
        # if ifzero:
        #     print(f'contain zeros in sum, max_sum={summall.max()}, min_sum={summall.min()}')
        #     # raise ValueError
        #     # attn_weights /= summall
        # # else:
        # attn_weights /= summall

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights



def lgl_v2_compute_mask2localmask(left_softmax, right_softmax, triu):
    triu_t = triu.transpose(1, 2)
    left_mask = torch.matmul(left_softmax, triu)
    right_mask = torch.matmul(right_softmax, triu_t)
    bw_left_mask = torch.matmul(left_softmax, triu_t)
    bw_right_mask = torch.matmul(right_softmax, triu)

    fw_mask = left_mask * right_mask
    bw_mask = bw_left_mask * bw_right_mask

    local_sum_mask = fw_mask + bw_mask
    local_prod_mask = fw_mask * bw_mask
    local_mask = local_sum_mask - local_prod_mask
    return local_mask


def run(d, gb):
    x = torch.Tensor(d, d).uniform_()
    triu = torch.ones(d, d).triu_().int()
    eye = torch.eye(d).int()
    future_mask = (triu - eye).byte()
    past_mask = future_mask.transpose(0, 1)
    left_attn_weights = x.masked_fill(future_mask, float('-inf'))
    right_attn_weights = x.masked_fill(past_mask, float('-inf'))
    left_softmax = F.softmax(gb * left_attn_weights, dim=-1)
    right_softmax = F.softmax(gb * right_attn_weights, dim=-1)
    triu_t = triu.transpose(0, 1)
    left_mask = torch.matmul(left_softmax, triu.float())
    right_mask = torch.matmul(right_softmax, triu_t.float())
    return left_mask * right_mask


# x = abc






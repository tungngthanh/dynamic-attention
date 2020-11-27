import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import multihead_attention

DEBUG = False

def maybe_print(s):
    if DEBUG:
        print('lgl_attention.py:: ' + s)


GPU = torch.device("cuda")
GPU0 = torch.device("cuda:0")


class RelativePositionMultiheadAttention(nn.Module):
    max_relative_position=16
    NUM_WEIGHTS = 3
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(
            # needs:
            # 2 for global q k
            # 1 for values
            #
            self.NUM_WEIGHTS * embed_dim, embed_dim
        ))
        self.embedding_relative_k=nn.Embedding(2*self.max_relative_position+1, self.head_dim)
        self.embedding_relative_v = nn.Embedding(2*self.max_relative_position+1, self.head_dim)
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
        # nn.init.xavier_uniform_(self.embedding_relative_k)
        # nn.init.xavier_uniform_(self.embedding_relative_v)
        self.embedding_relative_k.weight.data.uniform_()
        self.embedding_relative_v.weight.data.uniform_()

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

    def generate_relative_positions_embeddings(self,length, embedding_relative):
        """Generates tensor of size [length, length, head_dim]."""
        max_relative_position=self.max_relative_position
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length,1)
        distance_mat = range_mat - range_mat.transpose(0,1)
        distance_mat_clipped = torch.clamp(distance_mat, min= -max_relative_position, max= max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a relative
        # position difference.
        relative_positions_matrix = distance_mat_clipped + max_relative_position

        # Generates embedding for each relative position of dimension depth.
        embeddings = embedding_relative(relative_positions_matrix.cuda())
        # embedding:
        return embeddings

    def _relative_attention_inner_key(self,x, y, z):
        """Relative position-aware dot-product attention inner calculation.

		This batches matrix multiply calculations to avoid unnecessary broadcasting.

		Args:
		  x: Tensor with shape [batch_size * heads, length, depth].
		  y: Tensor with shape [batch_size * heads, length, depth].
		  z: Tensor with shape [length, length, depth].
		  transpose: Whether to transpose inner matrices of y and z. Should be true if
			  last dimension of x is depth, not length.

		Returns:
		  A Tensor with shape [batch_size, heads, length, length or depth].
		"""

        # xy_matmul is [batch_size * heads, length, length ]
        xy_matmul=torch.bmm(x, y.transpose(1, 2))
        # x_t_r is [length, batch_size * heads, depth]
        x_t_r = x.transpose(0, 1)
        # x_tz_matmul_r is [length, batch_size * heads, length]
        x_tz_matmul_r = torch.bmm(x_t_r,z.transpose(1,2))
        # x_tz_matmul_r_t is [batch_size* heads, length, length]
        x_tz_matmul_r_t = x_tz_matmul_r.transpose(0,1)
        return xy_matmul + x_tz_matmul_r_t

    def _relative_attention_inner_value(self,x, y, z):
        """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
          x: Tensor with shape [batch_size * heads, length, length].
          y: Tensor with shape [batch_size * heads, length, depth].
          z: Tensor with shape [length, length, depth].
          transpose: Whether to transpose inner matrices of y and z. Should be true if
              last dimension of x is depth, not length.

        Returns:
          A Tensor with shape [batch_size, heads, length, length or depth].
        """

        # xy_matmul is [batch_size * heads, length, depth ]
        xy_matmul = torch.bmm(x, y)
        # x_t_r is [length, batch_size * heads, length]
        x_t_r = x.transpose(0,1)
        # x_tz_matmul_r is [length, batch_size * heads, depth]
        x_tz_matmul_r = torch.bmm(x_t_r, z)
        # x_tz_matmul_r_t is [batch_size* heads, length, depth]
        x_tz_matmul_r_t = x_tz_matmul_r.transpose(0, 1)
        return xy_matmul + x_tz_matmul_r_t

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
            q, k, v = self.in_proj_qkv(query)
            # q, k, v, q_left, k_left, q_right, k_right, q_local, k_local = self.in_proj_all(query)
        elif kv_same:
            raise NotImplementedError(f'encoder-decoder attention not available')
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
            raise NotImplementedError(f'This case should not be here!')
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
        relation_k=self.generate_relative_positions_embeddings(tgt_len, self.embedding_relative_k)
        relation_v=self.generate_relative_positions_embeddings(tgt_len, self.embedding_relative_v)

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
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # q: [b*h, l, d]
        # k: [b*h, l, d]
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


            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
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

        # global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        global_attn_weights = self._relative_attention_inner_key(q,k,relation_k)

        assert_size = [bsz * self.num_heads, tgt_len, src_len]
        assert list(global_attn_weights.size()) == assert_size, f'{global_attn_weights.size()} != {assert_size}'
        # assert list(local_attn_weights.size()) == assert_size, f'{local_attn_weights.size()} != {assert_size}'

        attn_weights = global_attn_weights

        if attn_mask is not None:
            # raise NotImplementedError(f'attn_mask is not None not impl')
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        attn_weights = self.maybe_add_att_padding_mask(attn_weights, exp_key_padding_mask, bsz, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # TODO change to relative part here
        # attn = torch.bmm(attn_weights, v)
        attn = self._relative_attention_inner_value(attn_weights, v,relation_v)
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

class LocalnessMultiheadSelfAttention(nn.Module):
    #TODO here we implement central position prediction and query-specific window
    NUM_WEIGHTS = 3
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(
            # needs:
            # 2 for global q k
            # 1 for values
            #
            self.NUM_WEIGHTS * embed_dim, embed_dim
        ))

        #TODO declare parameter for localness model
        self.localness_weight = Parameter(torch.Tensor(
            #only need 1
            embed_dim, embed_dim
        ))

        self.localness_uP = Parameter(torch.Tensor(
            # only need 1
            1, self.head_dim
        ))
        self.localness_uD = Parameter(torch.Tensor(
            # only need 1
            1, self.head_dim
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
        nn.init.uniform_(self.localness_weight)
        nn.init.uniform_(self.localness_uP)
        nn.init.uniform_(self.localness_uD)

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
    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def in_proj_localness(self, input):
        weight = self.localness_weight
        bias = None
        return F.linear(input, weight, bias)

    def in_proj_localness_D(self, input):
        weight = self.localness_uD
        bias = None
        return F.linear(input, weight, bias)

    def in_proj_localness_P(self, input):
        weight = self.localness_uP
        bias = None
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
            q, k, v = self.in_proj_qkv(query)
            # q, k, v, q_left, k_left, q_right, k_right, q_local, k_local = self.in_proj_all(query)
        elif kv_same:
            raise NotImplementedError(f'encoder-decoder attention not available')
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
            raise NotImplementedError(f'This case should not be here!')
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
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # q: [b*h, l, d]
        # k: [b*h, l, d]
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


            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)

            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
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
        q_localness=self.in_proj_localness(query)
        q_localness=q_localness.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        p_localness = self.in_proj_localness_P(q_localness)
        z_localness = self.in_proj_localness_D(q_localness)
        # q_localness: [b*h, l, d]
        # p_localness: [b*h, l, 1]
        # z_localness: [b*h, l, 1]
        # Find original length of the sentence
        ori_key = key
        ori_src_len_, bsz_, embed_dim_ = ori_key.size()
        ori_src_len = torch.tensor(ori_src_len_, dtype=query.dtype, device=query.device)

        if key_padding_mask is None:
            localness_len = ori_src_len
        else:
            localness_len = (ori_src_len - key_padding_mask.type_as(query).sum(-1, keepdim=True)).unsqueeze(0).transpose(0,1)
            localness_len = localness_len.repeat(self.num_heads,1,1)
        # [b*h, 1, 1]
        p_center_localness=localness_len*p_localness.sigmoid()
        d_center_localness=localness_len*z_localness.sigmoid()
        range_matrix = torch.arange(1, tgt_len + 1, dtype=query.dtype,device=query.device
                                    ).unsqueeze(0).repeat(tgt_len, 1).unsqueeze(0
                                    ).repeat(bsz * self.num_heads, 1, 1)
        # g_localness= -2*(((range_matrix - p_center_localness) / d_center_localness)**2)
        # g_localness= -2*(range_matrix**2 ) / (d_center_localness**2)
        g_localness= -2*(((range_matrix - p_center_localness) / 5)**2)

        # q_localness: [b*h, l, d]
        # p_localness: [b*h, l, 1]
        global_attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert_size = [bsz * self.num_heads, tgt_len, src_len]
        assert list(global_attn_weights.size()) == assert_size, f'{global_attn_weights.size()} != {assert_size}'
        # assert list(local_attn_weights.size()) == assert_size, f'{local_attn_weights.size()} != {assert_size}'

        attn_weights = global_attn_weights + g_localness
        # attn_weights = global_attn_weights

        if attn_mask is not None:
            # raise NotImplementedError(f'attn_mask is not None not impl')
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
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

def run_check_localness():
    b=2
    l=3
    p=torch.Tensor(b,l,1).uniform_()
    z=torch.Tensor(b,l,1).uniform_()
    p_extend=p.repeat(1,1,l)
    z_extend=z.repeat(1,1,l)
    range_matrix=torch.arange(1,l+1).unsqueeze(0).repeat(l,1).unsqueeze(0).repeat(b,1,1).float()
    g=(range_matrix-p_extend)/z_extend
    print(p[0])
    print(z[0])
    print(g[0])

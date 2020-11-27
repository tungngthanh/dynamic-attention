import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,

)

from ..modules import (
    LearnableGlobalLocalMultiheadAttention,
    LearnableGlobalLocalMultiheadAttentionV2,
    LearnableGlobalLocalExpConMultiheadAttention,
    LearnableGlobalLocalExpConMultiheadAttentionV2,
    LearnableGlobalHardLocalMultiheadAttention,
    LearnableGlobalHardLocalAvgMultiheadAttention,
	LearnableGlobalLocalMultiheadAttentionSelfAttention,
	LearnableGlobalHardLocalMultiheadAttentionSelfAttention,
	LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock,
	LearnableGlobalLocalExpConMultiheadAttentionSelfAttention,
	LearnableGlobalHardLocalMultiheadAttentionSelfAttentionBlock,
    LearnableGlobalHardLocalExpConMultiheadAttentionSelfAttention,
    LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalization,
    LearnableGlobalHardLocalExpConMultiheadAttentionSelfAttentionNormalization,
    LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalizationBlock,
    LearnableGlobalLocalMultiheadAttentionDecoder,
    LearnableGlobalHardLocalMultiheadAttentionDecoder,
    LearnableGlobalHardLocalMultiheadAttentionNormalizationDecoder,
    LearnableGlobalLocalMultiheadAttentionDecoderBlock,
    LearnableGlobalHardLocalMultiheadAttentionDecoderBlock,
    LearnableGlobalHardLocalMultiheadAttentionNormalizationDecoderBlock,
    LearnableGlobalHardLocalMultiheadAttentionSelfAttentionAdditiveMasking,
    MultiheadAttentionCheckDecoder,
    RelativePositionMultiheadAttention,
    LocalnessMultiheadSelfAttention
)

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)

from fairseq.models import transformer


@register_model('lgl')
class LearnableGlobalLocalTransformerModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        # print(f'Encoder: left_pad_source={args.left_pad_source}')
        encoder = LearnableGlobalLocalTransformerEncoder(args, src_dict, encoder_embed_tokens,
                                                         left_pad=args.left_pad_source)
        decoder = LearnableGlobalLocalTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return LearnableGlobalLocalTransformerModel(encoder, decoder)


class LearnableGlobalLocalTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = transformer.PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LearnableGlobalLocalTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = transformer.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class LearnableGlobalLocalTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = transformer.Linear(input_embed_dim, embed_dim,
                                                 bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = transformer.PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LearnableGlobalLocalTransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = transformer.Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = transformer.LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class LearnableGlobalLocalTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.lgl_class = args.lgl_class
        self.encoder_att_class = args.encoder_att_class
        self.self_attn = self.encoder_att_class(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            # gumbel_alpha=args.gumbel_alpha,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = transformer.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = transformer.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([transformer.LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class LearnableGlobalLocalTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.encoder_att_class = args.encoder_att_class
        self.cross_att_class = args.cross_att_class
        self.decoder_att_class = args.decoder_att_class

        self.self_attn = self.decoder_att_class(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = transformer.LayerNorm(self.embed_dim)
        # self.lgl_class = args.lgl_class
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.cross_att_class(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
                # gumbel_alpha=args.gumbel_alpha
            )
            self.encoder_attn_layer_norm = transformer.LayerNorm(self.embed_dim)

        self.fc1 = transformer.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = transformer.Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = transformer.LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class LearnableGlobalLocalTransformerEncoderMix(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = transformer.PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LearnableGlobalLocalTransformerEncoderLayer(args)
            for i in range(args.num_mix_layers)
        ])
        self.layers.extend([
            transformer.TransformerEncoderLayer(args)
            for i in range(args.num_mix_layers,args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = transformer.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

@register_model('lgl_mix')
class LearnableGlobalLocalTransformerModelMix(LearnableGlobalLocalTransformerModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            ##TODO remember to change this back to default
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        # print(f'Encoder: left_pad_source={args.left_pad_source}')
        encoder = LearnableGlobalLocalTransformerEncoderMix(args, src_dict, encoder_embed_tokens,
                                                         left_pad=args.left_pad_source)
        decoder = LearnableGlobalLocalTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return LearnableGlobalLocalTransformerModel(encoder, decoder)

@register_model('lgl_mix_decoder')
class LearnableGlobalLocalTransformerModelMixDecoder(LearnableGlobalLocalTransformerModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        # print(f'Encoder: left_pad_source={args.left_pad_source}')
        encoder = LearnableGlobalLocalTransformerEncoder(args, src_dict, encoder_embed_tokens,
                                                         left_pad=args.left_pad_source)
        decoder = LearnableGlobalLocalTransformerDecoderMix(args, tgt_dict, decoder_embed_tokens)
        return LearnableGlobalLocalTransformerModel(encoder, decoder)

class LearnableGlobalLocalTransformerDecoderMix(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = transformer.Linear(input_embed_dim, embed_dim,
                                                 bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = transformer.PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LearnableGlobalLocalTransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.num_mix_layers)
        ])
        self.layers.extend([
            transformer.TransformerDecoderLayer(args, no_encoder_attn)
            for i in range(args.num_mix_layers, args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = transformer.Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = transformer.LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict

@register_model('lgl_mix_encoder_decoder')
class LearnableGlobalLocalTransformerModelMix(LearnableGlobalLocalTransformerModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        # print(f'Encoder: left_pad_source={args.left_pad_source}')
        encoder = LearnableGlobalLocalTransformerEncoderMix(args, src_dict, encoder_embed_tokens,
                                                         left_pad=args.left_pad_source)
        decoder = LearnableGlobalLocalTransformerDecoderMix(args, tgt_dict, decoder_embed_tokens)
        return LearnableGlobalLocalTransformerModel(encoder, decoder)

#======================================================================================================================================================================
@register_model('transformer_hard_local_lm')
class TransformerHardLocalLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', default=False, action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        transformer.base_lm_architecture(args)

        if hasattr(args, 'no_tie_adaptive_proj') and args.no_tie_adaptive_proj is False:
            # backward compatibility
            args.tie_adaptive_proj = True

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.dictionary), task.dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = transformer.Embedding(len(task.dictionary), args.decoder_input_dim, task.dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = LearnableGlobalLocalTransformerDecoder(
            args, task.output_dictionary, embed_tokens, no_encoder_attn=True, final_norm=False,
        )
        return TransformerHardLocalLanguageModel(decoder)

@register_model('transformer_relative_lm')
class TransformerRelativeLanguageModel(TransformerHardLocalLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', default=0., type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0., type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', default=False, action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', type=int, metavar='N', default=4,
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', type=int, metavar='N', default=2,
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        transformer.base_lm_architecture(args)

        if hasattr(args, 'no_tie_adaptive_proj') and args.no_tie_adaptive_proj is False:
            # backward compatibility
            args.tie_adaptive_proj = True

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.tokens_per_sample
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = args.tokens_per_sample

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.dictionary), task.dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = transformer.Embedding(len(task.dictionary), args.decoder_input_dim, task.dictionary.pad())

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = LearnableGlobalLocalTransformerDecoderMix(
            args, task.output_dictionary, embed_tokens, no_encoder_attn=True, final_norm=False,
        )
        return TransformerRelativeLanguageModel(decoder)

@register_model_architecture('transformer_hard_local_lm', 'transformer_full_hard_local_lm_base')
def transformer_full_hard_local_lm_base(args):
    transformer.base_lm_architecture(args)
    args.encoder_att_class = getattr(args, 'encoder_att_class', None)
    args.cross_att_class = getattr(args, 'cross_att_class', None)
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    # args.num_block = 5
    # args.num_mix_layers = 3

@register_model_architecture('transformer_hard_local_lm', 'transformer_relative_lm_base')
def transformer_relative_lm_base(args):
    transformer.base_lm_architecture(args)
    args.encoder_att_class = getattr(args, 'encoder_att_class', None)
    args.cross_att_class = getattr(args, 'cross_att_class', None)
    args.decoder_att_class = RelativePositionMultiheadAttention
    args.num_block = 5
    args.num_mix_layers = 3

@register_model_architecture('transformer_hard_local_lm', 'transformer_full_global_local_lm_base')
def transformer_full_global_local_lm_base(args):
    transformer.base_lm_architecture(args)
    args.encoder_att_class = getattr(args, 'encoder_att_class', None)
    args.cross_att_class = getattr(args, 'cross_att_class', None)
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    # args.num_block = 5
    # args.num_mix_layers = 3

#===========================================================================================================================================================================

def base_architecture(args):
    transformer.base_architecture(args)
    args.gumbel_alpha = getattr(args, 'gumbel_alpha', 1.0)

    args.lgl_class = getattr(args, 'lgl_class', LearnableGlobalLocalMultiheadAttention)

    args.encoder_att_class = getattr(args, 'encoder_att_class', LearnableGlobalLocalMultiheadAttention)
    args.cross_att_class = getattr(args, 'cross_att_class', LearnableGlobalLocalMultiheadAttention)
    args.decoder_att_class = getattr(args, 'decoder_att_class', MultiheadAttention)


@register_model_architecture('lgl', 'lgl_wmt_en_de')
def lgl_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture('lgl', 'lgl_alpha100_wmt_en_de')
def lgl_alpha100_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.gumbel_alpha = 100.0


@register_model_architecture('lgl', 'lgl_alpha10_wmt_en_de')
def lgl_alpha10_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.gumbel_alpha = 10.0


@register_model_architecture('lgl', 'lgl_alphasrclen_wmt_en_de')
def lgl_alphasrclen_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.gumbel_alpha = 'src_len'


# version 2
@register_model_architecture('lgl', 'lgl_v2_wmt_en_de')
def lgl_v2_wmt_en_de(args):
    base_architecture(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionV2
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionV2


@register_model_architecture('lgl', 'lgl_v2_alpha100_wmt_en_de')
def lgl_v2_alpha100_wmt_en_de(args):
    lgl_v2_wmt_en_de(args)
    args.gumbel_alpha = 100.0


@register_model_architecture('lgl', 'lgl_v2_alpha10_wmt_en_de')
def lgl_v2_alpha10_wmt_en_de(args):
    lgl_v2_wmt_en_de(args)
    args.gumbel_alpha = 10.0


@register_model_architecture('lgl', 'lgl_v2_alphasrclen_wmt_en_de')
def lgl_v2_alphasrclen_wmt_en_de(args):
    lgl_v2_wmt_en_de(args)
    args.gumbel_alpha = 'src_len'


# lgl with explicit conditions version 1
@register_model_architecture('lgl', 'lgl_expcon_wmt_en_de')
def lgl_expcon_wmt_en_de(args):
    base_architecture(args)
    args.encoder_att_class = LearnableGlobalLocalExpConMultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttention


@register_model_architecture('lgl', 'lgl_expcon_alpha100_wmt_en_de')
def lgl_expcon_alpha100_wmt_en_de(args):
    lgl_expcon_wmt_en_de(args)
    args.gumbel_alpha = 100.0


@register_model_architecture('lgl', 'lgl_expcon_alpha10_wmt_en_de')
def lgl_expcon_alpha10_wmt_en_de(args):
    lgl_expcon_wmt_en_de(args)
    args.gumbel_alpha = 10.0


@register_model_architecture('lgl', 'lgl_expcon_alphasrclen_wmt_en_de')
def lgl_expcon_alphasrclen_wmt_en_de(args):
    lgl_expcon_wmt_en_de(args)
    args.gumbel_alpha = 'src_len'


@register_model_architecture('lgl', 'lgl_expcon_v2_wmt_en_de')
def lgl_expcon_v2_wmt_en_de(args):
    base_architecture(args)
    args.encoder_att_class = LearnableGlobalLocalExpConMultiheadAttentionV2
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionV2


@register_model_architecture('lgl', 'lgl_hardlocal_wmt_en_de')
def lgl_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttention
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttention


@register_model_architecture('lgl', 'lgl_hardlocalavg_wmt_en_de')
def lgl_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalAvgMultiheadAttention
    args.cross_att_class = LearnableGlobalHardLocalAvgMultiheadAttention



# FIXME - Big Models ------------------------------------------------------------------------------------
@register_model_architecture('lgl', 'lgl_vaswani_wmt_en_de_big')
def lgl_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('lgl', 'lgl_wmt_en_de_big')
def lgl_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    lgl_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('lgl', 'lgl_wmt_en_de_big_t2t')
def lgl_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    lgl_vaswani_wmt_en_de_big(args)

# FIXME - ENCODER MODEL
@register_model_architecture('lgl', 'lgl_globallocal_encoder_wmt_en_de')
def lgl_globallocal_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_hardlocal_encoder_wmt_en_de')
def lgl_hardlocal_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_globallocal_block_encoder_wmt_en_de')
def lgl_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block=5

@register_model_architecture('lgl', 'lgl_expcon_globallocal_encoder_wmt_en_de')
def lgl_expcon_globallocal_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalExpConMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_expcon_hardlocal_encoder_wmt_en_de')
def lgl_expcon_hardlocal_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalExpConMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_hardlocal_block_encoder_wmt_en_de')
def lgl_hardlocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl', 'lgl_hardlocal_block_encoder_decoder_wmt_en_de')
def lgl_hardlocal_block_encoder_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionBlock
    args.num_block = 5

# @register_model_architecture('lgl', 'lgl_hardlocal_encoder_decoder_wmt_en_de')
# def lgl_hardlocal_encoder_decoder_wmt_en_de(args):
#     lgl_wmt_en_de(args)
#     args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
#     args.cross_att_class = MultiheadAttention
#     args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention


@register_model_architecture('lgl', 'lgl_hardlocal_encoder_normalize_wmt_en_de')
def lgl_hardlocal_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalization
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_expcon_hardlocal_encoder_normalize_wmt_en_de')
def lgl_expcon_hardlocal_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalExpConMultiheadAttentionSelfAttentionNormalization
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_hardlocal_block_encoder_normalize_wmt_en_de')
def lgl_hardlocal_block_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalizationBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl', 'lgl_hardlocal_decoder_wmt_en_de')
def lgl_hardlocal_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

@register_model_architecture('lgl', 'lgl_globallocal_decoder_wmt_en_de')
def lgl_globallocal_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
# @register_model_architecture('lgl', 'lgl_hardlocal_encoder_additive_masking_wmt_en_de')
# def lgl_hardlocal_encoder_additive_masking_wmt_en_de(args):
#     lgl_wmt_en_de(args)
#     args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionAdditiveMasking
#     args.cross_att_class = MultiheadAttention
#     args.decoder_att_class = MultiheadAttention

# @register_model_architecture('lgl', 'lgl_check_decoder_wmt_en_de')
# def lgl_check_decoder_wmt_en_de(args):
#     lgl_wmt_en_de(args)
#     args.encoder_att_class = MultiheadAttention
#     args.cross_att_class = MultiheadAttention
#     args.decoder_att_class = MultiheadAttentionCheckDecoder
# FIXME - ENCODER-DECODER_CROSS MODEL
@register_model_architecture('lgl', 'lgl_globallocal_all_wmt_en_de')
def lgl_globallocal_all_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    
# FIXME - ENCODER-DECODER MODEL
@register_model_architecture('lgl', 'lgl_globallocal_encoder_decoder_wmt_en_de')
def lgl_globallocal_encoder_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder

@register_model_architecture('lgl', 'lgl_hardlocal_encoder_decoder_wmt_en_de')
def lgl_hardlocal_encoder_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

@register_model_architecture('lgl', 'lgl_hardlocal_encoder_decoder_normalize_wmt_en_de')
def lgl_hardlocal_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalization
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionNormalizationDecoder

# FIXME - MIX MODEL
@register_model_architecture('lgl_mix', 'lgl_mix_encoder_hardlocal_encoder_wmt_en_de')
def lgl_mix_encoder_hardlocal_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix', 'lgl_mix_hardlocal_block_encoder')
def lgl_mix_hardlocal_block_encoder_normalize(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix', 'lgl_mix_encoder_hardlocal_encoder_normalize_wmt_en_de')
def lgl_mix_encoder_hardlocal_encoder_normalize_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalization
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix', 'lgl_mix_hardlocal_block_encoder_normalize')
def lgl_mix_hardlocal_block_encoder_normalize(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttentionNormalizationBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_encoder_wmt_en_de')
def lgl_mix_globallocal_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_block_encoder_wmt_en_de')
def lgl_mix_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5


@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_globallocal_wmt_en_de')
def lgl_mix_decoder_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_hardlocal_wmt_en_de')
def lgl_mix_decoder_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_decoder_wmt_en_de')
def lgl_mix_globallocal_encoder_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_hardlocal_encoder_decoder_wmt_en_de')
def lgl_mix_hardlocal_encoder_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

#FIXME DIFFERENT ENCODER DECODER MODEL
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_hardlocal_decoder_wmt_en_de')
def lgl_mix_globallocal_encoder_hardlocal_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_mix_layers= 3

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_block_encoder_hardlocal_decoder_wmt_en_de')
def lgl_mix_globallocal_block_encoder_hardlocal_decoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5
#FIXME - CROSS ATTENTION
@register_model_architecture('lgl', 'lgl_globallocal_cross_wmt_en_de')
def lgl_globallocal_cross_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_globallocal_block_cross_wmt_en_de')
def lgl_globallocal_block_cross_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
@register_model_architecture('lgl', 'lgl_hardlocal_cross_wmt_en_de')
def lgl_hardlocal_cross_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_cross_hardlocal_wmt_en_de')
def lgl_mix_cross_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_cross_globallocal_wmt_en_de')
def lgl_mix_cross_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_cross_globallocal_block_wmt_en_de')
def lgl_mix_cross_globallocal_block_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers=3

#TODO BASELINE
@register_model_architecture('lgl', 'lgl_self_attention_wmt_en_de')
def lgl_self_attention_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl', 'lgl_relative_position_self_attention_encoder_wmt_en_de')
def lgl_relative_position_self_attention_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = RelativePositionMultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
#
# @register_model_architecture('lgl_mix', 'lgl_mix_localness_self_attention_encoder_wmt_en_de')
# def lgl_mix_localness_self_attention_encoder_wmt_en_de(args):
#     lgl_wmt_en_de(args)
#     args.encoder_att_class = LocalnessMultiheadSelfAttention
#     args.cross_att_class = MultiheadAttention
#     args.decoder_att_class = MultiheadAttention

#FIXME ENCODER CROSS MODEL
@register_model_architecture('lgl', 'lgl_globallocal_block_encoder_globallocal_block_cross_wmt_en_de')
def lgl_globallocal_block_encoder_globallocal_block_cross_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_globallocal_cross_globallocal_block_encoder_wmt_en_de')
def lgl_mix_globallocal_cross_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_globallocal_block_cross_globallocal_block_encoder_wmt_en_de')
def lgl_mix_globallocal_block_cross_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_decoder', 'lgl_mix_hardlocal_cross_globallocal_block_encoder_wmt_en_de')
def lgl_mix_hardlocal_cross_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_block_encoder_globallocal_block_cross_wmt_en_de')
def lgl_mix_globallocal_block_encoder_globallocal_block_cross_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_cross_mix_globallocal_block_encoder_wmt_en_de')
def lgl_mix_globallocal_cross_mix_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_hardlocal_cross_mix_globallocal_block_encoder_wmt_en_de')
def lgl_mix_hardlocal_cross_mix_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_block_cross_mix_globallocal_block_encoder_wmt_en_de')
def lgl_mix_globallocal_block_cross_mix_globallocal_block_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
#TODO Run full mix model on all encoder, decoder, cross attention
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_de')
def lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_de')
def lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_de')
def lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5
    args.num_mix_layers =3

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_block_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_de')
def lgl_mix_globallocal_block_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_cross_mix_decoder_globallocal_wmt_en_de')
def lgl_mix_global_local_encoder_mix_globallocal_cross_mix_decoder_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_block_encoder_mix_globallocal_cross_mix_decoder_globallocal_wmt_en_de')
def lgl_mix_global_local_block_encoder_mix_globallocal_cross_mix_decoder_globallocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_block = 5

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_cross_mix_decoder_hardlocal_wmt_en_de')
def lgl_mix_global_local_encoder_mix_globallocal_cross_mix_decoder_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_block_encoder_mix_globallocal_cross_mix_decoder_hardlocal_wmt_en_de')
def lgl_mix_globallocal_block_encoder_mix_globallocal_cross_mix_decoder_hardlocal_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5

#TODO####################################################################################################
#TODO iwslt model
def base_architecture_iwslt(args):
    transformer.transformer_iwslt_de_en(args)
    args.share_all_embeddings=False
    args.gumbel_alpha = getattr(args, 'gumbel_alpha', 1.0)

    args.lgl_class = getattr(args, 'lgl_class', LearnableGlobalLocalMultiheadAttention)

    args.encoder_att_class = getattr(args, 'encoder_att_class', LearnableGlobalLocalMultiheadAttention)
    args.cross_att_class = getattr(args, 'cross_att_class', LearnableGlobalLocalMultiheadAttention)
    args.decoder_att_class = getattr(args, 'decoder_att_class', MultiheadAttention)


@register_model_architecture('lgl', 'lgl_iwslt_en_de')
def lgl_iwslt_en_de(args):
    base_architecture_iwslt(args)

#TODO BASELINE
@register_model_architecture('lgl', 'lgl_self_attention_encoder_iwslt_en_de')
def lgl_self_attention_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

@register_model_architecture('lgl_mix', 'lgl_mix_localness_self_attention_encoder_wmt_en_de')
def lgl_mix_localness_self_attention_encoder_wmt_en_de(args):
    lgl_wmt_en_de(args)
    args.encoder_att_class = LocalnessMultiheadSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_mix_layers = 3

#Global_local_block Encoder
@register_model_architecture('lgl', 'lgl_globallocal_block_encoder_iwslt_en_de')
def lgl_globallocal_block_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block=5

#Global_local Encoder
@register_model_architecture('lgl', 'lgl_globallocal_encoder_iwslt_en_de')
def lgl_globallocal_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention

#Mix_global_local_block_Encoder
@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_block_encoder_iwslt_en_de')
def lgl_mix_globallocal_block_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers = 2

#Mix_decoder_globallocal
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_globallocal_iwslt_en_de')
def lgl_mix_decoder_globallocal_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_mix_layers = 3

#Mix_decoder_hardlocal
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_hardlocal_iwslt_en_de')
def lgl_mix_decoder_hardlocal_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder

#Global_local_block_Cross
@register_model_architecture('lgl', 'lgl_globallocal_block_cross_iwslt_en_de')
def lgl_globallocal_block_cross_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

#Mix_global_local_block_Cross
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_cross_globallocal_iwslt_en_de')
def lgl_mix_cross_globallocal_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_mix_layers = 3

#Mix_globallocal_block_cross_mix_globallocal_block_encoder
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_globallocal_block_cross_globallocal_block_encoder_iwslt_en_de')
def lgl_mix_globallocal_block_cross_globallocal_block_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

#Mix_hardlocal_cross_mix_globallocal_block_encoder
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_hardlocal_cross_globallocal_block_encoder_iwslt_en_de')
def lgl_mix_hardlocal_cross_globallocal_block_encoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

#Mix_globallocal_encoder_hardlocal_decoder
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_hardlocal_decoder_iwslt_en_de')
def lgl_mix_globallocal_encoder_hardlocal_decoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
#Mix_globallocal_encoder_decoder
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_decoder_iwslt_en_de')
def lgl_mix_globallocal_encoder_decoder_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder

#Mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_iwslt_en_de')
def lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5
#Mix_globallocal_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_iwslt_en_de')
def lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_block = 5


#TODO WMT en-fr
#Global_local_block Encoder
@register_model_architecture('lgl', 'lgl_wmt_en_fr')
def lgl_wmt_en_fr(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)

@register_model_architecture('lgl_mix', 'lgl_mix_localness_self_attention_encoder_wmt_en_fr')
def lgl_mix_localness_self_attention_encoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LocalnessMultiheadSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_mix_layers = 3

@register_model_architecture('lgl', 'lgl_globallocal_block_encoder_wmt_en_fr')
def lgl_globallocal_block_encoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block=5

#Mix_global_local_block_Encoder
@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_block_encoder_wmt_en_fr')
def lgl_mix_globallocal_block_encoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers = 3

#Mix_decoder_globallocal
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_globallocal_wmt_en_fr')
def lgl_mix_decoder_globallocal_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_mix_layers = 3

#Mix_decoder_hardlocal
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_decoder_hardlocal_wmt_en_fr')
def lgl_mix_decoder_hardlocal_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_mix_layers = 3

#Global_local_block_Cross
@register_model_architecture('lgl', 'lgl_globallocal_block_cross_wmt_en_fr')
def lgl_globallocal_block_cross_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5

#Mix_global_local_block_Cross
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_cross_globallocal_wmt_en_fr')
def lgl_mix_cross_globallocal_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = MultiheadAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_mix_layers = 3

#Mix_globallocal_block_cross_mix_globallocal_block_encoder
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_globallocal_block_cross_globallocal_block_encoder_wmt_en_fr')
def lgl_mix_globallocal_block_cross_globallocal_block_encoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers = 3

#Mix_hardlocal_cross_mix_globallocal_block_encoder
@register_model_architecture('lgl_mix_decoder', 'lgl_mix_hardlocal_cross_globallocal_block_encoder_wmt_en_fr')
def lgl_mix_hardlocal_cross_globallocal_block_encoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalHardLocalMultiheadAttentionSelfAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers = 3

#Mix_globallocal_encoder_hardlocal_decoder
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_hardlocal_decoder_wmt_en_fr')
def lgl_mix_globallocal_encoder_hardlocal_decoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_mix_layers = 3
#Mix_globallocal_encoder_decoder
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_globallocal_encoder_decoder_wmt_en_fr')
def lgl_mix_globallocal_encoder_decoder_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_mix_layers = 3
#Mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_fr')
def lgl_mix_global_local_encoder_mix_globallocal_block_cross_mix_decoder_hardlocal_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttention
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalHardLocalMultiheadAttentionDecoder
    args.num_block = 5
    args.num_mix_layers = 3
#Mix_globallocal_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal
@register_model_architecture('lgl_mix_encoder_decoder', 'lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_fr')
def lgl_mix_global_local_block_encoder_mix_globallocal_block_cross_mix_decoder_globallocal_wmt_en_fr(args):
    lgl_wmt_en_fr(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.decoder_att_class = LearnableGlobalLocalMultiheadAttentionDecoder
    args.num_block = 5
    args.num_mix_layers = 3

#TODO tuning Global_local_block Encoder
@register_model_architecture('lgl', 'lgl_globallocal_block_encoder_tune_iwslt_en_de')
def lgl_globallocal_block_encoder_tune_iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block=10

#Mix_global_local_block_Encoder tuning
@register_model_architecture('lgl_mix', 'lgl_mix_globallocal_block_encoder_tune_iwslt_en_de')
def lgl_mix_globallocal_block_encoder_tune__iwslt_en_de(args):
    lgl_iwslt_en_de(args)
    args.encoder_att_class = LearnableGlobalLocalMultiheadAttentionSelfAttentionBlock
    args.cross_att_class = MultiheadAttention
    args.decoder_att_class = MultiheadAttention
    args.num_block = 5
    args.num_mix_layers=2
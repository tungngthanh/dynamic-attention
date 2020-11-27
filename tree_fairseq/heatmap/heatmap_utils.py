from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
try:
    import matplotlib
except Exception as e:
    print(f'matplotlib not found!')
    raise e

import torch
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import pprint

IMAGE_DECODE_LENGTH = 100

DPI = 80
FONT_MULTIPLIER = 3
CELL_FONT_MULTIPLIER = 2
FIG_MULTIPLIER = 2.0


def plot_attention_image(
        title,
        image,
        row_names=None,
        col_names=None,
        out_file=None,
        add_values=True,
        font_multiplier=FONT_MULTIPLIER,
        cell_multiplier=CELL_FONT_MULTIPLIER,
        show=False):
    figsize = np.round(np.array(image.T.shape) * FIG_MULTIPLIER).astype(np.int32).tolist()
    # figsize = np.round(np.array((len(col_names),len(row_names))) * FIG_MULTIPLIER).astype(np.int32).tolist()
    fig, ax = plt.subplots(
        dpi=DPI,
        figsize=figsize
        # frameon=False
    )

    # fontdict = {
    #     "size": font_multiplier * image.shape[0]
    # }
    # cell_fontdict = {
    #     "size": cell_multiplier * image.shape[0]
    # }
    fontdict = {
        "size": font_multiplier * len(col_names)
    }
    cell_fontdict = {
        "size": cell_multiplier * len(col_names)
    }
    change_image=image[1:len(row_names)+1,:len(col_names)]
    adjust_image=change_image/change_image.sum(-1, keepdims=True)
    # print(change_image)
    # print(adjust_image)
    # input()
    # adjust_image=np.concatenate((change_image[1:],change_image[0:1]))
    # im = ax.imshow(image[:len(row_names),:len(col_names)])
    # im = ax.imshow(image, cmap='YlGn')
    im = ax.imshow(adjust_image, cmap='YlGn')
    if row_names is not None:
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names, fontdict=fontdict)
    if col_names is not None:
        ax.set_xticks(np.arange(len(col_names)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(col_names, fontdict=fontdict)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # len_row = image.shape[0]
    # len_col = image.shape[1]
    len_row=len(row_names)
    len_col=len(col_names)
    add_values=False
    if add_values:
        # for i in range(len_row-1):
        #     for j in range(len_col):
        #         text = ax.text(
        #             j, i, "%.2f " % image[i, j], ha="center", va="center", color="w", fontdict=cell_fontdict)
        for i in range(len_row):
            for j in range(len_col):
                text = ax.text(
                    # j, i, "%.2f " % image[i+1, j], ha="center", va="center", color="b", fontdict=cell_fontdict)
                    j, i, "%.2f " % adjust_image[i, j], ha="center", va="center", color="b", fontdict=cell_fontdict)
        # for j in range(len_col):
        #     text = ax.text(
        #         j, len_row-1, "%.2f " % image[i+1, j], ha="center", va="center", color="b", fontdict=cell_fontdict)




    if len(title) > 100:
        title = title[:100] + "-\n" + title[100:]
    ax.set_title(title, fontdict=fontdict)
    fig.tight_layout()

    # plt.show()

    if out_file is not None:
        # plt.savefig(out_file, dpi=16 * (image.shape[0]))
        print(f'out: {out_file}')
        plt.savefig(out_file)

        # out_file_npz = "{}.npz".format(out_file)
        # np.savez(out_file_npz, image=image, cols=np.array(col_names), rows=np.array(row_names))


def save_merge_attention(hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    if torch.is_tensor(hypo_tokens):
        hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    if torch.is_tensor(src_tokens):
        src_tokens = src_tokens.cpu().numpy().tolist()

    assert isinstance(hypo_tokens, list)
    assert isinstance(src_tokens, list)

    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'[{index}] Attention shape: {attention.shape}, hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}], src_tokens({len(src_tokens)})=[{src_tokens}]')
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'Image {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_merge_attention_v2(hypo_str, src_str, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    # if torch.is_tensor(hypo_tokens):
    #     hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    # if torch.is_tensor(src_tokens):
    #     src_tokens = src_tokens.cpu().numpy().tolist()

    # assert isinstance(hypo_tokens, list)
    # assert isinstance(src_tokens, list)
    hypo_tokens = hypo_str.split()
    src_tokens = src_str.split()
    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'Image {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def token_string(dictionary, i, escape_unk=False):
    if i == dictionary.unk():
        return dictionary.unk_string(escape_unk)
    else:
        return dictionary[i]


def idx2tokens(tensor, dictionary):
    if torch.is_tensor(tensor) and tensor.dim() == 2:
        # return '\n'.join(self.string(t) for t in tensor)
        raise NotImplementedError(f'{tensor}')
    tokens = [token_string(dictionary, i) for i in tensor]
    return tokens


def save_merge_attention_v3(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()

    # if torch.is_tensor(hypo_tokens):
    #     hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    # if torch.is_tensor(src_tokens):
    #     src_tokens = src_tokens.cpu().numpy().tolist()

    # assert isinstance(hypo_tokens, list)
    # assert isinstance(src_tokens, list)
    # hypo_tokens = hypo_str.split()
    # src_tokens = src_str.split()
    hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)

    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'{model.__class__.__name__} {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_default_attention(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    # title = f'{merge_transformer.MergeTransformerModel.__class__.__name__} {index}'
    title = f'{model.__class__.__name__} {index}'

    shape = attention.shape
    rows = hypo_tokens
    cols = src_tokens
    # assert shape[0] == len(rows), f'{shape}, {len(rows)}'
    # assert shape[1] == len(cols), f'{shape}, {len(cols)}'

    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_attention_by_models(models, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    assert len(models) == 1, f'models only support 1'
    model = models[0]

    # if isinstance(model, merge_transformer.MergeTransformerModel):
    if model.__class__.__name__ == 'MergeTransformerModel':
        print(f'heat_map_by MergeTransformerModel')
        save_merge_attention_v3(model, hypo_tokens[:-1], src_tokens[:-1], attention, index, save_dir, src_dict, tgt_dict)
    else:
        print(f'heat_map_by Default')
        save_default_attention(model, hypo_tokens[:-1], src_tokens[:-1], attention, index, save_dir, src_dict, tgt_dict)


def save_agg_srcdict_histogram(models, src_agg_att, src_tokens, attention, save_dir, src_dict, tgt_dict):
    assert len(models) == 1, f'models only support 1'
    model = models[0]

    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    # hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)
    shape = attention.shape
    len_rows = shape[0]

    for i, token in enumerate(src_tokens):
        att = attention[:, i].tolist()
        src_agg_att[token] += att

    title = f'{model.__class__.__name__} Src histogram'

    positive_src_tokens = {k: v for k, v in src_agg_att.items() if len(v) > 0}

    positive_src_tokens = {k: sum(v) / len(v) for k, v in positive_src_tokens.items()}

    hist_tokens = [k for k in positive_src_tokens.keys()]
    hist_values = [v for k, v in positive_src_tokens.items()]
    index = np.arange(len(hist_tokens))
    y = np.array(hist_values)
    labels = hist_tokens

    # N_points = 100000

    # Generate a normal distribution, center at x=0 and y=5
    # x = np.random.randn(N_points)
    # y = .4 * x + np.random.randn(100000) + 5

    fig, ax = plt.subplots(dpi=DPI,
        figsize=(30, 30))

    # index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.8
    # error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, y, bar_width, alpha=opacity, color='b', label='Hist')

    # rects2 = ax.bar(index + bar_width, means_women, bar_width,
    #                 alpha=opacity, color='r',
    #                 yerr=std_women, error_kw=error_config,
    #                 label='Women')

    ax.set_xlabel('Token')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    # ax.set_xticks(index + bar_width / 2, rotation='vertical')
    # ax.set_xticklabels(labels)

    plt.xticks(index + bar_width / 2, labels, rotation='vertical')
    ax.legend()

    fig.tight_layout()
    # plt.show()
    out_file = os.path.join(save_dir, f'src_hist.png')
    plt.savefig(out_file)

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # plt.hist()
    # # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(x, bins=n_bins)
    # axs[1].hist(y, bins=n_bins)


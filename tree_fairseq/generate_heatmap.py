#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch
import os

from fairseq import bleu, options, progress_bar, tasks, utils
# from fairseq.tokenizer import Tokenizer
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
# from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module

from heatmap.sequence_scorer import SequenceScorer, SequenceAttentionEntropyScorer
from heatmap import heatmap_utils

MAXINFER = int(os.environ.get('MAXINFER', 100000000))


def add_heatmap_args(parser):
    group = parser.add_argument_group('Heat')

    group.add_argument('--image-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')

    # add_common_eval_args(group)
    # fmt: off
    # group.add_argument('--beam', default=5, type=int, metavar='N',
    #                    help='beam size')
    # group.add_argument('--nbest', default=1, type=int, metavar='N',
    #                    help='number of hypotheses to output')
    # group.add_argument('--max-len-a', default=0, type=float, metavar='N',
    #                    help=('generate sequences of maximum length ax + b, '
    #                          'where x is the source length'))
    # group.add_argument('--max-len-b', default=200, type=int, metavar='N',
    #                    help=('generate sequences of maximum length ax + b, '
    #                          'where x is the source length'))
    # group.add_argument('--min-len', default=1, type=float, metavar='N',
    #                    help=('minimum generation length'))
    # group.add_argument('--match-source-len', default=False, action='store_true',
    #                    help=('generations should match the source length'))
    # group.add_argument('--no-early-stop', action='store_true',
    #                    help=('continue searching even after finalizing k=beam '
    #                          'hypotheses; this is more correct, but increases '
    #                          'generation time by 50%%'))
    # group.add_argument('--unnormalized', action='store_true',
    #                    help='compare unnormalized hypothesis scores')
    # group.add_argument('--no-beamable-mm', action='store_true',
    #                    help='don\'t use BeamableMM in attention layers')
    # group.add_argument('--lenpen', default=1, type=float,
    #                    help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    # group.add_argument('--unkpen', default=0, type=float,
    #                    help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    # group.add_argument('--replace-unk', nargs='?', const=True, default=None,
    #                    help='perform unknown replacement (optionally with alignment dictionary)')
    # group.add_argument('--sacrebleu', action='store_true',
    #                    help='score with sacrebleu')
    # group.add_argument('--score-reference', action='store_true',
    #                    help='just score the reference translation')
    # group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
    #                    help='initialize generation by target prefix of given length')
    # group.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
    #                    help='ngram blocking such that this size ngram cannot be repeated in the generation')
    # group.add_argument('--sampling', action='store_true',
    #                    help='sample hypotheses instead of using beam search')
    # group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
    #                    help='sample from top K likely next words instead of all words')
    # group.add_argument('--sampling-temperature', default=1, type=float, metavar='N',
    #                    help='temperature for random sampling')
    # group.add_argument('--diverse-beam-groups', default=-1, type=int, metavar='N',
    #                    help='number of groups for Diverse Beam Search')
    # group.add_argument('--diverse-beam-strength', default=0.5, type=float, metavar='N',
    #                    help='strength of diversity penalty for Diverse Beam Search')
    group.add_argument('--histogram', action='store_true',
                       help='histogram')
    group.add_argument('--layer-att-entropy', action='store_true',
                       help='histogram')

    # fmt: on
    return group


def post_process_prediction(hypo_tokens, src_str, alignment, attention, align_dict, tgt_dict, remove_bpe):
    hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
    if align_dict is not None:
        hypo_str = utils.replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        # hypo_tokens = tokenizer.Tokenizer.tokenize(hypo_str, tgt_dict, add_if_not_exist=True)
        # hypo_tokens = tokenizer.Tokenizer.tokenize(hypo_str, tgt_dict, add_if_not_exist=True)
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    print(f'attention: {attention.shape}')
    return hypo_tokens, hypo_str, alignment, attention

# def post_process_prediction(hypo_tokens, src_str, alignment, align_dict, tgt_dict, remove_bpe):
#     from fairseq import tokenizer
#     hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
#     if align_dict is not None:
#         hypo_str = replace_unk(hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string())
#     if align_dict is not None or remove_bpe is not None:
#         # Convert back to tokens for evaluating with unk replacement or without BPE
#         # Note that the dictionary can be modified inside the method.
#         hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
#     return hypo_tokens, hypo_str, alignment


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    assert args.print_alignment

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    print(f'src_dict: {src_dict}')
    print(f'tgt_dict: {tgt_dict}, pad: {tgt_dict.pad()}')

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        # translator = SequenceScorer(models, task.target_dictionary)
        print(f'score: tgt_dict: {tgt_dict}, pad: {tgt_dict.pad()}')
        if args.layer_att_entropy:
            print(f'Sequence get layer entropies')

            translator = SequenceAttentionEntropyScorer(models, tgt_dict.pad())
        else:
            print(f'Sequence get attention')
            translator = SequenceScorer(models, tgt_dict.pad())
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam, minlen=args.min_len,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
            match_source_len=args.match_source_len, no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    src_indices = src_dict.indices
    src_agg_att = {k: [] for k in src_indices.keys()}

    entropies = []
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.score_reference:
            translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            )

        wps_meter = TimeMeter()
        for index, (sample_id, src_tokens, target_tokens, hypos) in enumerate(translations):
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                # hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                assert "attention" in hypo, f'{hypo.keys()}'
                hypo_tokens, hypo_str, alignment, attention = post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    attention=hypo['attention'].float().cpu() if hypo['attention'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                entropies.append(hypo['inner_att_entropy'])

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                        ))

                        if not args.layer_att_entropy:
                            heatmap_utils.save_attention_by_models(
                                models, hypo_tokens, src_tokens, attention, index, args.image_dir, src_dict, tgt_dict
                            )

                            if args.histogram:
                                heatmap_utils.save_agg_srcdict_histogram(
                                    models, src_agg_att, src_tokens, attention, args.image_dir, src_dict, tgt_dict
                                )

                # Score only the top hypothesis
                if has_target and i == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        # target_tokens = Tokenizer.tokenize(
                        #     target_str, tgt_dict, add_if_not_exist=True)
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += 1

            if index >= MAXINFER:
                break

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    if args.layer_att_entropy:
        total_entropies = torch.cat(entropies, 0)
        print(f'Entropies by layers: total_entropies: {total_entropies.size()}')
        mean_entropies = total_entropies.mean(dim=0)
        std_entropies = total_entropies.std(dim=0)
        print(f'Mean entropy by layers: mean:{mean_entropies}, std: {std_entropies}')




def cli_main():
    parser = options.get_generation_parser()
    add_heatmap_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config

from wenet.utils.context_filter import ContextFilter
from torch.nn.utils.rnn import pad_sequence

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring',
                            'attention_rescoring_with_context_filter',
                            'ctc_prefix_beam_search_with_context_filter',
                            'ctc_prefix_beam_search_with_spike_coldec'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--connect_symbol',
                        default='',
                        type=str,
                        help='used to connect the output characters')
    parser.add_argument('--context_mode',
                        default=4,
                        type=int,
                        help='context mode')
    parser.add_argument('--context_dic',
                        default='100',
                        type=str,
                        help='context dic')                    

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 30000
    test_conf['filter_conf']['min_length'] = 50
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['context_mode'] = args.context_mode
    dic = test_conf['context_dict'].split('/')
    dic[-1] = args.context_dic + '.dic'
    dic = '/'.join(dic)
    test_conf['context_dict'] = dic
    print("context_mode = ", test_conf['context_mode'])
    print("context_dict = ", test_conf['context_dict'])
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)


    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    all_time = 0
    all_context = 0
    time_cnt = 0

    context_list_recog = []
    if test_conf['context_mode'] == 2 or test_conf['context_mode'] == 3:
        if test_conf['context_mode'] == 2:
            context_list_file = test_conf['pad_conf']['context_list_valid']
        if test_conf['context_mode'] == 3:
            context_list_file = test_conf['pad_conf']['context_list_test'] 
        f = open(context_list_file)
        file_obj = f.readlines()
        for item in file_obj:
            context_list_recog.append(torch.tensor([int(id) for id in item.split()]))
        f.close()
    context_list_recog.insert(0, torch.tensor([0]))
    context_lengths_recog = torch.tensor([x.size(0) for x in context_list_recog],dtype=torch.int32)
    context_list_recog = pad_sequence(context_list_recog,
                                    batch_first=True,
                                    padding_value=-1)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths, context_list, context_lengths, context_label, context_label_lengths, context_decoder_label = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if test_conf['context_mode'] == 2 or test_conf['context_mode'] == 3:
                context_list = context_list_recog
                context_lengths = context_lengths_recog
            context_list = context_list.to(device)
            context_lengths = context_lengths.to(device)
            if args.mode == 'attention':
                hyps, _ = model.recognize(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps, _ = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp, _ = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'ctc_prefix_beam_search_with_spike_coldec':
                assert (feats.size(0) == 1)
                hyp, _ = model.ctc_prefix_beam_search_with_spike_coldec(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'ctc_prefix_beam_search_with_context_filter':
                assert (feats.size(0) == 1)
                hyp, _ = model.ctc_prefix_beam_search_with_context_filter(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                T1 = time.perf_counter()
                hyp, _ = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight)
                T2 = time.perf_counter()
                all_time += T2 - T1
                time_cnt += 1
                if time_cnt % 50 == 0:
                    print(time_cnt, "次解码  平均耗时", all_time * 1000 / time_cnt, "ms")
                    sys.stdout.flush()
                hyps = [hyp]
            elif args.mode == 'attention_rescoring_with_context_filter':
                assert (feats.size(0) == 1)
                T1 = time.perf_counter()
                hyp, _, filtered_context_list = model.attention_rescoring_with_context_filter(
                    feats,
                    feats_lengths,
                    context_list,
                    context_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight)
                all_context += len(filtered_context_list) - 1

                T2 = time.perf_counter()
                all_time += T2 - T1
                time_cnt += 1
                print("all cost ", 1000*(T2 - T1))
                if time_cnt % 10 == 0:
                    print(time_cnt, "次解码  平均耗时", all_time * 1000 / time_cnt, "ms")
                    print(time_cnt, "次解码  平均热词数量", all_context / time_cnt)
                    sys.stdout.flush()
                hyps = [hyp]
            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    if w == 1:
                        continue
                    content.append(char_dict[w])
                logging.info('{} {}'.format(key, args.connect_symbol.join(content)))
                fout.write('{} {}\n'.format(key, args.connect_symbol.join(content)))
                # with open("test.txt", "a") as f:
                #     f.write('{} {}\n'.format(key, str(context_filter.get_decoder_bias_list(20))[1:-1]))



if __name__ == '__main__':
    main()

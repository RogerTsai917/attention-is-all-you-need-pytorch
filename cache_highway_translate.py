''' Translate input text with trained model. '''

import os
import time
import math
import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.CacheHighWayModels import HighWayTransformer
from transformer.CacheHighWayTranslator import HighWayTranslator
from transformer.Cache import CacheVocabulary


def load_model(opt, device, cache_vocab_dict):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = HighWayTransformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        encoder_early_exit=opt.encoder_early_exit,
        decoder_early_exit=opt.decoder_early_exit,
        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        encoder_weight_sharing=model_opt.encoder_share_weight,
        decoder_weight_sharing=model_opt.decoder_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout,
        cache_vocab_dict=cache_vocab_dict).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def creat_count_early_exit_dict(n_layers):
    exit_layer_dict = {}
    for i in range(n_layers):
        exit_layer_dict[i+1] = 0
    return exit_layer_dict


def merge_two_dict(origin_dict, new_dict):
    for key in origin_dict.keys():
        origin_dict[key] += new_dict[key]
    return origin_dict


def encoder_layer(model_opt):
    total_flops = 0.0

    q = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_k
    k = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_k
    v = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_v
    total_flops += q + k + v

    ScaledDotProductAttention = model_opt.d_model * ((model_opt.n_head * model_opt.d_k) + (model_opt.n_head * model_opt.d_k) + (model_opt.n_head * model_opt.d_v))
    Attention_linear = model_opt.d_model * model_opt.n_head * model_opt.d_v * model_opt.d_model
    PositionwiseFeedForward = model_opt.d_model * ((model_opt.d_model * model_opt.d_inner_hid) + (model_opt.d_inner_hid * model_opt.d_model))
    total_flops += ScaledDotProductAttention + Attention_linear + PositionwiseFeedForward
    return total_flops


def decoder_layer(model_opt):
    total_flops = 0.0

    q = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_k
    k = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_k
    v = model_opt.d_model * model_opt.d_model * model_opt.n_head * model_opt.d_v
    total_flops += q + k + v

    ScaledDotProductAttention = model_opt.d_model * ((model_opt.n_head * model_opt.d_k) + (model_opt.n_head * model_opt.d_k) + (model_opt.n_head * model_opt.d_v))
    Attention_linear = model_opt.d_model * model_opt.n_head * model_opt.d_v * model_opt.d_model
    PositionwiseFeedForward = model_opt.d_model * ((model_opt.d_model * model_opt.d_inner_hid) + (model_opt.d_inner_hid * model_opt.d_model))
    total_flops += 2 * (ScaledDotProductAttention + Attention_linear + PositionwiseFeedForward)
    return total_flops


def predict_layer(model_opt):
    return model_opt.d_model * model_opt.n_head * model_opt.d_k * model_opt.trg_vocab_size

def calculate_FLOPs(model_opt, encoder_exit_layer, decoder_exit_layer_dict, cache_vocab_dict):
    total_flops = 0.0
    encoder_layer_FLOPs = encoder_layer(model_opt)
    decoder_layer_FLOPs = decoder_layer(model_opt)
    predict_layer_FLOPS = predict_layer(model_opt)

    total_flops += encoder_exit_layer * encoder_layer_FLOPs
    for key in decoder_exit_layer_dict.keys():
        total_flops += decoder_exit_layer_dict[key] * decoder_layer_FLOPs 
        total_flops += predict_layer_FLOPS

    return total_flops


def add_lsit_to_dict(_list, _dict):
    for value in _list:
        if value not in _dict:
            _dict[value] = 1
        else:
            _dict[value] += 1
    return _dict


def perpare_cache_vocab(opt):
    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    print('[Info] Get vocabulary size:', len(TRG.vocab))

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}
    train = Dataset(examples=data['train'], fields=fields)
    unk_idx = SRC.vocab.stoi[SRC.unk_token]

    words_frequency = {}
    for example in tqdm(train, mininterval=0.1, desc='  - (train)', leave=False):
        sentence = [Constants.BOS_WORD] + example.trg + [Constants.EOS_WORD]
        sentence = [TRG.vocab.stoi.get(word, unk_idx) for word in sentence]
        words_frequency = add_lsit_to_dict(sentence, words_frequency)

        sentence = example.src
        sentence = [SRC.vocab.stoi.get(word, unk_idx) for word in sentence]
        words_frequency = add_lsit_to_dict(sentence, words_frequency)

    sorted_words_frquency = dict(sorted(words_frequency.items(), key=lambda item: item[1], reverse=True))
    print("len(sorted_words_frquency):", len(sorted_words_frquency))

    len_ = math.pow(len(sorted_words_frquency), 1.0/6)

    cache_vocab_0 = CacheVocabulary(TRG, sorted_words_frquency, 5000, Constants.UNK_WORD, Constants.PAD_WORD)
    cache_vocab_1 = CacheVocabulary(TRG, sorted_words_frquency, 6000, Constants.UNK_WORD, Constants.PAD_WORD)
    cache_vocab_2 = CacheVocabulary(TRG, sorted_words_frquency, 7000, Constants.UNK_WORD, Constants.PAD_WORD)
    cache_vocab_3 = CacheVocabulary(TRG, sorted_words_frquency, 8000, Constants.UNK_WORD, Constants.PAD_WORD)
    cache_vocab_4 = CacheVocabulary(TRG, sorted_words_frquency, 9000, Constants.UNK_WORD, Constants.PAD_WORD)
    
    result_dict = {
                0: cache_vocab_0,
                1: cache_vocab_1,
                2: cache_vocab_2,
                3: cache_vocab_3,
                4: cache_vocab_4}
    return result_dict, TRG


def main(similarity=1.0, entropy=0.0):
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-save_folder', required=True)
    parser.add_argument('-beam_size', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-encoder_early_exit', action='store_true')
    parser.add_argument('-decoder_early_exit', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    cache_vocab_dict, TRG = perpare_cache_vocab(opt)
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = HighWayTranslator(
        model=load_model(opt, device, cache_vocab_dict),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    # set the early exit threshold
    translator.model.encoder.set_early_exit_similarity(similarity)
    translator.model.decoder.set_early_exit_entropy(entropy)

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']
    n_layers=model_opt.n_layers
    tatoal_encoder_exit_layer_dict = creat_count_early_exit_dict(n_layers)
    tatoal_decoder_exit_layer_dict = creat_count_early_exit_dict(n_layers)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    output_file_name = os.path.join(opt.save_folder, "prediction_similarity_" + str(similarity) + "_entropy_" + str(entropy) + ".txt")
    with open(output_file_name, 'w') as f:
        total_encoder_words = 0
        total_deocder_words = 0
        total_FLOPs = 0

        start_time = time.time()
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            total_encoder_words += len(src_seq)
            pred_seq, encoder_exit_layer, decoder_exit_layer_dict = translator.translate_sentence(torch.LongTensor([src_seq]).to(device), n_layers, cache_vocab_dict, TRG)
            # total_FLOPs += calculate_FLOPs(model_opt, encoder_exit_layer, decoder_exit_layer_dict, cache_vocab_dict)
            # print(pred_seq)
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            # print(pred_line)
            total_deocder_words += len(pred_line.split(" ")) - 2
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            f.write(pred_line.strip() + '\n')
            tatoal_encoder_exit_layer_dict[encoder_exit_layer] += 1
            tatoal_decoder_exit_layer_dict = merge_two_dict(tatoal_decoder_exit_layer_dict, decoder_exit_layer_dict)
        end_time = time.time()
    run_time = end_time - start_time
    print('[Info] Finished.')


    print("[Info] Predict finished with entropy: ", entropy)
    print("[Info] Encoder early exit dict: ", tatoal_encoder_exit_layer_dict)
    print("[Info] Decoder early exit dict: ", tatoal_decoder_exit_layer_dict)
    print("[Info] Total input words: ", total_encoder_words)
    print("[Info] Total time: ", run_time)
    print("[Info] Total predict words: ", total_deocder_words)
    print("[Info] Average predict a word time: ", run_time/total_deocder_words)
    # print("[Info] Total FLOPs: ", int(total_FLOPs/1000000), "M")
    # print("[Info] Average predict a word FLOPs: ", int(total_FLOPs/total_deocder_words/1000000), "M")

    output_record_file_name = os.path.join(opt.save_folder, "prediction_record.txt")
    with open(output_record_file_name, 'a') as f:
        f.write("Predict with similarity: " + str(similarity) + "\n")
        f.write("Predict with entropy: " + str(entropy) + "\n")
        f.write("Encoder early exit dict: " + str(tatoal_encoder_exit_layer_dict) + "\n")
        f.write("Decoder early exit dict: " + str(tatoal_decoder_exit_layer_dict) + "\n")
        f.write("Total input words: " + str(total_encoder_words) + "\n")
        f.write("Total time: " + str(run_time) + "\n")
        f.write("Total predict words: " + str(total_deocder_words) + "\n")
        f.write("Average predict a word time: " + str(run_time/total_deocder_words) + "\n")
        # f.write("Total FLOPs: " + str(int(total_FLOPs/1000000)) + "M" + "\n")
        # f.write("Average predict a word FLOPs: " + str(total_FLOPs/total_deocder_words/1000000) + "M" + "\n")
        f.write("\n")


if __name__ == "__main__":
    '''
    Usage: python hightway_translate.py -model model/base_early_exit/trained_highway.chkpt -data m30k_deen_shr.pkl -save_folder prediction/encoder_3_decoder_early_exit
    '''
    
    encoder_similarity = 1

    entropy_list = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5, 4.0]
    # entropy_list = [1.5]
    
    for entropy in entropy_list:   
        main(encoder_similarity, entropy)

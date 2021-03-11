''' Translate input text with trained model. '''

import os
import time
import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.HighWayModels import HighWayTransformer
from transformer.HighWayTranslator import HighWayTranslator


def load_model(opt, device):

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
        dropout=model_opt.dropout).to(device)

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

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = HighWayTranslator(
        model=load_model(opt, device),
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
        total_words = 0
        start_time = time.time()
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            pred_seq, encoder_exit_layer, decoder_exit_layer_dict = translator.translate_sentence(torch.LongTensor([src_seq]).to(device), n_layers)
            # print("pred_seq ", pred_seq)
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            # print("pred_line ", pred_line)
            total_words += len(pred_line.split(" ")) - 2
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
    print("[Info] Total time: ", run_time)
    print("[Info] Total predict words: ", total_words)
    print("[Info] average predict a word time: ", run_time/total_words)

    output_record_file_name = os.path.join(opt.save_folder, "prediction_record.txt")
    with open(output_record_file_name, 'a') as f:
        f.write("Predict with similarity: " + str(similarity) + "\n")
        f.write("Predict with entropy: " + str(entropy) + "\n")
        f.write("Encoder early exit dict: " + str(tatoal_encoder_exit_layer_dict) + "\n")
        f.write("Decoder early exit dict: " + str(tatoal_decoder_exit_layer_dict) + "\n")
        f.write("Total time: " + str(run_time) + "\n")
        f.write("Total predict words: " + str(total_words) + "\n")
        f.write("average predict a word time: " + str(run_time/total_words) + "\n")
        f.write("\n")


if __name__ == "__main__":
    '''
    Usage: python hightway_translate.py -model model/base_early_exit/trained_highway.chkpt -data m30k_deen_shr.pkl -save_folder prediction/encoder_3_decoder_early_exit
    '''
    
    encoder_similarity = 1

    entropy_list = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    
    for entropy in entropy_list:   
        main(encoder_similarity, entropy)

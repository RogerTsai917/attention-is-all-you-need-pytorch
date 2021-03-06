''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.CacheHighWayModels import HighWayTransformer, get_pad_mask, get_subsequent_mask, HighwayException
import transformer.Constants as Constants
from transformer.Cache import CacheVocabulary


def creat_count_early_exit_dict(n_layers):
    exit_layer_dict = {}
    for i in range(n_layers):
        exit_layer_dict[i+1] = 0
    return exit_layer_dict


class HighWayTranslator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(HighWayTranslator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask, translate=False):
        exit_layer =None
        trg_mask = get_subsequent_mask(trg_seq)
        try:
            dec_output, all_highway_exits, exit_layer, all_teacher_layers_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask, decoder_exit=True, translate=translate)
        except HighwayException as e:
            all_highway_exits = e.message
            seq_logit = all_highway_exits[-1]
            exit_layer = e.exit_layer

        if exit_layer != None:
            return F.softmax(seq_logit, dim=-1), exit_layer
        else:
            return F.softmax(self.model.trg_word_prj(dec_output), dim=-1), exit_layer


    def _get_init_state(self, src_seq, src_mask, translate):
        beam_size = self.beam_size

        try:
           enc_output, all_highway_exits, exit_layer, *_ = self.model.encoder(src_seq, src_mask, translate=translate)
        except HighwayException as e:
            all_highway_exits = e.message
            enc_output = all_highway_exits[-1]
            exit_layer = e.exit_layer

        dec_output, *_ = self._model_decode(self.init_seq, enc_output, src_mask)
        
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores, exit_layer


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step, cache_vocab_dict, TRG, exit_layer):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        
        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        
        # change best_k_idx by exit_layer and cache_vocab_dict
        if exit_layer != None:
            cache_word = cache_vocab_dict[exit_layer-1].value_word[best_k_idx.tolist()[0]]
            best_k_idx = TRG.vocab.stoi.get(cache_word, Constants.UNK_WORD)
            best_k_idx = torch.tensor([best_k_idx])

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, n_layers, cache_vocab_dict, TRG):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        exit_layer_dict = creat_count_early_exit_dict(n_layers)
        
        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores, encoder_exit_layer = self._get_init_state(src_seq, src_mask, translate=True)
            if encoder_exit_layer is None:
                encoder_exit_layer = n_layers

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output, exit_layer = self._model_decode(gen_seq[:, :step], enc_output, src_mask, translate=True)

                if exit_layer != None:
                    exit_layer_dict[exit_layer] += 1
                else:
                    exit_layer_dict[n_layers] += 1

                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step, cache_vocab_dict, TRG, exit_layer)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist(), encoder_exit_layer, exit_layer_dict

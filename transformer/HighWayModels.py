''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer, HighWayLayer

def entropy(x):
    """Calculate entropy of a pre-softmax logit Tensor"""
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B / A


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False, share_weight=True, early_exit_layer=None):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)

        if share_weight:
            for i in range(self.n_layers):
                enc_output, enc_slf_attn = self.layer_stack[0](enc_output, slf_attn_mask=src_mask)
                if early_exit_layer != None and i+1 == early_exit_layer:
                    break
        else:
            for layer_number, enc_layer in enumerate(self.layer_stack):
                enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
                enc_slf_attn_list += [enc_slf_attn] if return_attns else []
                if early_exit_layer != None and layer_number+1 == early_exit_layer:
                    break

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class HighWayDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # create highway layer
        self.highway = nn.ModuleList(
            [HighWayLayer(d_model, n_trg_vocab, bias=False)
            for _ in range(n_layers)])

        # create a entropy list for early exit threshold
        self.early_exit_entropy = [-1 for _ in range(n_layers)]

    def set_early_exit_entropy(self, x):
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x
    

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False, translate=False, share_weight=True):

        dec_slf_attn_list, dec_enc_attn_list, all_highway_exits  = [], [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for i in range(self.n_layers):
            if share_weight:
                dec_output, dec_slf_attn, dec_enc_attn = self.layer_stack[0](
                    dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            else:
                dec_output, dec_slf_attn, dec_enc_attn = self.layer_stack[i](
                   dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
             
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

            highway_seq_logit = self.highway[i](dec_output)
            all_highway_exits += [highway_seq_logit]

            if not self.training and translate:
                highway_entropy = entropy(highway_seq_logit[0])[-1]
                if highway_entropy < self.early_exit_entropy[i]:
                    raise HighwayException(all_highway_exits, i + 1)

        if return_attns:
            return dec_output, all_highway_exits, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, all_highway_exits, None

class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!

class HighWayTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = HighWayDecoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        exit_layer = None

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, all_highway_exits, exit_layer, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
        # seq_logit = seq_logit.view(-1, seq_logit.size(2))

        # reshape the tensor and return
        return seq_logit, all_highway_exits, exit_layer
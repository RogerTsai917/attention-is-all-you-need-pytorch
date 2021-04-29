'''
This script handles the training process.
'''

import os
import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.CacheHighWayModels import HighWayTransformer
from transformer.Optim import ScheduledOptim
from transformer.Cache import CacheVocabulary

TRAIN_BASE = "train_base_model"
TRAIN_ENCODER = "train_encoder_early_exit"
TRAIN_DECODER_TEACHER = "train_decoder_teacher_layers"
TRAIN_DECODER = "train_decoder_early_exit"

def load_model(opt, device, cache_vocab_dict, training_mode):
    if (training_mode == TRAIN_ENCODER) or \
        (not opt.encoder_early_exit and training_mode == TRAIN_DECODER_TEACHER) or \
        (not opt.encoder_early_exit and not opt.decoder_teacher and training_mode == TRAIN_DECODER):
        file_name = os.path.join(opt.save_folder, (opt.save_model + '.chkpt'))
        checkpoint = torch.load(file_name, map_location=device)
    elif (training_mode == TRAIN_DECODER_TEACHER) or (not opt.decoder_teacher and training_mode == TRAIN_DECODER):
        file_name = os.path.join(opt.save_folder, (opt.save_model + '_encoder_highway.chkpt'))
        checkpoint = torch.load(file_name, map_location=device)
    elif training_mode == TRAIN_DECODER:
        file_name = os.path.join(opt.save_folder, (opt.save_model + '_decoder_teacher.chkpt'))
        checkpoint = torch.load(file_name, map_location=device)
    model_opt = checkpoint['settings']

    model = HighWayTransformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,
        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,
        encoder_early_exit=model_opt.encoder_early_exit,
        decoder_teacher=model_opt.decoder_teacher,
        decoder_early_exit=model_opt.decoder_early_exit,
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


def cal_encoder_student_performance(enc_output, all_highway_exits):
    enc_output = enc_output.view(-1, enc_output.size(2))

    total_loss = 0.0
    for early_exit_output in all_highway_exits:
        loss_function = nn.CosineEmbeddingLoss()
        early_exit_output = early_exit_output.view(-1, early_exit_output.size(2))
        loss = loss_function(enc_output, early_exit_output, Variable(torch.Tensor(enc_output.size(0)).cuda().fill_(1.0)))
        total_loss += loss
    return total_loss
        

def transform_gold_by_cache_vocab(gold, cache_vocab, TRG, trg_pad_idx):
    gold = gold.tolist()
    cache_gold = []
    for element in gold:
        word = TRG.vocab.itos[element]
        if word == Constants.PAD_WORD:
            cache_gold.append(trg_pad_idx)
        elif word not in cache_vocab.word_value:
            # # cache layer use <unk> to replace unknown word
            # cache_gold.append(cache_vocab.word_value[Constants.UNK_WORD])
            
            # cache layer use <blank> to replace unknown word
            cache_gold.append(trg_pad_idx)
        else:
            cache_gold.append(cache_vocab.word_value[word])
    return cache_gold


def cal_decoder_student_performance(gold, trg_pad_idx, all_highway_exits, cache_vocab_dict, TRG, device, smoothing=False):
    loss = cal_student_loss(gold, trg_pad_idx, all_highway_exits, cache_vocab_dict, TRG, device, smoothing=smoothing)
    
    n_correct, n_word = 0, 0
    for i in range(len(all_highway_exits)):
        early_exit_pred = all_highway_exits[i].view(-1, all_highway_exits[i].size(2))
        early_exit_gold = transform_gold_by_cache_vocab(gold, cache_vocab_dict[i], TRG, trg_pad_idx)
        early_exit_gold = torch.tensor(early_exit_gold).to(device)
        early_exit_gold = early_exit_gold.contiguous().view(-1)
        non_pad_mask = early_exit_gold.ne(trg_pad_idx)
        early_exit_pred = early_exit_pred.max(1)[1]
        n_correct += early_exit_pred.eq(early_exit_gold).masked_select(non_pad_mask).sum().item()
    
    n_correct = n_correct // len(all_highway_exits)
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_student_loss(gold, trg_pad_idx, all_highway_exits, cache_vocab_dict, TRG, device, smoothing=False):
    gold_view = gold.contiguous().view(-1)
    total_loss = 0.0
    for i in range(len(all_highway_exits)):
        early_exit_pred = all_highway_exits[i].view(-1, all_highway_exits[i].size(-1))
        early_exit_gold = transform_gold_by_cache_vocab(gold, cache_vocab_dict[i], TRG, trg_pad_idx)
        early_exit_gold = torch.tensor(early_exit_gold).to(device)
        early_exit_gold = early_exit_gold.contiguous().view(-1)
        
        if smoothing:
            eps = 0.1
            n_class = early_exit_pred.size(1)
            one_hot = torch.zeros_like(early_exit_pred).scatter(1, early_exit_gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(early_exit_pred, dim=1)
            non_pad_mask = early_exit_gold.ne(trg_pad_idx)
            
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later 
        else:
            loss = F.cross_entropy(early_exit_pred, early_exit_gold, ignore_index=trg_pad_idx, reduction='sum')
        
        total_loss += loss
    
    return total_loss


def cal_decoder_student_performance_with_teacher_layers(gold, trg_pad_idx, all_highway_exits, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False):
    loss = cal_decoder_student_loss_with_teacher_layers(gold, trg_pad_idx, all_highway_exits, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=smoothing)

    n_correct, n_word = 0, 0
    for i in range(len(all_highway_exits)):
        early_exit_pred = all_highway_exits[i].view(-1, all_highway_exits[i].size(2))
        early_exit_pred = early_exit_pred.max(1)[1]
        early_exit_gold = transform_gold_by_cache_vocab(gold, cache_vocab_dict[i], TRG, trg_pad_idx)
        early_exit_gold = torch.tensor(early_exit_gold).to(device)
        early_exit_gold = early_exit_gold.contiguous().view(-1)
        non_pad_mask = early_exit_gold.ne(trg_pad_idx)
        n_correct += early_exit_pred.eq(early_exit_gold).masked_select(non_pad_mask).sum().item()

    n_correct = n_correct // len(all_highway_exits)
    n_word = non_pad_mask.sum().item()
    
    return loss, n_correct, n_word


def cal_decoder_student_loss_with_teacher_layers(gold, trg_pad_idx, all_highway_exits, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False):
    gold_view = gold.contiguous().view(-1)
    total_loss = 0.0
    for i in range(len(all_highway_exits)):
        early_exit_pred = all_highway_exits[i].view(-1, all_highway_exits[i].size(-1))
        log_early_exit_pred = F.log_softmax(early_exit_pred, dim=1)

        teacher_pred = all_teacher_layers_output[i].view(-1, all_teacher_layers_output[i].size(-1))
        log_teacher_pred = F.softmax(teacher_pred, dim=1)

        early_exit_gold = transform_gold_by_cache_vocab(gold_view, cache_vocab_dict[i], TRG, trg_pad_idx)
        early_exit_gold = torch.tensor(early_exit_gold).to(device)
        early_exit_gold = early_exit_gold.contiguous().view(-1)
        non_pad_mask = early_exit_gold.ne(trg_pad_idx)
        
        loss = -(log_teacher_pred * log_early_exit_pred).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
        total_loss += loss
    return total_loss


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''
    pred = pred.view(-1, pred.size(2))
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(trg_pad_idx)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later 
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def cal_performance_with_teacher_layers(pred, gold, trg_pad_idx, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False):
    ''' Apply label smoothing if needed '''
    loss = cal_loss_with_teacher_layers(pred, gold, trg_pad_idx, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=smoothing)
    
    n_correct, n_word = 0, 0
    for i in range(len(all_teacher_layers_output)):
        teacher_pred = all_teacher_layers_output[i].view(-1, all_teacher_layers_output[i].size(2))
        teacher_pred = teacher_pred.max(1)[1]
        teacher_gold = transform_gold_by_cache_vocab(gold, cache_vocab_dict[i], TRG, trg_pad_idx)
        teacher_gold = torch.tensor(teacher_gold).to(device)
        teacher_gold = teacher_gold.contiguous().view(-1)
        non_pad_mask = teacher_gold.ne(trg_pad_idx)
        n_correct += teacher_pred.eq(teacher_gold).masked_select(non_pad_mask).sum().item()

    n_correct = n_correct // (len(all_teacher_layers_output) + 1)
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss_with_teacher_layers(pred, gold, trg_pad_idx, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    total_loss = 0.0
    for i in range(len(all_teacher_layers_output)):
        teacher_pred = all_teacher_layers_output[i].view(-1, all_teacher_layers_output[i].size(-1))
        teacher_gold = transform_gold_by_cache_vocab(gold, cache_vocab_dict[i], TRG, trg_pad_idx)
        teacher_gold = torch.tensor(teacher_gold).to(device)
        teacher_gold = teacher_gold.contiguous().view(-1)
        
        if smoothing:
            eps = 0.1
            n_class = teacher_pred.size(1)
            one_hot = torch.zeros_like(teacher_pred).scatter(1, teacher_gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(teacher_pred, dim=1)
            non_pad_mask = teacher_gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later 
        else:
            loss = F.cross_entropy(teacher_pred, teacher_gold, ignore_index=trg_pad_idx, reduction='sum')
        total_loss += loss
    
    return total_loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, cache_vocab_dict, TRG, smoothing, training_mode):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct, batch_count = 0, 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        if training_mode == TRAIN_BASE:
            pred = model(src_seq, trg_seq)
        elif training_mode == TRAIN_ENCODER:
            enc_output, all_highway_exits, _ = model(src_seq, trg_seq, encoder_exit=True)
        elif training_mode == TRAIN_DECODER_TEACHER:
            pred, all_teacher_layers_output = model(src_seq, trg_seq, decoder_teacher=True)
        elif training_mode == TRAIN_DECODER:
            if opt.decoder_teacher:
                decoder_all_highway_exits, all_teacher_layers_output = model(src_seq, trg_seq, decoder_teacher=True, decoder_exit=True)
            else:
               pred, decoder_all_highway_exits = model(src_seq, trg_seq, decoder_exit=True)


        # backward and update parameters
        if training_mode == TRAIN_BASE:
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=smoothing)
        
        elif training_mode == TRAIN_ENCODER:
            loss = cal_encoder_student_performance(enc_output, all_highway_exits)
        elif training_mode == TRAIN_DECODER_TEACHER:
            loss, n_correct, n_word = cal_performance_with_teacher_layers(
                    pred, gold, opt.trg_pad_idx, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=smoothing)           
        elif training_mode == TRAIN_DECODER:
            if opt.decoder_teacher:
                loss, n_correct, n_word = cal_decoder_student_performance_with_teacher_layers(
                    gold, opt.trg_pad_idx, decoder_all_highway_exits, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=smoothing)
            else:
               loss, n_correct, n_word = cal_decoder_student_performance(
                    gold, opt.trg_pad_idx, decoder_all_highway_exits, cache_vocab_dict, TRG, device, smoothing=smoothing)   
            
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        if training_mode == TRAIN_BASE or training_mode == TRAIN_DECODER_TEACHER or training_mode == TRAIN_DECODER:
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()
        else:
            batch_count +=1
            total_loss += loss.item()

    if training_mode == TRAIN_BASE or training_mode == TRAIN_DECODER_TEACHER or training_mode == TRAIN_DECODER:
        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy
    else:
        loss_per_batch = total_loss/batch_count
        return loss_per_batch, 0


def eval_epoch(model, validation_data, opt, device, cache_vocab_dict, TRG, training_mode):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct, batch_count = 0, 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            if training_mode == TRAIN_BASE:
                pred = model(src_seq, trg_seq)
            elif training_mode == TRAIN_ENCODER:
                enc_output, all_highway_exits, _ = model(src_seq, trg_seq, encoder_exit=True)
            elif training_mode == TRAIN_DECODER_TEACHER:
                pred, all_teacher_layers_output = model(src_seq, trg_seq, decoder_teacher=True)
            elif training_mode == TRAIN_DECODER:
                if opt.decoder_teacher:
                    decoder_all_highway_exits, all_teacher_layers_output = model(src_seq, trg_seq, decoder_teacher=True, decoder_exit=True)
                else:
                    pred, decoder_all_highway_exits = model(src_seq, trg_seq, decoder_exit=True)
                
            if training_mode == TRAIN_BASE:
                loss, n_correct, n_word = cal_performance(
                    pred, gold, opt.trg_pad_idx, smoothing=False)
            elif training_mode == TRAIN_ENCODER:
                loss = cal_encoder_student_performance(enc_output, all_highway_exits)
            elif training_mode == TRAIN_DECODER_TEACHER:
                loss, n_correct, n_word = cal_performance_with_teacher_layers(
                        pred, gold, opt.trg_pad_idx, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False)           
            elif training_mode == TRAIN_DECODER:
                if opt.decoder_teacher:
                    loss, n_correct, n_word = cal_decoder_student_performance_with_teacher_layers(
                        gold, opt.trg_pad_idx, decoder_all_highway_exits, all_teacher_layers_output, cache_vocab_dict, TRG, device, smoothing=False)
                else:
                    loss, n_correct, n_word = cal_decoder_student_performance(
                        gold, opt.trg_pad_idx, decoder_all_highway_exits, cache_vocab_dict, TRG, device, smoothing=False)

            # note keeping
            if training_mode == TRAIN_BASE or training_mode == TRAIN_DECODER_TEACHER or training_mode == TRAIN_DECODER:
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()
            else:
                batch_count += 1
                total_loss += loss.item()

    if training_mode == TRAIN_BASE or training_mode == TRAIN_DECODER_TEACHER or training_mode == TRAIN_DECODER:
        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy
    else:
        loss_per_batch = total_loss/batch_count
        return loss_per_batch, 0


def train(model, training_data, validation_data, device, opt, cache_vocab_dict, TRG, training_mode):
    ''' Start training '''

    log_train_file, log_valid_file = None, None

    try:
        os.makedirs(opt.save_folder)
    except FileExistsError:
        pass

    if opt.log:
        if training_mode == TRAIN_BASE:
            log_train_file = os.path.join(opt.save_folder, opt.log + '.train.base.log')
            log_valid_file = os.path.join(opt.save_folder, opt.log + '.valid.base.log')
        elif training_mode == TRAIN_ENCODER:
            log_train_file = os.path.join(opt.save_folder, opt.log + '.train.encoder.highway.log')
            log_valid_file = os.path.join(opt.save_folder, opt.log + '.valid.encoder.highway.log')
        elif training_mode == TRAIN_DECODER_TEACHER:
            log_train_file = os.path.join(opt.save_folder, opt.log + '.train.decoder.teacher.log')
            log_valid_file = os.path.join(opt.save_folder, opt.log + '.valid.decoder.teacher.log')
        elif training_mode == TRAIN_DECODER:
            log_train_file = os.path.join(opt.save_folder, opt.log + '.train.decoder.highway.log')
            log_valid_file = os.path.join(opt.save_folder, opt.log + '.valid.decoder.highway.log')


        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    no_decay = ["bias", "LayerNorm.weight"]
    if training_mode == TRAIN_BASE:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("encoder_highway" not in n) and ("decoder_highway" not in n) and (not any(nd in n for nd in no_decay))
                ],
                "weight_decay": opt.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("encoder_highway" not in n) and ("decoder_highway" not in n) and (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

    elif training_mode == TRAIN_ENCODER:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("encoder_highway" in n) and (not any(nd in n for nd in no_decay))
                ],
                "weight_decay": opt.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("encoder_highway" in n) and (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
    elif training_mode == TRAIN_DECODER_TEACHER:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("decoder_teacher_layers" in n) and (not any(nd in n for nd in no_decay))
                ],
                "weight_decay": opt.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("decoder_teacher_layers" in n) and (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
    elif training_mode == TRAIN_DECODER:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("decoder_highway" in n) and (not any(nd in n for nd in no_decay))
                ],
                "weight_decay": opt.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("decoder_highway" in n) and (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]
        

    optimizer = ScheduledOptim(
        optim.Adam(optimizer_grouped_parameters, betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, opt.n_warmup_steps)

    if training_mode == TRAIN_BASE:
        training_epoch = opt.base_epoch
    elif training_mode == TRAIN_ENCODER:
        training_epoch = opt.highway_encoder_epoch
    elif training_mode == TRAIN_DECODER_TEACHER:
        training_epoch = opt.teacher_decoder_epoch
    elif training_mode == TRAIN_DECODER:
        training_epoch = opt.highway_decoder_epoch

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(training_epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, cache_vocab_dict, TRG, smoothing=opt.label_smoothing, training_mode=training_mode)
        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, opt, device, cache_vocab_dict, TRG, training_mode)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                if training_mode == TRAIN_BASE:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu))
                elif training_mode == TRAIN_ENCODER:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_loss_{loss:3.3f}_encoder_highway.chkpt'.format(valid_loss))
                elif training_mode == TRAIN_DECODER:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_accu_{accu:3.3f}_decoder_highway.chkpt'.format(accu=100*valid_accu))
                torch.save(checkpoint, model_name)

            elif opt.save_mode == 'best':
                if training_mode == TRAIN_BASE:
                     model_name = os.path.join(opt.save_folder, opt.save_model + '.chkpt')
                elif training_mode == TRAIN_ENCODER:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_encoder_highway.chkpt')
                elif training_mode == TRAIN_DECODER_TEACHER:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_decoder_teacher.chkpt')
                elif training_mode == TRAIN_DECODER:
                    model_name = os.path.join(opt.save_folder, opt.save_model + '_decoder_highway.chkpt')
                   
                if training_mode == TRAIN_BASE and valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                elif training_mode == TRAIN_ENCODER or training_mode == TRAIN_DECODER_TEACHER or training_mode == TRAIN_DECODER:
                    torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


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
    print('[Info] vocabulary size:', len(TRG.vocab))

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
    print("[Info] sorted_words_frquency size:", len(sorted_words_frquency))

    # len_ = math.pow(len(sorted_words_frquency), 1.0/6)
    
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-base_epoch', type=int, default=10)
    parser.add_argument('-highway_encoder_epoch', type=int, default=10)
    parser.add_argument('-teacher_decoder_epoch', type=int, default=10)
    parser.add_argument('-highway_decoder_epoch', type=int, default=10)
    parser.add_argument('-train_b', '--train_batch_size', type=int, default=2048)
    parser.add_argument('-val_b', '--val_batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=0.0) # Weight deay if we apply some.

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-encoder_early_exit', action='store_true')
    parser.add_argument('-decoder_teacher', action='store_true')
    parser.add_argument('-decoder_early_exit', action='store_true')
    parser.add_argument('-encoder_share_weight', action='store_true')
    parser.add_argument('-decoder_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_folder', default="./")
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-seed', type=int, default=1024)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        raise

    if (opt.train_batch_size < 2048 or opt.val_batch_size < 2048) and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, device)
    else:
        raise

    #========= set seed =========#
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    #========= create cache vocab =========#
    cache_vocab_dict, TRG = perpare_cache_vocab(opt)

    print(opt)

    high_way_transformer = HighWayTransformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        encoder_early_exit=opt.encoder_early_exit,
        decoder_teacher=opt.decoder_teacher,
        decoder_early_exit=opt.decoder_early_exit,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        encoder_weight_sharing=opt.encoder_share_weight,
        decoder_weight_sharing=opt.decoder_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,  
        dropout=opt.dropout,
        cache_vocab_dict=cache_vocab_dict).to(device)

    train(high_way_transformer, training_data, validation_data, device, opt, cache_vocab_dict, TRG, training_mode=TRAIN_BASE)

    if opt.encoder_early_exit: 
        high_way_transformer = load_model(opt, device, cache_vocab_dict, training_mode=TRAIN_ENCODER)
        train(high_way_transformer, training_data, validation_data, device, opt, cache_vocab_dict, TRG, training_mode=TRAIN_ENCODER)

    if opt.decoder_teacher:
        high_way_transformer = load_model(opt, device, cache_vocab_dict, training_mode=TRAIN_DECODER_TEACHER)
        train(high_way_transformer, training_data, validation_data, device, opt, cache_vocab_dict, TRG, training_mode=TRAIN_DECODER_TEACHER)

    if opt.decoder_early_exit: 
        high_way_transformer = load_model(opt, device, cache_vocab_dict, training_mode=TRAIN_DECODER)
        train(high_way_transformer, training_data, validation_data, device, opt, cache_vocab_dict, TRG, training_mode=TRAIN_DECODER)



def prepare_dataloaders_from_bpe_files(opt, device):
    train_batch_size = opt.train_batch_size
    val_batch_size = opt.val_batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=train_batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=val_batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    train_batch_size = opt.train_batch_size
    val_batch_size = opt.val_batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=train_batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=val_batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
 
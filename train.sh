#!/bin/bash

python train_faster_encoder_decoder.py\
    -data_pkl m30k_deen_shr.pkl\
    -log m30k_deen_shr\
    -encoder_early_exit\
    -decoder_early_exit\
    -embs_share_weight\
    -proj_share_weight\
    -encoder_share_weight\
    -decoder_share_weight\
    -label_smoothing\
    -save_model trained\
    -train_b 128\
    -val_b 128\
    -warmup 128000\
    -base_epoch 400\
    -highway_encoder_epoch 200\
    -highway_decoder_epoch 200
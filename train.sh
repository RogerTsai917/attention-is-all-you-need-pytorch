#!/bin/bash

python train_cache_transformer.py\
    -data_pkl m30k_deen_shr.pkl\
    -log m30k_deen_shr\
    -decoder_teacher\
    -decoder_early_exit\
    -embs_share_weight\
    -proj_share_weight\
    -label_smoothing\
    -save_model trained\
    -train_b 128\
    -val_b 128\
    -warmup 128000\
    -base_epoch 400\
    -highway_decoder_epoch 300\
    -save_folder model/early_exit_base
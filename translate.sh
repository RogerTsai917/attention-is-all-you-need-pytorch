#!/bin/bash

python hightway_translate.py\
    -model model/two_way_early_exit_base/trained_decoder_highway.chkpt\
    -data m30k_deen_shr.pkl\
    -save_folder prediction/two_way_early_exit_base_origin_encoder\
    -encoder_early_exit\
    -decoder_early_exit
#!/bin/bash

python hightway_translate.py\
    -model model/encoder_early_decoder_early/trained_decoder_highway.chkpt\
    -data m30k_deen_shr.pkl\
    -save_folder prediction/encoder_early_decoder_early\
    -encoder_early_exit\
    -decoder_early_exit
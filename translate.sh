#!/bin/bash

python hightway_translate.py\
    -model model/smaller_early_exit_base_2/trained_decoder_highway.chkpt\
    -data m30k_deen_shr.pkl\
    -save_folder prediction/smaller_early_exit_base_2\
    -decoder_early_exit
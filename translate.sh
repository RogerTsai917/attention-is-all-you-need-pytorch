#!/bin/bash

python cache_highway_translate.py\
    -model model/cache_early_exit_inverted_triangle_with_nuk/trained_decoder_highway.chkpt\
    -data m30k_deen_shr.pkl\
    -save_folder prediction/cache_early_exit_inverted_triangle_with_nuk\
    -decoder_early_exit

import math
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset


class CacheVocabulary:
    def __init__(self, vocab_data, vocab_dict, cache_number, unknown_word):
        self.word_value = {}

        self.word_value[unknown_word] = 0
        count = 0
        for key in vocab_dict.keys():
            word = vocab_data.vocab.itos[key]
            if word != unknown_word:
                self.word_value[word] = count+1
                count += 1
                if count == cache_number:
                    break

        self.value_word = dict([(value, key) for key, value in self.word_value.items()])



def add_lsit_to_dict(_list, _dict):
    for value in _list:
        if value not in _dict:
            _dict[value] = 1
        else:
            _dict[value] += 1
    return _dict

data = pickle.load(open("m30k_deen_shr.pkl", 'rb'))
SRC, TRG = data['vocab']['src'], data['vocab']['trg']
print('[Info] Get merged vocabulary size:', len(TRG.vocab))

fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}
train = Dataset(examples=data['train'], fields=fields)

unk_idx = SRC.vocab.stoi[SRC.unk_token]

words_frequency = {}
for example in tqdm(train, mininterval=0.1, desc='  - (train)', leave=False):
    sentence = [Constants.BOS_WORD] + example.trg + [Constants.EOS_WORD]
    sentence = [TRG.vocab.stoi.get(word, unk_idx) for word in sentence]
    words_frequency = add_lsit_to_dict(sentence, words_frequency)

sorted_words_frquency = dict(sorted(words_frequency.items(), key=lambda item: item[1], reverse=True))
print(len(sorted_words_frquency))

len_ = math.pow(len(sorted_words_frquency), 1.0/6)
print(len(sorted_words_frquency))
print(len_)

cache_vocab = CacheVocabulary(TRG, sorted_words_frquency, 50, TRG.unk_token)

print(cache_vocab.word_value)
print()
print(cache_vocab.value_word)

print(TRG.vocab.itos[2])
print(TRG.vocab.itos[4])
print(TRG.vocab.itos[14])
print(TRG.vocab.itos[1])
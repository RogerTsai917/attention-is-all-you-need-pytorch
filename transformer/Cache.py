
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
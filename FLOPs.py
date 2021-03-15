
encoder_layers = 6
decoder_layers = 6
vocabulary_size = 9521
d_model = 512
d_inner_hid = 2048
n_head = 8
d_k = 64
d_v = 64
avg_encoder_input = 12

def encoder_layer(words_number):
    total_flops = 0.0

    q = words_number * d_model * n_head * d_k
    k = words_number * d_model * n_head * d_k
    v = words_number * d_model * n_head * d_v
    total_flops += q + k + v

    ScaledDotProductAttention = words_number * ((n_head * d_k) + (n_head * d_k) + (n_head * d_v))
    Attention_linear = words_number * n_head * d_v * d_model
    PositionwiseFeedForward = words_number * ((d_model * d_inner_hid) + (d_inner_hid * d_model))
    total_flops += ScaledDotProductAttention + Attention_linear + PositionwiseFeedForward
    return total_flops

def decoder_layer(words_number):
    total_flops = 0.0

    q = words_number * d_model * n_head * d_k
    k = words_number * d_model * n_head * d_k
    v = words_number * d_model * n_head * d_v
    total_flops += q + k + v

    ScaledDotProductAttention = words_number * ((n_head * d_k) + (n_head * d_k) + (n_head * d_v))
    Attention_linear = words_number * n_head * d_v * d_model
    PositionwiseFeedForward = words_number * ((d_model * d_inner_hid) + (d_inner_hid * d_model))
    total_flops += 2 * (ScaledDotProductAttention + Attention_linear + PositionwiseFeedForward)
    return total_flops
    

def predict_layer(words_number):
    return words_number * d_model * vocabulary_size



# total_flops = 0.0
# total = 0.0
# total_flops += encoder_layers * encoder_layer(512)
# for i in range(12):
#     total_flops += decoder_layers * decoder_layer(512)
#     total += predict_layer(512)
#     total_flops += predict_layer(512)

# print(int(total_flops/1000000), "M")
# print(int(total/1000000), "M")

print(encoder_layer(512)/1000000)
print(decoder_layer(512)/1000000)
print(predict_layer(512)/1000000)
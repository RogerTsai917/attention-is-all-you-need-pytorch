from bleu import file_bleu


def main(ref_file_name, hyp_file_name, entropy):
    sorce = file_bleu(ref_file_name, hyp_file_name)
    print("====================")
    print("[Info] entropy: ", entropy)
    print("BLEU sorce:", sorce)


if __name__ == "__main__":
    # entropy_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # entropy_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    entropy_list = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

    for entropy in entropy_list:
        ref_file_name = "test2016.en.txt" 
        hyp_file_name = "prediction_" + str(entropy) + ".txt"
        main(ref_file_name, hyp_file_name, entropy)
    
    # ref_file_name = "prediction_0.0.txt" 
    # hyp_file_name = "prediction_1.0.txt"
    # main(ref_file_name, hyp_file_name, 1.0)
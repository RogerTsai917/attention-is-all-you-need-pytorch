from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR", "CIDEr"])

def read_text_to_list(file_name):
    result_list = []
    with open(file_name, 'r') as fp:
        line = fp.readline()
        while line:
            line = line.replace("\n", "").replace("\r", "")
            line = line.replace(".", " .").replace(",", " ,")
            result_list.append(line)
            line = fp.readline()
    return result_list


def main(answer_file_name, predict_file_name, entropy):
    
    answer_list = read_text_to_list(answer_file_name)
    predict_list = read_text_to_list(predict_file_name)

    result = nlgeval.compute_metrics(ref_list=[answer_list],hyp_list=predict_list)
    print("====================")
    print("[Info] entropy: ", entropy)
    print("[Info] sorce:", result)


if __name__ == "__main__":

    predict_folder = "cache_early_exit_inverted_triangle_with_KD"

    encoder_similarity = 1

    entropy_list = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5, 4.0]
    # entropy_list = [5.0, 5.5, 6.0]
    

    for entropy in entropy_list:
        answer_file_name = "prediction/test2016.en.txt" 
        # predict_file_name = "prediction/" + "/prediction_" + str(entropy) + ".txt"
        predict_file_name = "prediction/" + predict_folder + "/prediction_similarity_" + str(encoder_similarity) + "_entropy_" + str(entropy) + ".txt"
        main(answer_file_name, predict_file_name, entropy)
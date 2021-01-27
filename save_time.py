
if __name__ == "__main__":

    base_line_time = 6.31

    compare_list = [ 7.33,
7.03,
6.67,
6.24,
6.05,
5.77,
5.62,
5.36,
5.09,
4.91,
4.57
]

    print("base line time: ", base_line_time)
    for comapre_time in compare_list:
        time_save = (base_line_time - comapre_time)/base_line_time*100

        print("time: ", comapre_time, " time save: ", time_save)

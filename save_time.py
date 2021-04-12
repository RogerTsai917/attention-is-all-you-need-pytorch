
if __name__ == "__main__":

    base_line_time = 6.47

    compare_list = [7.40,
7.29,
7.16,
7.08,
7.00,
7.03,
6.93,
6.85,
6.88,
6.78,
6.77,
]

    print("base line time: ", base_line_time)
    for comapre_time in compare_list:
        time_save = (base_line_time - comapre_time)/base_line_time*100

        print("time: ", comapre_time, " time save: ", time_save)

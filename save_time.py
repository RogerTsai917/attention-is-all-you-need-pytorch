
if __name__ == "__main__":

    base_line_time = 51.04

    compare_list = [51.04,
51.00,
51.09,
51.09,
50.83,
50.62,
50.45,
49.95,
49.11,
48.56,
47.73
]

    print("base line time: ", base_line_time)
    for comapre_time in compare_list:
        time_save = (base_line_time - comapre_time)/base_line_time*100

        print("time: ", comapre_time, " time save: ", time_save)

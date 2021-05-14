
if __name__ == "__main__":

    base_line_time = 18.95

    compare_list = [ 21.07,
20.39,
18.56,
17.31,
16.49,
15.58,
14.64,
13.62,
12.42,
11.11,
9.78,

]

    print("base line time: ", base_line_time)
    for comapre_time in compare_list:
        time_save = (base_line_time - comapre_time)/base_line_time*100

        print("time: ", comapre_time, " time save: ", time_save)

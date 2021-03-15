
if __name__ == "__main__":

    base_line_time = 75.32

    compare_list = [87.33,
85.98,
80.57,
77.48,
73.58,
71.32,
68.91,
67.69,
65.66,
63.48,
62.32
]

    print("base line time: ", base_line_time)
    for comapre_time in compare_list:
        time_save = (base_line_time - comapre_time)/base_line_time*100

        print("time: ", comapre_time, " time save: ", time_save)

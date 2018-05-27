import numpy as np

def log_arr_sum(array):
    log_sum = np.log(0)
    for x in array:
        log_sum = np.logaddexp(log_sum, x)
    return log_sum


def pprint_chart(chart):
    list_form = [[0 for i in range(len(chart[0]))] for j in range(len(chart))]
    for i, row in enumerate(chart):
        str_row = ""
        for j, element in enumerate(row):

            # list_form[i][j] = chart[i][j].cell_sum()
            str_row += " {:5.4f} ".format(chart[i][j].cell_sum())
        print(str_row)
    # print(to_ret)
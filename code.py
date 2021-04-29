from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utilities import *
import sys
import os


# Error function
def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


# K FOLD
'''
x_current_segment = x[left:right]
        y_current_segment = y[left:right]
        x_train, x_test = k_fold(x_current_segment, 1, 20)
        y_train, y_test = k_fold(y_current_segment, 1, 20)
        print(x_current_segment[1:])
        print(y_current_segment[1:])
        print(x_train)
        print(y_train)
'''
''' 
k_fold (y, 1, 20):
    n = 20
    test_set = y [0 : 1]
    train_set = y - test_set
        
'''


def k_fold(dataset, i, k):
    n = len(dataset)
    mask = np.ones(n, dtype=bool)
    test_set = dataset[n * (i - 1) // k:n * i // k]
    mask[n * (i - 1) // k:n * i // k] = False
    train_set = dataset[mask, ...]
    return np.array(train_set), np.array(test_set)


# functions:
def linear_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)

    x_e = np.column_stack((ones, x))
    test_x_e = np.column_stack((test_ones, test_x))

    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_2_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)

    x_e = np.column_stack((ones, x, x ** 2))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_3_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_4_least_squares_fit_predict(x, y, test_x):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_5_least_squares_fit_predict(x, y, test_x):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4, x ** 5))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4, test_x ** 5))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_6_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4, test_x ** 5, test_x ** 6))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def sin_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    test_x_e = np.column_stack((test_ones, np.sin(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def cos_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.cos(x)))
    test_x_e = np.column_stack((test_ones, np.cos(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def exp_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.exp(x)))
    test_x_e = np.column_stack((test_ones, np.exp(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


# # general function:
# def fit_and_error_linear(x, y, test_x):
#     # print(x)
#     # print(y)
#
#     linear = linear_least_squares_fit_predict(x, y, test_x)
#     print(x.max())
#     print(x.min())
#     new_xs = np.linspace(x.min(), x.max(), 100)
#     new_y_hat = linear[1] * new_xs + linear[0]
#     plt.scatter(x, y)
#     plt.plot(new_xs, new_y_hat, 'red')
#     # sinus = sine_least_squares_fit_predict(x, y)
#     print(linear)
#     # print(b)


# def fit_and_error_poly(x, y, test_x):
#     new_y_hat = poly_3_least_squares_fit_predict(x, y, test_x)
#     print(x.max())
#     print(x.min())
#     new_xs = np.linspace(x.min(), x.max(), 100)
#     plt.scatter(x, y)
#     plt.plot(new_xs, new_y_hat, 'red')
#     # sinus = sine_least_squares_fit_predict(x, y)
#     print(func)
#     # print(b)


def fit_and_error_poly_find_helper(train_x, test_x, train_y, test_y, order):
    assert (len(test_x) == 1)
    assert (len(test_y) == 1)

    if order == 1:
        new_y_hat, v = linear_least_squares_fit_predict(train_x, train_y, test_x)
    elif order == 2:
        new_y_hat, v = poly_2_least_squares_fit_predict(train_x, train_y, test_x)
    elif order == 3:
        new_y_hat, v = poly_3_least_squares_fit_predict(train_x, train_y, test_x)
    elif order == 4:
        new_y_hat, v = poly_4_least_squares_fit_predict(train_x, train_y, test_x)
    elif order == 5:
        new_y_hat, v = poly_5_least_squares_fit_predict(train_x, train_y, test_x)
    elif order == 6:
        new_y_hat, v = poly_6_least_squares_fit_predict(train_x, train_y, test_x)
    # sin
    elif order == 7:
        new_y_hat, v = sin_least_squares_fit_predict(train_x, train_y, test_x)
    # cos
    elif order == 8:
        new_y_hat, v = cos_least_squares_fit_predict(train_x, train_y, test_x)
    # exponential
    elif order == 9:
        new_y_hat, v = exp_least_squares_fit_predict(train_x, train_y, test_x)
    return square_error(test_y, new_y_hat)


def check_poly_orders(x, y):
    hashtable_order = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    hashtable_error = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    segments_number = len(x) // 20
    print("SEGMENTS NUMBER: " + str(segments_number))
    for i in range(0, segments_number):
        error = 999999999999999
        order = -1
        x_current_segment = x[20 * i:20 * (i + 1)]
        y_current_segment = y[20 * i:20 * (i + 1)]
        hashtable_error_current = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        print("Segment " + str(i))

        for j in range(1, 10):
            error_current = 0
            for batch in range(1, 21):
                x_train, x_test = k_fold(x_current_segment, batch, 20)
                y_train, y_test = k_fold(y_current_segment, batch, 20)
                error_current_batch = fit_and_error_poly_find_helper(x_train, x_test, y_train, y_test, j)
                # print("Error batch " + str(batch) + ": " + str(error_current_batch))

                error_current += error_current_batch

            error_current /= 20
            print(" Polynomial order " + str(j) + ": error.." + str(error_current))
            if error_current < error:
                error = error_current
                order = j
            hashtable_error_current[j] = hashtable_error[j] + error_current

        if order == 1:
            hashtable_error[1] = hashtable_error[1] + hashtable_error_current[1]
        elif order in [2, 3, 4, 5, 6]:
            for o in [2, 3, 4, 5, 6]:
                hashtable_error[o] = hashtable_error[o] + hashtable_error_current[o]
        else:
            for o in [7, 8, 9]:
                hashtable_error[o] = hashtable_error[o] + hashtable_error_current[o]

        hashtable_order[order] = hashtable_order[order] + 1
        # hashtable_error[order] = hashtable_error[order] + error

        print("BEST ORDER: " + str(order))
        print("BEST ERROR: " + str(error))

    return hashtable_error, hashtable_order


' SCRIPT TO FIND THE POLYNOM ORDER '


def find_poly_order():
    hashtable_order = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    hashtable_error = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    directory = '/Users/edwardgeorge/Documents/2nd year/data-driven/coursework/train_data'
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print("File: " + filename)
            x, y = load_points_from_file("train_data/" + filename)
            hashtable_error_file, hashtable_order_file = check_poly_orders(x, y)
            for it in range(1, 10):
                hashtable_order[it] = hashtable_order[it] + hashtable_order_file[it]
                hashtable_error[it] = hashtable_error[it] + hashtable_error_file[it]
    print(hashtable_order)
    print(hashtable_error)
    for order in range(1, 10):
        if hashtable_order[order] == 0:
            print("NO SCORE")
            continue
        print("ORDER " + str(order) + " MEAN ERROR SCORE: " + str(hashtable_error[order] / hashtable_order[order]))


def plot_predictions(file, file_x_plot, file_y_plot, x, y):
    plt.title("File: " + file)
    assert len(x) == len(y)
    assert len(x) % 20 == 0
    len_data = len(x)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(x, y, c=colour)
    # print(file_x_plot)
    # print(x)
    for i in range(0, num_segments):
        plt.plot(file_x_plot[i], file_y_plot[i], c='red')
    plt.show()


# if __name__ == '__main__':
#     # READ ARGUMENTS FOR FILE AND PLOT
#     file = sys.argv[1]
#     plot = False
#     if len(sys.argv) == 3:
#         if sys.argv[2] == "--plot":
#             plot = True
#         else:
#             print("ERROR")
#     print("PLOT: {}".format(plot))
#
#     x, y = load_points_from_file("train_data/" + file)
#     # view_data_segments(x, y)
#     segments_number = len(x) // 20
#     print(segments_number)
#     file_total_error = 0
#     file_x_plot = []
#     file_y_plot = []
#     print("File: " + file)
#     for i in range(0, segments_number):
#         left = 20 * i
#         right = 20 * (i + 1)
#         x_current_segment = x[left:right]
#         y_current_segment = y[left:right]
#
#         linear_mean_error = 0
#         poly_mean_error = 0
#         sin_mean_error = 0
#
#         for batch in range(1, 5):
#             x_train, x_test = k_fold(x_current_segment, batch, 4)
#             y_train, y_test = k_fold(y_current_segment, batch, 4)
#
#             y_hat_linear = linear_least_squares_fit_predict(x_train, y_train, x_test)
#             y_hat_poly = poly_3_least_squares_fit_predict(x_train, y_train, x_test)
#             y_hat_sin = sin_least_squares_fit_predict(x_train, y_train, x_test)
#
#             # assert (len(y_hat_linear) == 4)
#             # assert (len(y_hat_poly) == 4)
#             # assert (len(y_hat_sin) == 4)
#             # assert (len(y_test) == 4)
#
#             error_linear = square_error(y_test, y_hat_linear)
#             error_poly = square_error(y_test, y_hat_poly)
#             error_sin = square_error(y_test, y_hat_sin)
#
#             print("error_linear: " + str(error_linear))
#             print("error_poly: " + str(error_poly))
#             print("error_sin: " + str(error_sin))
#             print("-----------------------------")
#             linear_mean_error += error_linear
#             poly_mean_error += error_poly
#             sin_mean_error += error_sin
#
#         linear_mean_error /= 20
#         poly_mean_error /= 20
#         sin_mean_error /= 20
#
#
#         # y_hat_linear = linear_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
#         # y_hat_poly = poly_3_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
#         # y_hat_sin = sin_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
#         # assert (len(y_hat_linear) == 20)
#         # assert (len(y_hat_poly) == 20)
#         # assert (len(y_hat_sin) == 20)
#         # error_linear = square_error(y[left:right], y_hat_linear)
#         # error_poly = square_error(y[left:right], y_hat_poly)
#         # error_sin = square_error(y[left:right], y_hat_sin)
#         # print("error_linear: " + str(linear_mean_error))
#         # print("error_poly: " + str(poly_mean_error))
#         # print("error_sin: " + str(sin_mean_error))
#         # print("--------------------------------")
#         if linear_mean_error < poly_mean_error and linear_mean_error < sin_mean_error:
#             file_total_error += linear_mean_error
#             print("linear")
#             file_y_plot.append(linear_least_squares_fit_predict(x_current_segment, y_current_segment, x_current_segment))
#
#         elif poly_mean_error < linear_mean_error and poly_mean_error < sin_mean_error:
#             file_total_error += poly_mean_error
#             print("poly")
#             file_y_plot.append(poly_3_least_squares_fit_predict(x_current_segment, y_current_segment, x_current_segment))
#
#         else:
#             file_total_error += sin_mean_error
#             print("sin")
#             file_y_plot.append(sin_least_squares_fit_predict(x_current_segment, y_current_segment, x_current_segment))
#
#         file_x_plot.append(x_current_segment)
#
#     if plot:
#         plot_predictions(file_x_plot, file_y_plot, x, y)
#     print("ERROR: " + str(file_total_error))

if __name__ == '__main__':
    # READ ARGUMENTS FOR FILE AND PLOT
    file = sys.argv[1]
    plot = False
    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":
            plot = True
        else:
            print("ERROR")
    print("PLOT: {}".format(plot))

    x, y = load_points_from_file("train_data/" + file)
    # view_data_segments(x, y)
    segments_number = len(x) // 20
    print(segments_number)
    file_total_error = 0
    file_x_plot = []
    file_y_plot = []
    print("File: " + file)
    errors_plot = []
    # for jj in range(1, 21):
    file_total_error = 0
    for i in range(0, segments_number):
        left = 20 * i
        right = 20 * (i + 1)

        x_current_segment = x[left:right]
        y_current_segment = y[left:right]
        # x_train, x_test = k_fold(x_current_segment, 1, 20)
        # y_train, y_test = k_fold(y_current_segment, 1, 20)
        # print(x_current_segment[1:])
        # print(y_current_segment[1:])
        # print(x_train)
        # print(y_train)

        x_train, x_test = x_current_segment[:19], x_current_segment[19:]
        y_train, y_test = y_current_segment[:19], y_current_segment[19:]

        y_hat_linear, _ = linear_least_squares_fit_predict(x_train, y_train, x_test)
        y_hat_poly, _ = poly_3_least_squares_fit_predict(x_train, y_train, x_test)
        y_hat_sin, _ = sin_least_squares_fit_predict(x_train, y_train, x_test)
        # assert (len(y_hat_linear) == 2)
        # assert (len(y_hat_poly) == 2)
        # assert (len(y_hat_sin) == 2)
        # assert (len(y_test) == 2)

        error_linear = square_error(y_test, y_hat_linear)
        error_poly = square_error(y_test, y_hat_poly)
        error_sin = square_error(y_test, y_hat_sin)

        print("error_linear: " + str(error_linear))
        print("error_poly: " + str(error_poly))
        print("error_sin: " + str(error_sin))
        print("-----------------------------")

        # y_hat_linear = linear_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
        # y_hat_poly = poly_3_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
        # y_hat_sin = sin_least_squares_fit_predict(x[left:right], y[left:right], x[left:right])
        # assert (len(y_hat_linear) == 20)
        # assert (len(y_hat_poly) == 20)
        # assert (len(y_hat_sin) == 20)
        # error_linear = square_error(y[left:right], y_hat_linear)
        # error_poly = square_error(y[left:right], y_hat_poly)
        # error_sin = square_error(y[left:right], y_hat_sin)
        # print("error_linear: " + str(linear_mean_error))
        # print("error_poly: " + str(poly_mean_error))
        # print("error_sin: " + str(sin_mean_error))
        # print("--------------------------------")
        if error_linear < error_poly and error_linear < error_sin:
            print("liniar")
            y_hat_linear_recon, _ = linear_least_squares_fit_predict(x_current_segment, y_current_segment, x_current_segment)
            error_linear_recon = square_error(y_current_segment, y_hat_linear_recon)
            file_total_error += error_linear_recon
            file_y_plot.append(y_hat_linear_recon)

        elif error_poly < error_linear and error_poly < error_sin:
            print("poly")
            y_hat_poly_recon, _ = poly_3_least_squares_fit_predict(x_current_segment, y_current_segment, x_current_segment)
            error_poly_recon = square_error(y_current_segment, y_hat_poly_recon)
            file_total_error += error_poly_recon
            file_y_plot.append(y_hat_poly_recon)
        else:
            print("sin")
            y_hat_sin_recon, _ = sin_least_squares_fit_predict(x_current_segment, y_current_segment,
                                                                   x_current_segment)
            error_sin_recon = square_error(y_current_segment, y_hat_sin_recon)
            file_total_error += error_sin_recon
            file_y_plot.append(y_hat_sin_recon)

        file_x_plot.append(x_current_segment)

        # errors_plot.append(file_total_error)
    # print(errors_plot)

    if plot:
        plot_predictions(file, file_x_plot, file_y_plot, x, y)
    print("ERROR: " + str(file_total_error))
'Script:'
# main()
#find_poly_order()

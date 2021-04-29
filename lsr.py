from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import os


# Utilities
def load_points_from_file(filename):
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def plot_predictions(file, file_x_plot, file_y_plot, x, y):
    plt.title("File: " + file)
    assert len(x) == len(y)
    assert len(x) % 20 == 0
    len_data = len(x)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(x, y, c=colour)
    for i in range(0, num_segments):
        plt.plot(file_x_plot[i], file_y_plot[i], c='red')
    plt.show()


# Error function
def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


def linear_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)

    x_e = np.column_stack((ones, x))
    test_x_e = np.column_stack((test_ones, test_x))

    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def poly_3_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def sin_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    test_x_e = np.column_stack((test_ones, np.sin(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, test_x_e.T), v


def leave_one_out(dataset, e):
    n = len(dataset)
    mask = np.ones(n, dtype=bool)
    test_set = dataset[e - 1]
    mask[e - 1] = False
    train_set = dataset[mask, ...]
    return np.array(train_set), np.array(test_set)


if __name__ == '__main__':
    # READ ARGUMENTS
    file = sys.argv[1]

    plot = False
    if len(sys.argv) == 3:
        if sys.argv[2] == "--plot":
            plot = True
        else:
            print("WRONG PLOT ARGUMENT")

    x, y = load_points_from_file("train_data/" + file)

    segments_number = len(x) // 20

    file_total_error = 0
    file_x_plot = []
    file_y_plot = []
    errors_plot = []

    for i in range(0, segments_number):
        left = 20 * i
        right = 20 * (i + 1)

        x_current_segment, y_current_segment = x[left:right], y[left:right]

        error_linear = 0
        error_poly = 0
        error_sin = 0
        for e in range(0, len(x_current_segment)):
            x_train, x_test = leave_one_out(x_current_segment, e)
            y_train, y_test = leave_one_out(y_current_segment, e)

            y_hat_linear, _ = linear_least_squares_fit_predict(x_train, y_train, x_test)
            y_hat_poly, _ = poly_3_least_squares_fit_predict(x_train, y_train, x_test)
            y_hat_sin, _ = sin_least_squares_fit_predict(x_train, y_train, x_test)

            error_linear += square_error(y_test, y_hat_linear)
            error_poly += square_error(y_test, y_hat_poly)
            error_sin += square_error(y_test, y_hat_sin)

        error_linear /= 20
        error_poly /= 20
        error_sin /= 20

        if error_linear < error_poly and error_linear < error_sin:

            y_hat_linear_recon, _ = linear_least_squares_fit_predict(x_current_segment, y_current_segment,
                                                                     x_current_segment)

            error_linear_recon = square_error(y_current_segment, y_hat_linear_recon)
            file_total_error += error_linear_recon
            file_y_plot.append(y_hat_linear_recon)

        elif error_poly < error_linear and error_poly < error_sin:

            y_hat_poly_recon, _ = poly_3_least_squares_fit_predict(x_current_segment, y_current_segment,
                                                                   x_current_segment)
            error_poly_recon = square_error(y_current_segment, y_hat_poly_recon)
            file_total_error += error_poly_recon
            file_y_plot.append(y_hat_poly_recon)
        else:

            y_hat_sin_recon, _ = sin_least_squares_fit_predict(x_current_segment, y_current_segment,
                                                               x_current_segment)
            error_sin_recon = square_error(y_current_segment, y_hat_sin_recon)
            file_total_error += error_sin_recon
            file_y_plot.append(y_hat_sin_recon)

        file_x_plot.append(x_current_segment)


    if plot:
        plot_predictions(file, file_x_plot, file_y_plot, x, y)

    print(str(file_total_error))

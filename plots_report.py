from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utilities import *
import sys
import os


def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)



def linear_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)

    x_e = np.column_stack((ones, x))
    test_x_e = np.column_stack((test_ones, test_x))

    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v

# def plot_1_noise():
#     errors = [939735.7387720481, 627046.0908504814, 101084.53572133082, 49590.41429822825,
#               23010.373309447335, 26052.05659840825, 32152.96244721118, 42912.1923298576,
#               40851.638162521514, 39986.26764536064, 36854.02828585203, 3002.9194110036246,
#               6111.36935630862, 12986.414878861122, 1449.171357298648, 3124.629034422203,
#               1057.0204970716466, 46.1446805409029, 23.619047723555088, 0.0]
#
#     k = [i for i in range(1, 21)]
#     print(k)
#     plt.plot(k[10:], errors[10:])
#     plt.xticks(k[10:])
#     plt.show()

def poly_2_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)

    x_e = np.column_stack((ones, x, x ** 2))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v


def poly_3_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v


def poly_4_least_squares_fit_predict(x, y, test_x):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v


def poly_5_least_squares_fit_predict(x, y, test_x):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4, x ** 5))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4, test_x ** 5))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v


def poly_6_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6))
    test_x_e = np.column_stack((test_ones, test_x, test_x ** 2, test_x ** 3, test_x ** 4, test_x ** 5, test_x ** 6))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v

def sin_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    test_x_e = np.column_stack((test_ones, np.sin(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v

def cos_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.cos(x)))
    test_x_e = np.column_stack((test_ones, np.cos(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v


def exp_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.exp(x)))
    test_x_e = np.column_stack((test_ones, np.exp(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v

def tan_least_squares_fit_predict(x, y, test_x):
    ones = np.ones(x.shape)
    test_ones = np.ones(test_x.shape)
    x_e = np.column_stack((ones, np.factorial(x)))
    test_x_e = np.column_stack((test_ones, np.mat.factorial(test_x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return np.matmul(v, x_e.T), np.matmul(v, test_x_e.T), v

def plot_2order_vs_3order():
    x, y = load_points_from_file("train_data/basic_5.csv")
    pivot = 17
    poly_2_yhat_train, poly_2_yhat_test, v1 = poly_2_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v1)
    poly_3_yhat_train, poly_3_yhat_test, v2 = poly_3_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v2)
    poly_4_yhat_train, poly_4_yhat_test, v3 = poly_4_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v3)
    poly_5_yhat_train, poly_5_yhat_test, v4 = poly_5_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v4)
    poly_6_yhat_train, poly_6_yhat_test, v5 = poly_6_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v5)

    poly_2_error = square_error(poly_2_yhat_test, y[pivot:])
    poly_3_error = square_error(poly_3_yhat_test, y[pivot:])
    poly_4_error = square_error(poly_4_yhat_test, y[pivot:])
    poly_5_error = square_error(poly_5_yhat_test, y[pivot:])
    poly_6_error = square_error(poly_6_yhat_test, y[pivot:])
    print("Error order 2:" + str(poly_2_error))
    print("Error order 3:" + str(poly_3_error))
    print("Error order 4:" + str(poly_4_error))
    print("Error order 5:" + str(poly_5_error))
    print("Error order 6:" + str(poly_6_error))
    #print(len(np.concatenate((y[:pivot], poly_2_yhat))))
    plt.plot(x, np.concatenate((poly_2_yhat_train, poly_2_yhat_test)), c='red')
    plt.plot(x, np.concatenate((poly_3_yhat_train, poly_3_yhat_test)), c='green')
    plt.plot(x, np.concatenate((poly_4_yhat_train, poly_4_yhat_test)), c='blue')
    plt.plot(x, np.concatenate((poly_5_yhat_train, poly_5_yhat_test)), c='purple')
    plt.plot(x, np.concatenate((poly_6_yhat_train, poly_6_yhat_test)), c='orange')
    plt.scatter(x, y)
    plt.legend(["2nd order", "3rd order", "4th order", "5th order", "6th order"])
    plt.title("file: basic_5, Cross Validation: 17 train / 3 test")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.savefig("plot.png")
    plt.show()


def plot_sin_cos_exp():
    x, y = load_points_from_file("train_data/basic_5.csv")
    pivot = 17
    sin_yhat_train, sin_yhat_test, v1 = sin_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v1)
    cos_yhat_train, cos_yhat_test, v2 = cos_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v2)
    exp_yhat_train, exp_yhat_test, v3 = exp_least_squares_fit_predict(x[:pivot], y[:pivot], x[pivot:])
    print(v3)


    sin_error = square_error(sin_yhat_test, y[pivot:])
    cos_error = square_error(cos_yhat_test, y[pivot:])
    exp_error = square_error(exp_yhat_test, y[pivot:])
    print("Error order 2:" + str(sin_error))
    print("Error order 3:" + str(cos_error))
    print("Error order 4:" + str(exp_error))
    #print(len(np.concatenate((y[:pivot], poly_2_yhat))))
    plt.plot(x, np.concatenate((sin_yhat_train, sin_yhat_test)), c='red')
    plt.plot(x, np.concatenate((cos_yhat_train, cos_yhat_test)), c='green')
    plt.plot(x, np.concatenate((exp_yhat_train, exp_yhat_test)), c='blue')
    #plt.plot(x, np.concatenate((tan_yhat_train, tan_yhat_test)), c='orange')
    # plt.plot(x, np.concatenate((poly_5_yhat_train, poly_5_yhat_test)), c='orange')
    # plt.plot(x, np.concatenate((poly_6_yhat_train, poly_6_yhat_test)), c='yellow')
    plt.scatter(x, y)
    plt.legend(["sin", "cos", "exp"])
    plt.title("file: basic_5, Cross Validation: 17 train / 3 test")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.savefig("sin_cos_exp.png")
    plt.show()

def plot_3():
    x, y = load_points_from_file("train_data/adv_3.csv")
    xx = x[:20]
    yy = y[:20]
    sin_yhat_train, sin_yhat_test, v1 = sin_least_squares_fit_predict(xx, yy, xx)
    sin_error = square_error(sin_yhat_test, yy)
    print(sin_error)

    linear_yhat_train, linear_yhat_test, v1 = linear_least_squares_fit_predict(xx, yy, xx)
    linear_error = square_error(linear_yhat_test, yy)
    print(linear_error)
    plt.plot(xx, linear_yhat_train, c="green")
    plt.plot(xx, sin_yhat_train, c = "red")
    plt.legend(['linear error: ' + str(int(linear_error)), 'sin error: ' + str(int(sin_error))])
    plt.title('1st segment of adv_3.csv')
    plt.scatter(xx,yy)
    plt.savefig('1st_seg.png')
    plt.show()

def plot_1():
    pivot = 17
    errors_adv = [906167032.0871224, 1471896.7293645605, 1401420.0010745653, 1431123.6333541726,
                  1375955.9313743482, 1301675.0877248459, 1209147.267415879, 1035791.835773191,
                  253135.4454735336, 349665.1913626412, 40474.54043305992, 179176.66533791434,
                  52798.9900999682, 10989.391118780222, 3307.0342488684482, 1246.8135950519136,
                  2413.3206107746496, 203.94281398323542, 44.48594054588395, 0.0]
    errors_noise = [939735.7387720481, 627046.0908504814, 101084.53572133082, 49590.41429822825,
                    23010.373309447335, 26052.05659840825, 32152.96244721118, 42912.1923298576,
                    40851.638162521514, 39986.26764536064, 36854.02828585203, 3002.9194110036246,
                    6111.36935630862, 12986.414878861122, 1449.171357298648, 3124.629034422203,
                    1057.0204970716466, 46.1446805409029, 23.619047723555088, 0.0]
    k = [i for i in range(1, 21)]
    print(k)
    plt.plot(k[pivot:], errors_adv[pivot:], ls="--", color="red")
    plt.plot(k[pivot:], errors_noise[pivot:], ls="--")
    plt.legend(["adv_3", "noise_3"])
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Train size ( out of 20 )")
    plt.title("Error Evolution of Cross Validation")
    plt.xticks(k[pivot:])
    plt.grid(True)
    plt.savefig("plot_errors.png")
    plt.show()



def plot_error(dataset):
    plt.title("Error plot for k = 10 ")
    plt.plot(range(2, 7), dataset)
    plt.xticks([2, 3, 4, 5, 6], ['2', '3', '4', '5', '6'])
    plt.show()


def histograms(dataset):
    plt.hist(dataset_10, range=(2, 5))
    plt.yticks((1, 2, 3, 4))
    plt.xticks([2, 3, 4, 5], ['2', '3', '4', '5'])
    plt.show()


dataset_10_errors = [102288903.2400576, 12792289.69877561, 4865.525175215878, 8838.691819062378, 6710254.593521966]
dataset_10 = [2, 2, 2, 2, 3, 3, 3]
# histograms(dataset_10)
# plot_error(dataset_10_errors)
# plot_1_noise()
#plot_1()
plot_3()
#plot_2order_vs_3order()
#plot_sin_cos_exp()

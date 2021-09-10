import re
import numpy as np
import random
import math

import matplotlib.pyplot as plt


# id:206262123

# Python 3.8.6

# plot function to show the optimal rules on 2D graph (not required)
def plott(mat, lines):
    for i in range(len(mat)):
        x = mat[i][0]
        y = mat[i][1]
        sign = mat[i][2]
        if sign == 1:
            plt.plot(x, y, 'o', color='red');
        else:
            plt.plot(x, y, 'o', color='blue');

    for i in range(len(lines)):
        slope = lines[i][0]
        intercept = lines[i][1]
        dir = lines[i][2]
        axes = plt.gca()
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        if dir == 1:
            plt.plot(x_vals, y_vals, '--', color='red')
        else:
            plt.plot(x_vals, y_vals, '--', color='blue')
    plt.show()


def create_data(file):
    lines = list(file.read().splitlines())
    mat = []
    for i in range(len(lines)):
        mat.append(list(map(float, re.split(" ", lines[i]))))

    return np.array(mat)  # convert to numpy


def create_train_and_test(mat):
    random.shuffle(mat)
    n = int(len(mat) / 2)
    test_mat = mat[:n]
    train_mat = mat[n:]
    return train_mat, test_mat


def isAbove(x, y, m, n):
    if y > m * x + n:
        return True
    else:
        return False


def isBelow(x, y, m, n):
    if y < m * x + n:
        return True
    else:
        return False


def get_dir(mat, m, n):
    miss_a = 0
    miss_b = 0
    for point in range(len(mat)):
        x = mat[point][0]
        y = mat[point][1]
        sign = mat[point][2]

        # if red is above line
        if isAbove(x, y, m, n) and sign == -1:
            miss_a = miss_a + 1
        if isBelow(x, y, m, n) and sign == 1:
            miss_a = miss_a + 1
        # if red is under line
        if isAbove(x, y, m, n) and sign == 1:
            miss_b = miss_b + 1
        if isBelow(x, y, m, n) and sign == -1:
            miss_b = miss_b + 1

    if miss_a < miss_b:
        return 1  # red above line
    else:
        return -1  # red under line


def create_rules(mat):
    rules_list = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):

            # create all possible pairs of points
            x1 = mat[i][0]
            y1 = mat[i][1]
            x2 = mat[j][0]
            y2 = mat[j][1]

            if (x2 - x1) != 0:
                m = (y2 - y1) / (x2 - x1)  # slope
            else:
                m = 1000  # when (x2 - x1 = 0) ignore vertical lines for simplicity

            if m != 1000:
                n = y1 - (m * x1)  # y = mx + n
                dir = get_dir(mat, m, n)
                rules_list.append([m, n, dir])

    rules_list = np.asarray(rules_list)
    return rules_list


def func(x, y, sign, rule):
    m = rule[0]
    n = rule[1]
    dir = rule[2]

    if dir == 1:  # if reds are above line

        if sign == 1:
            if isAbove(x, y, m, n):
                return True
            else:
                return False

        else:  # sign = -1
            if isBelow(x, y, m, n):
                return True
            else:
                return False

    else:  # if reds are below line

        if sign == 1:
            if isBelow(x, y, m, n):
                return True
            else:
                return False

        else:  # sign = -1
            if isAbove(x, y, m, n):
                return True
            else:
                return False


def adaboost(mat, rules, point_weights, rule_weights, iters):  # return one optimal line and new weights

    op_rules = []  # optimal rules list
    op_alphas = np.zeros(iters)  # optimal alpha weights list

    for i in range(iters):  # run adaboost 'iters' times
        # rule_weights = np.zeros(len(rules))
        for rule in range(len(rule_weights)):
            r_weight = 0

            for p in range(len(mat)):
                x = mat[p][0]
                y = mat[p][1]
                point_sign = mat[p][2]

                if not func(x, y, point_sign, rules[rule]):
                    r_weight = r_weight + point_weights[p]

            rule_weights[rule] = r_weight

        index_of_optimal = np.argmin(rule_weights)

        ht = rules[index_of_optimal]  # optimal rule

        op_rules.append(ht)  # add optimal rule to the list

        error = rule_weights[index_of_optimal]

        alpha = 0.5 * np.log((1 - error) / error)

        op_alphas[i] = alpha

        # update weights

        new_weights = np.zeros(len(point_weights))
        for w in range(len(point_weights)):
            x = mat[w][0]
            y = mat[w][1]
            point_sign = mat[w][2]
            if func(x, y, point_sign, ht):
                ht_x = 1
            else:
                ht_x = -1

            n_weight = point_weights[w] * math.exp((-1) * alpha * ht_x)
            new_weights[w] = n_weight

        norm_weights = new_weights / np.sum(new_weights)
        point_weights = norm_weights

    return np.asarray(op_rules), op_alphas


def Hk_func(rules, alphas, x, y, rules_num):
    sum = 0
    for i in range(rules_num):

        dir = rules[i][2]

        if dir == 1:
            if isAbove(x, y, rules[i][0], rules[i][1]):
                hi_x = 1
            else:
                hi_x = -1

        if dir == -1:
            if isAbove(x, y, rules[i][0], rules[i][1]):
                hi_x = -1
            else:
                hi_x = 1

        sum = sum + alphas[i] * hi_x

    if sum < 0:
        return -1
    else:
        return 1


def eHk_func(mat, rules, alphas):
    n = len(mat)
    e_errors = np.zeros(len(rules))  # empirical errors
    for i in range(len(rules)):

        sum = 0
        for point in range(n):
            x = mat[point][0]
            y = mat[point][1]
            point_sign = mat[point][2]

            Hk_x = Hk_func(rules, alphas, x, y, i + 1)
            point_sign = int(point_sign)
            if Hk_x != point_sign:
                sum = sum + 1

        e_errors[i] = sum / n

    return e_errors


def compute_empirical_errors(mat):

    train_mat, test_mat = create_train_and_test(mat)  # randomize every iter
    rules_train = create_rules(train_mat)

    #### train ####

    n = train_mat.shape[0]  # number of points
    p_weights = np.zeros(n)  # point weights
    p_weights.fill(1 / n)  # initial weights
    rule_weights = np.zeros(len(rules_train))

    op_rules, op_alphas = adaboost(train_mat, rules_train, p_weights, rule_weights, 8)

    errors_train = eHk_func(train_mat, op_rules, op_alphas)

    #### test ####

    errors_test = eHk_func(test_mat, op_rules, op_alphas)

    return errors_train, errors_test


def main():
    epochs = 10  # number of main iterations

    sum_train = np.zeros(8)
    sum_test = np.zeros(8)
    for i in range(epochs):
        rec_file = open("rectangle.txt", "r")
        rec_mat = create_data(rec_file)  # creates a matrix for data
        errors_train, errors_test = compute_empirical_errors(rec_mat)

        for j in range(8):
            sum_train[j] = sum_train[j] + errors_train[j]
            sum_test[j] = sum_test[j] + errors_test[j]

    print("train errors average")
    print(sum_train / epochs)
    print("test errors average")
    print(sum_test / epochs)


if __name__ == '__main__':
    main()

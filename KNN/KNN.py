import numpy as np
import re


#id:206262123


def create_data(file):
    lines = list(file.read().splitlines())
    mat = []
    for i in range(len(lines)):
        mat.append(list(map(float, re.split(" ", lines[i]))))

    return np.array(mat)  # convert to numpy


def create_train_and_test(mat):
    np.random.shuffle(mat)
    n = int(len(mat) / 2)
    test_mat = mat[:n]
    train_mat = mat[n:]
    return train_mat, test_mat


def calc_KNN(k, p, test_mat, train_mat, emp):
    miss_count = 0
    for i in range(len(test_mat)):

        distances = np.zeros(len(train_mat))

        x_test = test_mat[i][0]
        y_test = test_mat[i][1]
        type_test = test_mat[i][2]

        for j in range(len(train_mat)):

            if emp:  # if train_mat = test_mat
                if i == j:
                    distances[j] = float('inf')
                    continue

            x_train = train_mat[j][0]
            y_train = train_mat[j][1]

            dist = ((abs(x_train - x_test)) ** p + (abs(y_train - y_test)) ** p) ** (1 / p)  # euclidean distance

            distances[j] = dist

        pos_count = 0
        for d in range(k):

            m = np.argmin(distances) # get the index of minimum distance
            closest = train_mat[m]
            if closest[2] == 1:
                pos_count = pos_count + 1
            distances = np.delete(distances, m) # delete to get next minimum

        if pos_count > int(k / 2):
            sign = 1
        else:
            sign = -1

        if type_test != sign:
            miss_count = miss_count + 1

    error = round((miss_count / len(test_mat)), 4)  # error percentage

    return error


def main():
    epochs = 100
    ans_emp = np.zeros(15)
    ans_true = np.zeros(15)
    for i in range(epochs):
        rec_file = open("rectangle.txt", "r")
        rec_mat = create_data(rec_file)  # creates a matrix for data
        train_mat, test_mat = create_train_and_test(rec_mat)

        ans_index = 0

        for k in range(1, 11, 2):  # k = 1,3,5,7,9
            for j in range(1, 4):  # p = 3 = infinity
                if j == 3:
                    p = float('inf')
                else:
                    p = j
                ans_true[ans_index] += calc_KNN(k, p, test_mat, train_mat, False)
                ans_emp[ans_index] += calc_KNN(k, p, train_mat, train_mat, True)
                ans_index += 1

    ans_emp = ans_emp / epochs
    ans_true = ans_true / epochs

    print("order:")
    print("[(k=1,p=1) (k=1,p=2) (k=1,p=inf) (k=3,p=1) (k=3,p=2) (k=3,p=inf) (k=5,p=1) (k=5,p=2) (k=5,p=inf)\n "
          "(k=7,p=1) (k=7,p=2) (k=7,p=inf) (k=9,p=1) (k=9,p=2) (k=9,p=inf)]")
    print("empirial error average")
    print(ans_emp)
    print("true error average")
    print(ans_true)


if __name__ == '__main__':
    main()

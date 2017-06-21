# Author: Wang Haoting


import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression


eps = 1e-6


def transform(y):
    new_y = np.array(y)
    new_y[new_y == 0] = -1
    return new_y


y = np.genfromtxt('targets.csv', delimiter=',').flatten()
X = np.genfromtxt('data.csv', delimiter=',')
X = normalize(X)
kf = KFold(n_splits=10)
kf.get_n_splits(X)
ensemble_size = [1, 5, 10, 100]


for e_size in ensemble_size:
    cc = 0
    for train_index, test_index in kf.split(X):
        cc += 1
        Xt = X[train_index, :]
        yt = y[train_index]
        with open('experiments/base%d_fold%d.csv' % (e_size, cc), 'w') as file:
            sample_weight = np.ones_like(yt)
            sample_weight /= sample_weight.sum()

            logistic_classifier = []
            alpha = []
            for i in range(e_size):
                logistic_classifier.append(LogisticRegression(C=1000))
                logistic_classifier[i] = logistic_classifier[i].fit(Xt, yt, sample_weight)
                score = logistic_classifier[i].score(Xt, yt, sample_weight)
                # print(score)
                if score <= 0.5:
                    break
                alpha.append(np.log((score + eps) / (1 - score + eps)) / 2)
                h = logistic_classifier[i].predict(Xt)
                sample_weight *= np.exp(-alpha[i] * transform(h) * transform(yt))
                sample_weight /= np.sum(sample_weight)
            y_pred = np.zeros_like(y[test_index])
            X_test = X[test_index, :]
            # print(len(alpha))
            for i in range(len(alpha)):
                y_pred_cur = logistic_classifier[i].predict(X_test)
                y_pred += alpha[i] * transform(y_pred_cur)
            y_pred[y_pred > 0] = 1
            y_pred[y_pred <= 0] = 0
            for i in range(test_index.size):
                file.write(str(test_index[i] + 1) + "," + str(int(y_pred[i])) + "\r\n")
            file.close()

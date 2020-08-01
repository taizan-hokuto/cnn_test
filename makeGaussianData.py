import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def getData(nclass, seed=None):

    assert nclass == 2 or nclass == 3

    if seed != None:
        np.random.seed(seed)

    # 2次元の spherical な正規分布3つからデータを生成
    X0   = 0.10 * np.random.randn(200, 2) + [ 0.3, 0.3 ]
    X1   = 0.10 * np.random.randn(200, 2) + [ 0.7, 0.6 ]
    X2   = 0.05 * np.random.randn(200, 2) + [ 0.3, 0.7 ]

    # それらのラベル用のarray
    lab0 = np.zeros(X0.shape[0], dtype = int)
    lab1 = np.zeros(X1.shape[0], dtype = int) + 1
    lab2 = np.zeros(X2.shape[0], dtype = int) + 2

    # X （入力データ）, label （クラスラベル）, t（教師信号） をつくる
    if nclass == 2:
        X = np.vstack((X0, X1))
        label = np.hstack((lab0, lab1))
        t = np.zeros(X.shape[0])
        t[label == 1] = 1.0
    else:
        X = np.vstack((X0, X1, X2))
        label = np.hstack((lab0, lab1, lab2))
        t = np.zeros((X.shape[0], nclass))
        for ik in range(nclass):
            t[label == ik, ik] = 1.0

    return X, label, t


if __name__ == '__main__':


    K = 3

    X, lab, t = getData(K)

    fig = plt.figure()
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1)
    ax.scatter(X[lab == 0, 0], X[lab == 0, 1], color = 'red')
    ax.scatter(X[lab == 1, 0], X[lab == 1, 1], color = 'green')
    if K == 3:
        ax.scatter(X[lab == 2, 0], X[lab == 2, 1], color = 'blue')
    plt.show()
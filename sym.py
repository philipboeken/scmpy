import math as m
import numpy as np
import scipy.stats as stats
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge
from contextlib import contextmanager
import sys
import os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def sample(a, n):
    X, Y, Z = [], [], []
    for i in range(N):
        Z.append(np.random.normal())
        X.append(f(a, Z[-1]) + 0.3 * np.random.normal())
        # X.append(Z[-1] + 0.3 * np.random.normal())
        Y.append(f(a, Z[-1]) + 0.3 * np.random.normal())
        # Y.append(Z[-1] + 0.3 * np.random.normal())
    return X, Y, Z


def pc_test(X, Y, Z, alpha):
    # H0: X independent of Y given Z <=> rho = 0 <=> z = 0
    def rho(X, Y):
        return np.corrcoef(X, Y)[0, 1]

    def partial_correlation(X, Y, Z):
        n = rho(X, Y) - rho(X, Z) * rho(Y, Z)
        d = m.sqrt((1 - rho(X, Z)**2) * (1 - rho(Y, Z)**2))
        return n/d

    def z(X, Y, Z):
        rho = partial_correlation(X, Y, Z)
        return (1/2) * m.log((1+rho) / (1-rho))

    F = m.sqrt(len(X) - 4) * abs(z(X, Y, Z))
    print(F)
    return F > stats.norm.ppf(1 - alpha/2)


def gcm_test(X, Y, Z, alpha, method='xgboost'):
    # H0: X independent of Y given Z <=> T ~ N(0,1)
    def pred_xgboost(data, label):
        dtrain = xgb.DMatrix(np.array([data]).T, label=np.array([label]).T)
        dtest = xgb.DMatrix(np.array([Z]).T)
        with suppress_stdout():
            return xgb.train(
                {
                    'eta': 0.27,
                    'max_depth': 100,
                    'eval_metric': 'rmse'
                },
                dtrain,
                num_boost_round=12,
                maximize=True
            ).predict(dtest)

    def pred_krr(data, label):
        clf = KernelRidge(alpha=1.0, kernel='polynomial')
        return clf.fit(np.array([data]).T, np.array([label]).T).predict(np.array([label]).T)

    if method == 'xgboost':
        f_hat, g_hat = pred_xgboost(X, Z), pred_xgboost(Y, Z)
    else:
        f_hat, g_hat = pred_krr(X, Z), pred_krr(Y, Z)

    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = m.sqrt(len(X)) * np.average(R) / \
        m.sqrt(np.average(R2) - np.average(R)**2)
    print(abs(T))

    return abs(T) > stats.norm.ppf(1 - alpha/2)


def f(a, x):
    return m.exp(-x**2/2) * m.sin(a * x)


a = 3
N = 1000
X, Y, Z = sample(a, N)
alpha = 0.05
pc = pc_test(X, Y, Z, alpha)
gcm = gcm_test(X, Y, Z, alpha)

pc = '' if pc else 'in'
gcm = '' if gcm else 'in'

print('Partial correlation output: conditionally %sdependent' % pc)
print('Generalized Covariance Measure output: conditionally %sdependent' % gcm)

from sklearn.kernel_ridge import KernelRidge
from scipy.stats import ks_2samp, kruskal
from contextlib import contextmanager
import scipy.stats as stats
import xgboost as xgb
import pandas as pd
import numpy as np
import math as m
import sklearn
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


def rho(X, Y):
    return np.corrcoef(X, Y)[0, 1]


def partial_correlation(X, Y, Z):
    n = rho(X, Y) - rho(X, Z) * rho(Y, Z)
    d = m.sqrt((1 - rho(X, Z)**2) * (1 - rho(Y, Z)**2))
    return n/d


def pcorr_test(X, Y, Z):
    # H0: X independent of Y given Z <=> pc = 0 <=> z = 0
    pc = partial_correlation(X, Y, Z)
    z = (1/2) * m.log((1+pc) / (1-pc))
    F = m.sqrt(len(X) - 4) * abs(z)
    return 2*stats.norm.sf(F)


def corr_test(X, Y):
    # H0: X independent of Y <=> r = 0 <=> z = 0
    r = rho(X, Y)
    z = (1/2) * m.log((1+r) / (1-r))
    F = m.sqrt(len(X) - 3) * abs(z)
    return 2*stats.norm.sf(F)


def discr_cont_indep_test(C, X):
    # H0: X independent of C <=> X|C==1 !~ X|C==0
    data = pd.concat([C, X], axis=1)
    X1 = data.loc[data[C.name] == 0][X.name]
    X2 = data.loc[data[C.name] == 1][X.name]
    # return kruskal(X1, X2).pvalue
    return ks_2samp(X1, X2).pvalue


def pred_xgboost(data, label):
    dtrain = xgb.DMatrix(np.array([data]).T, label=np.array([label]).T)
    dtest = xgb.DMatrix(np.array([label]).T)
    with suppress_stdout():
        model = xgb.train(
            {
                'eta': 0.27,
                'max_depth': 100,
                'eval_metric': 'rmse'
            },
            dtrain,
            num_boost_round=12,
            maximize=True
        )
    return model.predict(dtest)


def pred_krr(data, label):
    clf = KernelRidge(alpha=1.0, kernel='polynomial')
    model = clf.fit(np.array([data]).T, np.array([label]).T)
    return model.predict(np.array([label]).T)


def gcm_test(X, Y, Z, alpha, method='xgboost'):
    # H0: X independent of Y given Z <=> T ~ N(0,1)
    if method == 'xgboost':
        f_hat, g_hat = pred_xgboost(X, Z), pred_xgboost(Y, Z)
    else:
        f_hat, g_hat = pred_krr(X, Z), pred_krr(Y, Z)

    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = m.sqrt(len(X)) * np.average(R) / \
        m.sqrt(np.average(R2) - np.average(R)**2)

    return abs(T) > stats.norm.ppf(1 - alpha/2)

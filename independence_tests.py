from rpy2.robjects import pandas2ri, globalenv
from rpy2.robjects.packages import importr
from sklearn.kernel_ridge import KernelRidge
from pygam import LinearGAM, LogisticGAM, s
import scipy.stats as stats
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import sys
import os

mgcv = importr('mgcv')
base = importr('base')
rstats = importr('stats')
pandas2ri.activate()


def pred_xgboost(Y, X):
    model = xgb.XGBRegressor(
        eta=0.27,
        max_depth=100,
        eval_metric='rmse',
        num_boost_round=12,
        maximize=True
    ).fit(np.array([X]).T, np.array([Y]).T)
    return model.predict(np.array([Y]).T)


def pred_krr(Y, X):
    model = KernelRidge(
        alpha=1.0,
        kernel='polynomial'
    ).fit(np.array([X]).T, np.array([Y]).T)
    return model.predict(np.array([Y]).T)


# def pred_gam(Y, X):
#     discrete = len(Y.unique()) < len(Y) / 2
#     if discrete:
#         gam = LogisticGAM(s(0)).fit(X, Y)
#     else:
#         gam = LinearGAM(s(0)).fit(X, Y)
#     return gam.predict(X)


def pred_gam(Y, X):
    globalenv['Y'] = Y
    globalenv['X'] = X
    discrete = len(Y.unique()) < len(Y) / 2
    if discrete:
        gam = mgcv.gam(rstats.formula('Y ~ s(X)'),
                       family=base.as_symbol('binomial'))
    else:
        gam = mgcv.gam(rstats.formula('Y ~ s(X)'))
    return rstats.predict(gam, base.as_data_frame(base.as_symbol('X')))


def rho(X, Y):
    return np.corrcoef(X, Y)[0, 1]


def partial_correlation(X, Y, Z):
    n = rho(X, Y) - rho(X, Z) * rho(Y, Z)
    d = np.sqrt((1 - rho(X, Z)**2) * (1 - rho(Y, Z)**2))
    return n/d


def get_gam(X, Y):
    globalenv['Y'] = Y
    globalenv['X'] = X
    return mgcv.gam(rstats.formula('Y ~ s(X)'))


def pval_gam(gam):
    return base.summary(gam).rx2('s.pv')[0]


def corr(X, Y):
    # H0: X independent of Y <=> r = 0 <=> z = 0
    r = rho(X, Y)
    z = (1/2) * np.log((1+r) / (1-r))
    F = np.sqrt(len(X) - 3) * abs(z)
    return 2*stats.norm.sf(F)


def cdf(C, X):
    # H0: X independent of C <=> X|C=1 !~ X|C=0
    data = pd.concat([C, X], axis=1)
    X1 = data.loc[data[C.name] == 0][X.name]
    X2 = data.loc[data[C.name] == 1][X.name]
    # return stats.kruskal(X1, X2).pvalue
    return stats.ks_2samp(X1, X2).pvalue


# Tends to reject H0 too often, pygam has buggy p-values
# https://github.com/dswah/pyGAM/issues/163
# def gam(X, Y):
    # # H0: X independent of Y
    # X_discrete = len(X.unique()) < len(X) / 2
    # Y_discrete = len(Y.unique()) < len(Y) / 2
    # if X_discrete:
    #     gam = LogisticGAM(s(0)).fit(Y, X)
    # elif Y_discrete:
    #     gam = LogisticGAM(s(0)).fit(X, Y)
    # else:
    #     gam1 = LinearGAM(s(0)).fit(X, Y)
    #     gam2 = LinearGAM(s(0)).fit(Y, X)
    #     return min(gam1._compute_p_value(0),
    #                gam2._compute_p_value(0))
    # return gam._compute_p_value(0)


def gam(X, Y):
    # H0: X independent of Y
    X_discrete = len(X.unique()) < len(X) / 2
    globalenv['X'] = X
    globalenv['Y'] = Y
    if X_discrete:
        gam = mgcv.gam(rstats.formula('X ~ s(Y)'),
                       family=base.as_symbol('binomial'))
        return base.summary(gam).rx2('s.pv')[0]
    else:
        gam1 = mgcv.gam(rstats.formula('Y~s(X)'))
        gam2 = mgcv.gam(rstats.formula('X~s(Y)'))
        return min(base.summary(gam1).rx2('s.pv')[0],
                   base.summary(gam2).rx2('s.pv')[0])


def pcorr(X, Y, Z):
    # H0: X independent of Y given Z <=> pc = 0 <=> z = 0
    pc = partial_correlation(X, Y, Z)
    z = (1/2) * np.log((1+pc) / (1-pc))
    F = np.sqrt(len(X) - 4) * abs(z)
    return 2*stats.norm.sf(F)


def gcm(X, Y, f_hat, g_hat):
    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = np.sqrt(len(X)) * np.average(R) / \
        np.sqrt(np.average(R2) - np.average(R)**2)
    return 2*stats.norm.sf(abs(T))


def get_pred_from_gam(gam, X):
    globalenv['X'] = X
    return rstats.predict(gam, base.as_data_frame(base.as_symbol('X')))


def gam_gcm(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_gam(X, Z), pred_gam(Y, Z)
    return gcm(X, Y, f_hat, g_hat)


def xgb_gcm(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_xgboost(X, Z), pred_xgboost(Y, Z)
    return gcm(X, Y, f_hat, g_hat)


def krr_gcm(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_krr(X, Z), pred_krr(Y, Z)
    return gcm(X, Y, f_hat, g_hat)

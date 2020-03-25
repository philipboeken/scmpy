from rpy2.robjects import pandas2ri, globalenv
from rpy2.robjects.packages import importr
from sklearn.kernel_ridge import KernelRidge
from pygam import LinearGAM, LogisticGAM, s
from scipy.stats import ks_2samp, kruskal
import scipy.stats as stats
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import sys
import os


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


def pred_gam(Y, X):
    discrete = len(Y.unique()) < len(Y) / 2
    if discrete:
        gam = LogisticGAM(s(0)).fit(X, Y)
    else:
        gam = LinearGAM(s(0)).fit(X, Y)
    return gam.predict(X)


def pred_gam_r(Y, X):
    mgcv = importr('mgcv')
    base = importr('base')
    stats = importr('stats')
    pandas2ri.activate()
    globalenv['Y'] = Y
    globalenv['X'] = X
    discrete = len(Y.unique()) < len(Y) / 2
    if discrete:
        gam = mgcv.gam(stats.formula('Y ~ s(X)'),
                       family=base.as_symbol('binomial'))
    else:
        gam = mgcv.gam(stats.formula('Y ~ s(X)'))
    return stats.predict(gam, base.as_data_frame(base.as_symbol('X')))


def rho(X, Y):
    return np.corrcoef(X, Y)[0, 1]


def partial_correlation(X, Y, Z):
    n = rho(X, Y) - rho(X, Z) * rho(Y, Z)
    d = np.sqrt((1 - rho(X, Z)**2) * (1 - rho(Y, Z)**2))
    return n/d


def corr_ind_test(X, Y):
    # H0: X independent of Y <=> r = 0 <=> z = 0
    r = rho(X, Y)
    z = (1/2) * np.log((1+r) / (1-r))
    F = np.sqrt(len(X) - 3) * abs(z)
    return 2*stats.norm.sf(F)


def cdf_ind_test(C, X):
    # H0: X independent of C <=> X|C=1 !~ X|C=0
    data = pd.concat([C, X], axis=1)
    X1 = data.loc[data[C.name] == 0][X.name]
    X2 = data.loc[data[C.name] == 1][X.name]
    # return kruskal(X1, X2).pvalue
    return ks_2samp(X1, X2).pvalue


def gam_ind_test(X, Y):
    # Tends to reject H0 too often
    # H0: X independent of Y
    X_discrete = len(X.unique()) < len(X) / 2
    Y_discrete = len(Y.unique()) < len(Y) / 2
    if X_discrete:
        gam = LogisticGAM(s(0)).fit(Y, X)
    elif Y_discrete:
        gam = LogisticGAM(s(0)).fit(X, Y)
    else:
        gam1 = LinearGAM(s(0)).fit(X, Y)
        gam2 = LinearGAM(s(0)).fit(Y, X)
        return min(gam1._compute_p_value(0),
                   gam2._compute_p_value(0))
    return gam._compute_p_value(0)


def gam_r_ind_test(X, Y):
    # H0: X independent of Y
    X_discrete = len(X.unique()) < len(X) / 2
    mgcv = importr('mgcv')
    base = importr('base')
    stats = importr('stats')
    pandas2ri.activate()
    globalenv['X'] = X
    globalenv['Y'] = Y
    if X_discrete:
        gam = mgcv.gam(stats.formula('X ~ s(Y)'),
                       family=base.as_symbol('binomial'))
        return base.summary(gam).rx2('s.pv')[0]
    else:
        gam1 = mgcv.gam(stats.formula('Y~s(X)'))
        gam2 = mgcv.gam(stats.formula('X~s(Y)'))
        pandas2ri.deactivate()
        return min(base.summary(gam1).rx2('s.pv')[0],
                   base.summary(gam2).rx2('s.pv')[0])


def pcorr_cond_ind_test(X, Y, Z):
    # H0: X independent of Y given Z <=> pc = 0 <=> z = 0
    pc = partial_correlation(X, Y, Z)
    z = (1/2) * np.log((1+pc) / (1-pc))
    F = np.sqrt(len(X) - 4) * abs(z)
    return 2*stats.norm.sf(F)


def gam_cond_ind_test(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_gam_r(X, Z), pred_gam_r(Y, Z)
    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = np.sqrt(len(X)) * np.average(R) / \
        np.sqrt(np.average(R2) - np.average(R)**2)
    return 2*stats.norm.sf(abs(T))


def xgb_cond_ind_test(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_xgboost(X, Z), pred_xgboost(Y, Z)
    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = np.sqrt(len(X)) * np.average(R) / \
        np.sqrt(np.average(R2) - np.average(R)**2)
    return 2*stats.norm.sf(abs(T))


def krr_cond_ind_test(X, Y, Z):
    # H0: X independent of Y given Z <=> GCM: T ~ N(0,1)
    f_hat, g_hat = pred_krr(X, Z), pred_krr(Y, Z)
    R = [(X[i] - f_hat[i]) * (Y[i] - g_hat[i]) for i in range(len(X))]
    R2 = np.square(R)
    T = np.sqrt(len(X)) * np.average(R) / \
        np.sqrt(np.average(R2) - np.average(R)**2)
    return 2*stats.norm.sf(abs(T))

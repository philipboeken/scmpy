from itertools import combinations, product, permutations
from sklearn.metrics import roc_curve, auc
from independence_tests import *
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import math as m


class SCMEstimator:
    def __init__(self, data, system=set(), context=set(), alpha=0.02):
        self.system = system
        self.context = context
        self.alpha = alpha
        self.last_alg = ''
        self.data = data
        self.depcies = DataFrame(0, index=self.nodes(),
                                 columns=self.nodes('system'))
        self.arel = DataFrame(0, index=self.nodes('system'),
                              columns=self.nodes('system'))
        self.conf = DataFrame(0.0, index=self.nodes('system'),
                              columns=self.nodes('system'))

    def nodes(self, type=None):
        if type:
            return sorted(getattr(self, type), key=lambda node: node.name)
        return sorted(self.system | self.context, key=lambda node: node.name)

    def lcd_speedup(self, surgical=False):
        # {(X dep Y), (C dep X), (Y indep C given X)} => X->Y
        for X, Y in permutations(self.nodes('system'), 2):
            for C in X.parents(type='context'):
                data = self.get_data(obs=True, context=C)
                if corr(data[C], data[X]) > self.alpha:
                    continue
                self.depcies.at[C, X] = 1
                data = self.get_data(obs=True)
                gam_y = get_gam(data[X], data[Y])
                if pval_gam(gam_y) > self.alpha:
                    continue
                self.depcies.at[X, Y] = 1
                data = self.get_data(obs=True, context=C)
                pred_c = pred_gam(data[C], data[X])
                pred_y = get_pred_from_gam(gam_y, data[X])
                pval = gcm(data[Y], data[C], pred_y, pred_c)
                if pval < self.alpha:
                    continue
                self.arel.at[X, Y] = 1
                self.conf.at[X, Y] = max(self.conf.at[X, Y], pval)
        self.last_alg = 'lcd-speedup'

    def lcd(self, indep_test, c_indep_test):
        # {(X dep Y), (C dep X), (Y indep C given X)} => X->Y
        p_dep = DataFrame(0.0, index=self.nodes(),
                          columns=self.nodes('system'))
        for X, Y in combinations(self.nodes('system'), 2):
            data = self.get_data(obs=True)
            p_dep.at[X, Y] = indep_test(data[X], data[Y])
            p_dep.at[Y, X] = p_dep.at[X, Y]
        for C, X in product(self.nodes('context'), self.nodes('system')):
            data = self.get_data(obs=True, context=C)
            p_dep.at[C, X] = corr(data[C], data[X])

        for X, Y in permutations(self.nodes('system'), 2):
            if p_dep.at[X, Y] > self.alpha:
                continue
            for C in self.nodes('context'):
                if p_dep.at[C, X] > self.alpha or not C.is_parent_of(X):
                    continue
                data = self.get_data(obs=True, context=C)
                pval = c_indep_test(data[C], data[Y], data[X])
                if pval < self.alpha:
                    continue
                self.arel.at[X, Y] = 1
                self.conf.at[X, Y] = max(self.conf.at[X, Y], pval)
                # self.conf.at[X, Y] = max(self.conf.at[X, Y], -np.log(p_dep.at[C, X]))
        self.depcies = p_dep.applymap(lambda x: 1 if x < self.alpha else 0)
        self.last_alg = 'lcd'

    def save_to(self, outdir):
        self.depcies.to_csv(
            f'{outdir}/{self.last_alg}-dependencies.csv',
            sep='\t'
        )
        self.conf.to_csv(f'{outdir}/{self.last_alg}-arel-confidence.csv')
        self.arel.to_csv(f'{outdir}/{self.last_alg}-arel.csv', sep='\t')

    def plot_roc(self, labels, outdir):
        fpr, tpr, _ = roc_curve(
            labels.values.flatten().tolist(),
            self.conf.values.flatten().tolist()
        )
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.last_alg} ROC')
        plt.legend(loc="lower right")
        plt.savefig(f'{outdir}/{self.last_alg}-roc.png', format='png')

    def get_data(self, context=None, obs=True):
        if not context and not obs:
            return self.data
        data = self.data.iloc[0:0].copy()
        if context:
            data = data.append(
                self.data.loc[self.data[context] == 1],
                ignore_index=True
            )
        if obs:
            conditions = [self.data[C] == 0 for C in self.nodes('context')]
            observational = self.data.loc[np.logical_and.reduce(conditions)]
            data = data.append(observational, ignore_index=True)
        return data

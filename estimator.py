from independence_tests import discr_cont_indep_test
from itertools import combinations, product
from pandas import DataFrame
import numpy as np
import math as m


class SCMEstimator:
    def __init__(self, data, system=set(), context=set(), alpha=0.02):
        self.data = data
        self.system = system
        self.context = context
        self.alpha = alpha
        self.last_alg = ''

    def nodes(self, type=None):
        if type:
            return sorted(getattr(self, type), key=lambda node: node.name)
        return sorted(self.system | self.context, key=lambda node: node.name)

    def lcd(self, indep_test, c_indep_test):
        p_vals_indep = DataFrame(0.0, index=self.nodes(), columns=self.nodes())
        anc_rel = DataFrame(0, index=self.nodes(), columns=self.nodes())
        conf = DataFrame(0.0, index=self.nodes(), columns=self.nodes())

        for X, Y in combinations(self.nodes('system'), 2):
            p_vals_indep.at[X, Y] = indep_test(self.data[X], self.data[Y])
        for C, X in product(self.nodes('context'), self.nodes('system')):
            p_vals_indep.at[C, X] = indep_test(self.data[C], self.data[X])

        for X, Y in combinations(self.nodes('system'), 2):
            if p_vals_indep.at[X, Y] > self.alpha:
                continue
            for C in self.nodes('context'):
                if p_vals_indep.at[C, X] > self.alpha:
                    continue
                pval = c_indep_test(self.data[C], self.data[Y], self.data[X])
                if pval >= self.alpha:
                    anc_rel.at[X, Y] = 1
                    conf.at[X, Y] = max(conf.at[X, Y], pval)
        print(p_vals_indep.round(4))
        self.conf = conf
        self.anc_rel = anc_rel
        self.last_alg = 'lcd'

    def save_to(self, outdir):
        self.conf.to_csv(f'{outdir}/{self.last_alg}-anc-rel-confidence.csv')
        self.anc_rel.to_csv(f'{outdir}/{self.last_alg}-anc-rel.csv')

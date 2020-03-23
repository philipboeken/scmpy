from simulator import SCMGenerator, SCMSimulator
from estimator import SCMEstimator
from independence_tests import *
from helpers import set_seed
import shutil
import os

# The number of system variables
p = 4
# The number of context variables
q = 4
# Probability of drawing a latent confounder
eps = 0.1
# Probability of drawing a directed edge
eta = 0.5
# Whether the graph is acyclic
acyclic = True
# Relation between system variables: 'linear' | 'additive' | 'nonlinear'
rel = 'additive'
# Type of interventions: True: perfect interventions | False: mechanism changes
surgical = True
# Number of samples drawn from each context
N = 500
# The seed for the random number generators
seed = 4
# Independence test
ind_test = rho_ind_test
# Conditional independece test
cond_ind_test = gam_cond_ind_test

outdir = f'./out/p={p}_q={q}_eps={eps}_eta={eta}_acyclic={acyclic}' \
         + f'_rel={rel}_surgical={surgical}_N={N}_seed={seed}' \
         + f'_itest={ind_test.__name__}_citest={cond_ind_test.__name__}'

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

set_seed(seed)

generator = SCMGenerator(p, q, eps, eta, acyclic, surgical, rel)
scm = generator.generate_scm()
scm.save_to(outdir)

simulator = SCMSimulator(scm)
simulator.simulate(N)
simulator.save_to(outdir)

data = simulator.data
estimator = SCMEstimator(
    data=data,
    system=scm.system,
    context=scm.context,
    alpha=0.02
)
estimator.lcd(ind_test, cond_ind_test)
estimator.save_to(outdir)
estimator.plot_roc(
    labels=scm.H.ancestral_matrix('system'),
    outdir=outdir
)

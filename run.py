from independence_tests import pcorr_test, corr_test
from simulator import SCMGenerator, SCMSimulator
from estimator import SCMEstimator
from helpers import set_seed, draw
import shutil
import os

# The number of system variables
p = 4
# The number of context variables
q = 4
# Probability of drawing a latent confounder
eps = 0.3
# Probability of drawing a directed edge
eta = 0.5
# Whether the graph is acyclic
acyclic = True
# Relation between system variables: 'linear' | 'additive' | 'nonlinear'
rel = 'linear'
# Type of interventions: True: perfect interventions | False: mechanism changes
surgical = True
# Number of samples drawn from each context
N = 500
# The seed for the random number generators
seed = 3

outdir = f"./out/p={p}_q={q}_eps={eps}_eta={eta}_acyclic={acyclic}" \
         + f"_rel={rel}_surgical={surgical}_N={N}_seed={seed}"

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
estimator.lcd(corr_test, pcorr_test)
estimator.save_to(outdir)
estimator.plot_roc(
    labels=scm.H.ancestral_matrix('system'),
    outdir=outdir
)

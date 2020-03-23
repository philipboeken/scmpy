from independence_tests import pcorr_test, corr_test
from simulator import SCMGenerator, SCMSimulator
from estimator import SCMEstimator
from helpers import set_seed, draw
import shutil
import os

# The number of system variables
p = 4
# The number of context variables
q = 3
# Probability of drawing a latent confounder
eps = 0.05
# Probability of drawing a directed edge
eta = 0.1
# Whether the graph is acyclic
acyclic = True
# Relation between system variables: 'linear' | 'additive' | 'nonlinear'
rel = 'additive'
# Type of interventions: True: perfect interventions | False: mechanism changes
surgical = False
# Number of samples drawn from each context
N = 100
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
estimator = SCMEstimator(data=data, system=scm.system, context=scm.context)
estimator.lcd(corr_test, pcorr_test)
estimator.save_to(outdir)

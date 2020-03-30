from simulator import SCMGenerator, SCMSimulator
from estimator import SCMEstimator
from independence_tests import *
from helpers import set_seed
import shutil
import os

# The number of system variables
p = 4
# The number of context variables
q = 3
# Probability of drawing a latent confounder
eps = 0.0
# Probability of drawing a directed edge
eta = 0.25
# Relation between system variables: 'linear' | 'additive' | 'nonlinear'
rel = 'additive'
# Type of interventions: True: perfect interventions | False: mechanism changes
surgical = True
# Number of samples drawn from each context
N = 100
# The seed for the random number generators
seed = 2

outdir = f'./out/p={p}_q={q}_eps={eps}_eta={eta}' \
         + f'_rel={rel}_surgical={surgical}_N={N}_seed={seed}'

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

set_seed(seed)

generator = SCMGenerator(p, q, eps, eta, surgical, rel)
scm = generator.generate_scm()
scm.save_to(outdir)

simulator = SCMSimulator(scm)
simulator.simulate(N)
simulator.save_to(outdir)

SCMEstimator(
    data=simulator.data,
    system=scm.system,
    context=scm.context,
    alpha=0.01
).lcd_linear().save_to(outdir)

SCMEstimator(
    data=simulator.data,
    system=scm.system,
    context=scm.context,
    alpha=0.01
).lcd_gam_speedup().save_to(outdir)

SCMEstimator(
    data=simulator.data,
    system=scm.system,
    context=scm.context,
    alpha=0.01
).lcd_gam().save_to(outdir)


SCMEstimator(
    data=simulator.data,
    system=scm.system,
    context=scm.context,
    alpha=0.01
).lcd_dhsic_gamgcm().save_to(outdir)

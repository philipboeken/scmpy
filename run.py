from simulator import SCMGenerator, SCMSimulator
# from estimator import SCMEstimator
from helpers import set_seed
import shutil
import os

p = 3  # The number of system variables
q = 2  # The number of context variables
eps = 1  # Probability of drawing a latent confounder
eta = 0.1  # Probability of drawing a directed edge
acyclic = True  # Whether the graph is acyclic
rel = 'linear'  # Relation between system variables: 'linear' | 'additive' | 'nonlinear'
surgical = False  # True: perfect interventions | False: mechanism changes
N = 10  # Number of samples drawn from each context
seed = 1

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

# data = simulator.data
# estimator = SCMEstimator(data=data, system=scm.system, context=scm.context)

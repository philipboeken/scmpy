from simulator import SCMGenerator, SCMSimulator
from helpers import set_seed
import shutil
import os

p = 3  # The number of system variables
q = 2  # The number of context variables
eps = 1  # Probability of drawing a latent confounder
eta = 0.1  # Probability of drawing a directed edge
N = 10  # Number of samples drawn from each context
acyclic = True  # Whether the graph is acyclic
seed = 1
surgical = False  # True: perfect interventions | False: mechanism changes

outdir = f"./out/p={p}_q={q}_eps={eps}_eta={eta}_N={N}" \
         + f"_acyclic={acyclic}_surgical={surgical}_seed={seed}"

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

set_seed(seed)

generator = SCMGenerator(p, q, eps, eta, acyclic, surgical, seed)
scm = generator.generate_scm()
scm.save_to(outdir)

simulator = SCMSimulator(scm)
simulator.simulate(N)
simulator.save_to(outdir)

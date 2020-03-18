from simulator import SCMSimulator
import shutil
import os

p = 5
q = 5
eps = 0.2
eta = 0.3
N = 200
acyclic = 0
surgical = 0
seed = 1

outdir = f"./out/p={p}_q={q}_eps={eps}_eta={eta}_N={N}" \
         + f"_acyclic={acyclic}_surgical={surgical}_seed={seed}"

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

sim = SCMSimulator(p, q, eps, eta, N, acyclic, surgical, seed)
sim.simulate()
sim.saveTo(outdir)

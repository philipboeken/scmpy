from simulator import Simulator
import shutil
import os

p = 5
q = 5
eps = 0.25
eta = 0.1
N = 200
acyclic = 0
surgical = 0
seed = 0
dep = 'linear'

outdir = f"./out/p={p}_q={q}_eps={eps}_eta={eta}_N={N}" \
         + f"_acyclic={acyclic}_surgical={surgical}_seed={seed}" \
         + f"_dep={dep}"

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

sim = Simulator(p, q, eps, eta, N, acyclic, surgical, seed, dep)
sim.simulate()
sim.saveTo(outdir)

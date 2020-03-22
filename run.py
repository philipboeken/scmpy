from simulator import SCMGenerator, SCMSimulator
import shutil
import os

p = 3
q = 4
eps = 0.05
eta = 0.1
N = 10
acyclic = True
surgical = False
seed = 1

outdir = f"./out/p={p}_q={q}_eps={eps}_eta={eta}_N={N}" \
         + f"_acyclic={acyclic}_surgical={surgical}_seed={seed}"

if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.mkdir(outdir)

generator = SCMGenerator(p, q, eps, eta, acyclic, surgical, seed)
scm = generator.generate_scm()
scm.save_to(outdir)

simulator = SCMSimulator(scm)
simulator.simulate(N)
simulator.save_to(outdir)

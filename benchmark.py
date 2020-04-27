from optimization import OptimizationProblem
import numpy as np


def random_problem(T, O):

    op = OptimizationProblem(T, O)
    op.set_bmax(14)
    op.set_bmin(0)
    op.set_charge(0.5)
    op.set_dmax(5)
    op.set_dmin(5)
    op.set_effc(0.95)
    op.set_effd(0.95)

    p_ = np.random.uniform(0, 3, T)
    for o in range(O):
        op.set_prices(o, p_)
        p_ += np.random.uniform(0, 3, T)
    return op
    
T = 48
op = random_problem(T, 2)
%timeit l = np.random.uniform(-3, 3, T); op.set_load(l); op.optimize()

## 40.8 ms per iteration


T = 96
op = random_problem(T, 2)
%timeit l = np.random.uniform(-3, 3, T); op.set_load(l); op.optimize()

## 127 ms per iteration


T = 96
op = random_problem(T, 2)
l = np.random.uniform(-3, 3, T * 20).reshape(20, T)
i = 0
%timeit i=0; op.set_load(l[i]); op.optimize(); i += 1

## 127 ms per iteration

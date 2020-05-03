from structure import init_problem, update_problem
from collections import Counter
import numpy as np
import time


def get_data(T, L):

    #prices = np.zeros((L, 7))
    prices = np.zeros((L, 7))
    for i in range(L): 
        x = np.cumsum(np.random.uniform(0, .5, 2))
        y = [1, 3 - 2 * x[1], 3 - 2 * x[0], 3]
        prices[i, :4] = y
#        prices[i, :4] = [1, 2, 3, 4]

        break_left = np.random.uniform(-0.5, 0, 1)
        break_right = np.random.uniform(0, 0.5, 1)
        prices[i, 4] = break_left
        prices[i, 6] = break_right

    loads = np.random.uniform(-2, 2, L)


    data = {
        'T':          T,
        'num_slopes': 4,
        'efc':        0.95,
        'efd':        0.95,
        'bmax':       13,
        'bmin':       0,
        'charge':     0,
        'dmax':       5,
        'dmin':       5,
        'price': np.zeros((T, 7)),
        'load':  np.zeros(T),
    }

    return data, prices, loads


T, H = 48, 500
D, P, L = get_data(T, H)



def simulate():
    status = []
    times = []
    mo, c_, v_ = init_problem(D)
    for i in range(0, H - T):
        start = time.perf_counter()
        price = P[i: i + T, :]
        load = L[i : i + T]
        D['price'] = price
        D['load'] = load
        mo = update_problem(mo, c_, v_, D)
        sol = mo.solve()
        end = time.perf_counter() - start
        status.append(sol.solve_status)
        times.append(end)
        
        if i % 50 == 0:
            print('ITER', i)
    return status, times

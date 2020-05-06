import pytest
import numpy as np
from structure import build_price, init_problem, update_problem, cleanup_solution

@pytest.fixture
def opt_problem_1():
    """
    Initializes an optimization problem
    with 3 time-slots and a 2-step-objective.
    """

    data = {
    'T':          2,
    'num_slopes': 4,
    'efc':        0.8,
    'efd':        0.9,
    'price':    np.array([[1, 2, 3, 4, -0.5, 0, 0.5],
        [1, 1.5, 3.7, 4, -0.6, 0, 0]]),
    'load':       np.array([-1, 3.1]),
    'bmax':       3,
    'bmin':       0,
    'charge':     0.5,
    'dmax':       5,
    'dmin':       4.9,
    }

    x_ = 1 / (0.8)
    A = np.array([
        [1,   0, -1,  0, 0,   0, 0,       0],
        [0,   1, 0,  -1, 0,   0, 0,       0],
        [2,   0, -1,  0, 0,   0, 0,       0],
        [0, 1.5, 0,  -1, 0,   0, 0,       0],
        [3,   0, -1,  0, 0,   0, 0,       0],
        [0,   3.7, 0,  -1, 0,   0, 0,       0],
        [4,   0, -1,  0, 0,   0, 0,       0],
        [0,   4, 0,  -1, 0,   0, 0,       0], ## End cost function
        [1,   0, 0,   0, -x_,  0, 0.9,    0],
        [0,   1, 0,   0, 0,  -x_, 0,    0.9], ## End def of z
        [0,   0, 0,   0, 1,   0, -1,      0],
        [0,   0, 0,   0, 1,   1, -1,     -1], # End of charge pairs
        [0,   0, 0,   0, 1,   0, 0,       0],
        [0,   0, 0,   0, 0,   1, 0,       0],
        [0,   0, 0,   0, 0,   0, 1,       0],
        [0,   0, 0,   0, 0,   0, 0,       1], # End of max delta
        [1,   0, 0,   0, 0,   0, 0,       0], # Commitment
    ])

    n = - np.infty
    l = np.array([n, n, n, n, n, n, n, n, -1, 3.1, -0.5, -0.5, 0, 0, 0, 0, n])
    u = np.array([0.5, 0, 0, 0.5, 0.3, 0, 0, 0, -1, 3.1, 2.5, 2.5, 5, 5, 4.9, 4.9, np.inf ])
    cost = np.array([0,0,1,1,0,0,0,0])


    return data, (A, l, u, cost)


@pytest.fixture
def opt_problem_2():

    data = {
    'T':          2,
    'num_slopes': 4,
    'efc':        0.8,
    'efd':        0.9,
    'price':    np.array([
        [2, 4, 6, 8, -0.1, 0, 1.1],
        [1, 2, 3, 4, -0.5, 0, 0.5],
        ]),
    'load':       np.array([2, 2,]),
    'bmax':       0,
    'bmin':       0,
    'charge':     0,
    'dmax':       5,
    'dmin':       4.9,
    }

    return data

    
    
def test_simple_opt_1(opt_problem_2):

    data = opt_problem_2 
    mo, c_, v_ = init_problem(data) 
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)
    np.testing.assert_allclose(res['obj'], 21.3, atol=1e-5)
    np.testing.assert_allclose(res['var'], np.array([13.8, 7.5, 0, 0, 0, 0]), atol=1e-5)
    np.testing.assert_allclose(res['net'], np.array([2.0, 2.0]), atol=1e-5)


    data['charge'] = 0.5
    data['bmax'] = 0.5
    mo = update_problem(mo, c_, v_, data)
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)
    np.testing.assert_allclose(res['obj'], 17.7, atol=1e-5)
    np.testing.assert_allclose(res['var'], np.array([10.2, 7.5, 0, 0, 0.5, 0]), atol=1e-5)
    np.testing.assert_allclose(res['net'], np.array([1.55, 2.0]), atol=1e-5)


    data['price'][1] = np.array([2, 2, 3, 3, -1, 0 , 1])
    mo = update_problem(mo, c_, v_, data)
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)
    np.testing.assert_allclose(res['obj'], 16.2, atol=1e-5)
    np.testing.assert_allclose(res['var'], np.array([10.2, 6, 0, 0, 0.5, 0]), atol=1e-5)
    np.testing.assert_allclose(res['net'], np.array([1.55, 2]), atol=1e-5)

    data['commitment'] = 1.7
    mo = update_problem(mo, c_, v_, data)
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)
    np.testing.assert_allclose(res['obj'], 16.95, atol=1e-4)
    np.testing.assert_allclose(res['var'], np.array([11.4, 5.55, 0, 0, 1/3, 1/6]), atol=1e-4)
    np.testing.assert_allclose(res['net'], np.array([1.7, 1.85]), atol=1e-4)


    del data['commitment']
    mo = update_problem(mo, c_, v_, data)
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)
    np.testing.assert_allclose(res['obj'], 16.2, atol=1e-5)
    np.testing.assert_allclose(res['var'], np.array([10.2, 6, 0, 0, 0.5, 0]), atol=1e-5)
    np.testing.assert_allclose(res['net'], np.array([1.55, 2]), atol=1e-5)


def test_simple_opt_2(opt_problem_2):

    data = opt_problem_2 
    data['T'] = 1
    data['price'] = np.array([[1, 1, 3, 3, 0, 0, 0]])
    data['load'] = np.array([-2])
    data['commitment'] = -1

    mo, c_, v_ = init_problem(data) 
    _ = mo.solve()
    res = cleanup_solution(mo, c_, v_, data)

    np.testing.assert_allclose(res['obj'], -2, atol=1e-5)
    np.testing.assert_allclose(res['var'], np.array([-2, 0, 0]), atol=1e-5)
    np.testing.assert_allclose(res['net'], np.array([-2]), atol=1e-5)

import pytest
import numpy as np
from structure import build_A, build_lu, build_price, init_problem, update_problem

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


def test_check_build(opt_problem_1):

    data, (A, l, u, cost) = opt_problem_1
    
    A_, cost_, idx_, idy_ = build_A(**data)

    
    offsets = build_price(data['price'], data['num_slopes']).flatten()

    
    u_, l_ = build_lu(offset=offsets, **data)


    new_prices = data['price'][:, :data['num_slopes']].flatten('F') 
    A_[idx_, idy_] = new_prices

    np.testing.assert_allclose(A, A_.toarray())
    np.testing.assert_allclose(l_, l)
    np.testing.assert_allclose(u_, u)
    
# def test_simple_opt_1(opt_problem_2):

    # data = opt_problem_2 
    # pr = init_problem(data) 
    # sol = pr['osqp'].solve()
    # np.testing.assert_allclose(sol.info.obj_val, 21.3, atol=1e-5)
    # np.testing.assert_allclose(sol.x, np.array([2, 2, 13.8, 7.5, 0, 0, 0, 0]), atol=1e-5)

    # data['charge'] = 0.5
    # data['bmax'] = 0.5
    # pr = update_problem(pr, data)
    # sol = pr['osqp'].solve()
    # np.testing.assert_allclose(sol.info.obj_val, 17.7, atol=1e-5)
    # np.testing.assert_allclose(sol.x, np.array([1.55, 2, 10.2, 7.5, 0, 0, 0.5, 0]), atol=1e-5)

    # data['price'][0] = np.array([2, 2, 3, 3, -1, 0 , 1])
    # pr = update_problem(pr, data)
    # sol = pr['osqp'].solve()
    # np.testing.assert_allclose(sol.info.obj_val, 11.7, atol=1e-5)
    # np.testing.assert_allclose(sol.x, np.array([2, 1.55, 6, 5.7, 0, 0, 0, 0.5]), atol=1e-5)
    
def test_simple_opt_1(opt_problem_2):

    data = opt_problem_2 
    mo, c_, v_ = init_problem(data) 
    sol = mo.solve()
    np.testing.assert_allclose(sol.objective_value, 21.3, atol=1e-5)
    np.testing.assert_allclose(sol.x, np.array([2, 2, 13.8, 7.5, 0, 0, 0, 0]), atol=1e-5)

    data['charge'] = 0.5
    data['bmax'] = 0.5
    mo = update_problem(mo, c_, v_, data)
    sol = mo.solve()

#    pr = update_problem(pr, data)
#    sol = pr['osqp'].solve()
#    np.testing.assert_allclose(sol.info.obj_val, 17.7, atol=1e-5)
#    np.testing.assert_allclose(sol.x, np.array([1.55, 2, 10.2, 7.5, 0, 0, 0.5, 0]), atol=1e-5)

    data['price'][0] = np.array([2, 2, 3, 3, -1, 0 , 1])
    mo = update_problem(mo, c_, v_, data)
    sol = mo.solve()

#    pr = update_problem(pr, data)
#    sol = pr['osqp'].solve()
#    np.testing.assert_allclose(sol.info.obj_val, 11.7, atol=1e-5)
#    np.testing.assert_allclose(sol.x, np.array([2, 1.55, 6, 5.7, 0, 0, 0, 0.5]), atol=1e-5)

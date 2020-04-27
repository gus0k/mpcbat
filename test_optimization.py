import pytest
import numpy as np
from optimization import OptimizationProblem

@pytest.fixture
def opt_problem_1():
    """
    Initializes an optimization problem
    with 3 time-slots and a 2-step-objective.
    """

    op = OptimizationProblem(3, 2)
    op.set_load([-2, 8.0], [0, 1]) 
    op.set_bmax(3)
    op.set_bmin(0)
    op.set_prices_slope(0, [1, 1, 1], [0, 1, 2])
    op.set_charge(0.5)
    op.set_dmax(5)
    op.set_prices_slope(1, [3, 3, 3.1])
    op.set_dmin(4.5)
    op.set_effc(0.9)
    op.set_effd(0.8)

    x_ = 1 / (0.9)
    A = np.array([
        [1, 0, 0, -1, 0, 0, 0, 0, 0], # C - D 0
        [1, 1, 0, -1, -1, 0, 0, 0, 0],
        [1, 1, 1, -1, -1, -1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # C 0 0
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0], # 0 D 0
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [x_, 0, 0, -0.8, 0, 0, -1, 0, 0], ## pC - pD - T (1)
        [0, x_, 0, 0, -0.8, 0, 0, -1, 0],
        [0, 0, x_, 0, 0, -0.8, 0, 0, -1],
        [3* x_, 0, 0, -2.4, 0, 0, -1, 0, 0], ## pC - pD - T (2)
        [0, 3* x_, 0, 0, -2.4, 0, 0, -1, 0],
        [0, 0, 3.1 * x_, 0, 0, -2.48, 0, 0, -1],
    ])

    n = - np.infty
    l = np.array([-.5, -.5, -.5, 0, 0, 0, 0, 0, 0, n, n, n, n, n, n])
    u = np.array([2.5, 2.5, 2.5, 5, 5, 5, 4.5, 4.5, 4.5, 2, -8, 0, 6, -24, 0])
    cost = np.array([0,0,0,0,0,0,1,1,1])


    return op, (A, l, u, cost)


def test_check_build(opt_problem_1):
    """
    Builds a simple optimization problem
    and checks that all the parameters are
    have been correctly setup
    """
    op, (A_, l_, u_, cost_) = opt_problem_1
    A = op.A
    u = op._build_u()
    l = op._build_l()
    cost = op.cost
    np.testing.assert_allclose(A, A_)
    np.testing.assert_allclose(l, l_)
    np.testing.assert_allclose(u, u_)
    np.testing.assert_allclose(cost, cost_)
    r = op.optimize()
    np.testing.assert_allclose(r.info.obj_val, 18.48)
    bat_change = r.x[:3] - r.x[3:6]
    np.testing.assert_allclose(bat_change, np.array([1.8, -2.3, 0]), atol=1e-8)


def test_complex_objective(opt_problem_1):

    """
    Testing with complex objective function
    """

    op = OptimizationProblem(2, 4)
    op.set_bmax(2)
    op.set_bmin(0)
    op.set_charge(0)
    op.set_bmax(2)
    op.set_bmin(2)
    op.set_load(np.array([1, 1]))



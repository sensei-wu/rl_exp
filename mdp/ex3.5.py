import numpy as np

# ----- Environment definition (Sutton & Barto, Example 3.5 / Figure 3.2) -----

N = 5
A, A_PRIME, R_A = (0, 1), (4, 1), 10
B, B_PRIME, R_B = (0, 3), (2, 3), 5

ACTIONS = {
    'north': (-1, 0),
    'south': (1, 0),
    'east':  (0, 1),
    'west':  (0, -1),
}

STATES = [(r, c) for r in range(N) for c in range(N)]
IDX = {s: i for i, s in enumerate(STATES)}


def step(state, action):
    """Deterministic environment dynamics for one action from `state`."""
    if state == A:
        return A_PRIME, R_A
    if state == B:
        return B_PRIME, R_B

    dr, dc = ACTIONS[action]
    r, c = state
    nr, nc = r + dr, c + dc

    if 0 <= nr < N and 0 <= nc < N:
        return (nr, nc), 0
    else:
        return state, -1  # off-grid: bounce back, penalty


def build_policy_model():
    """
    Build the policy-averaged reward vector R^pi and transition matrix P^pi
    for the equiprobable random policy (used by both solution methods).
    """
    P = np.zeros((N * N, N * N))
    R = np.zeros(N * N)

    for s in STATES:
        i = IDX[s]
        for a in ACTIONS:
            s_next, r = step(s, a)
            P[i, IDX[s_next]] += 0.25
            R[i] += 0.25 * r

    return R, P


# ----- Method 1: direct linear solve -----

def solve_linear(gamma=0.9):
    """
    Solve (I - gamma * P^pi) v = R^pi directly.
    Exact up to floating-point precision; no iteration needed.
    """
    R, P = build_policy_model()
    v = np.linalg.solve(np.eye(N * N) - gamma * P, R)
    return v.reshape(N, N)


# ----- Method 2: iterative policy evaluation -----

def solve_iterative(gamma=0.9, theta=1e-8, max_iters=10_000):
    """
    Iterative policy evaluation: repeatedly apply the Bellman expectation
    backup v <- R^pi + gamma * P^pi @ v until the change is below theta.
    Converges to the same fixed point as solve_linear (contraction mapping,
    since gamma < 1 and P^pi is a stochastic matrix).
    """
    R, P = build_policy_model()
    v = np.zeros(N * N)

    for i in range(max_iters):
        v_new = R + gamma * (P @ v)
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < theta:
            break

    return v.reshape(N, N), i + 1  # also return iterations to convergence


if __name__ == "__main__":
    v_direct = solve_linear()
    v_iter, n_iters = solve_iterative()

    print("Direct linear solve:")
    print(np.round(v_direct, 1))
    print()
    print(f"Iterative policy evaluation (converged in {n_iters} iterations):")
    print(np.round(v_iter, 1))
    print()
    print("Max abs difference between methods:",
          np.max(np.abs(v_direct - v_iter)))
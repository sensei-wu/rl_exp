import gymnasium as gym
import numpy as np


def run():
    import time

    # Solve
    env = gym.make("Taxi-v3", render_mode="human")
    initV = np.zeros(env.observation_space.n)
    _, pi = value_iteration(env, initV, 1e-12, gamma=0.99)
    state, _ = env.reset()

    done = terminated = False
    G = 0.0
    step = 0

    while not done and not terminated:
        action = pi[state]
        state, R, done, terminated, _ = env.step(action)
        #os.system("clear")
        #env.render()
        print(f"step={step:3d}  action={action}  reward={R:+.0f}")
        time.sleep(0.5)
        step += 1
        G += R

    print(f"\nTotal reward: {G}")
    env.close()

def value_iteration(env, V, tol=1e-12, gamma=0.9):
    """
    Value iteration algorithm for solving MDPs.

    Args:
        env: The environment to solve.
        V: Initial value function.
        tol: Tolerance for convergence.
        gamma: Discount factor.
        """
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n))
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            break

    # Extract policy
    pi = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax([sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(env.action_space.n)])

    return V, pi


if __name__ == "__main__":
    run()
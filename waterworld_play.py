import numpy as np

from waterworld import env as Waterworld


def main():
    env = Waterworld()
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = 0.005 * (2 * np.random.random(2) - 1)
        env.step(action)
        env.render()


if __name__ == "__main__":
    main()

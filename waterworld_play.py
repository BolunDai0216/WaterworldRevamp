import numpy as np

from waterworld import env as Waterworld


def main():
    env = Waterworld(render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        env.last()
        action = 0.01 * (2 * np.random.random(2) - 1)
        env.step(action)

        if agent == "pursuer_0":
            env.render()


if __name__ == "__main__":
    main()

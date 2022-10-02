import numpy as np

from waterworld import parallel_env as Waterworld


def main():
    env = Waterworld(render_mode="human")
    env.reset()
    max_cycles = 500

    for step in range(max_cycles):
        actions = {agent: 0.01 * (2 * np.random.random(2) - 1) for agent in env.agents}
        env.step(actions)
        env.render()


if __name__ == "__main__":
    main()

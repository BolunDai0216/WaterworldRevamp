from pdb import set_trace

import numpy as np

from waterworld import parallel_env as Waterworld


def main():
    env = Waterworld(render_mode="human")
    env.reset()
    max_cycles = 500

    for step in range(max_cycles):
        env.render()
        actions = {agent: 0.01 * (2 * np.random.random(2) - 1) for agent in env.agents}
        res = env.step(actions)

        set_trace()


if __name__ == "__main__":
    main()

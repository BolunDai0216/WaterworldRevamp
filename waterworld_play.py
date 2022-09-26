import numpy as np

from waterworld_base import WaterworldBase


def main():
    base = WaterworldBase(obstacle_coord=None)
    base.reset()

    for j in range(1000):
        obs = base.render(mode="human")

        for i, p in enumerate(base.pursuers):
            if i == 4:
                islast = True
            else:
                islast = False

            action = 1000 * (2 * np.random.random(2) - 1)
            base.step(action, i, islast)

    base.close()


if __name__ == "__main__":
    main()

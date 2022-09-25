import numpy as np
import pygame

from waterworld_base import WaterworldBase


def main():
    base = WaterworldBase(obstacle_coord=None)
    base.reset()

    for j in range(10000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        for i, p in enumerate(base.pursuers):
            if i == 4:
                islast = True
            else:
                islast = False

            action = 1000 * (2 * np.random.random(2) - 1)
            base.step(action, i, islast)

    pygame.quit()


if __name__ == "__main__":
    main()

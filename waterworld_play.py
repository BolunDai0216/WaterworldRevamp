from pdb import set_trace

import numpy as np
from pettingzoo.sisl import waterworld_v4


def main():
    env = waterworld_v4.env(
        render_mode="human", n_coop=1, n_pursuers=1, n_evaders=6, speed_features=False
    )
    env.reset()
    counter = 0

    for agent in env.agent_iter():
        if agent == "pursuer_0":
            env.render()

        observation, reward, termination, truncation, info = env.last()
        action = 0.01 * (2 * np.random.random(2) - 1)
        env.step(action)

        # set_trace()

        if termination:
            break
        counter += 1

        if counter >= 1000:
            break

        # print(
        #     # f"Reward: {reward}, food: {env.unwrapped.env.pursuers[0].shape.food_indicator}, poison: {env.unwrapped.env.pursuers[0].shape.poison_indicator}"
        #     f"_food: {env.unwrapped.env.pursuers[0].shape.food_touched_indicator}"
        # )

        # if reward > 0:
        # set_trace()

    env.reset()
    counter = 0

    for agent in env.agent_iter():
        if agent == "pursuer_0":
            env.render()

        observation, reward, termination, truncation, info = env.last()
        action = 0.01 * (2 * np.random.random(2) - 1)
        env.step(action)

        if termination:
            break

        counter += 1
        if counter >= 100:
            break


if __name__ == "__main__":
    main()

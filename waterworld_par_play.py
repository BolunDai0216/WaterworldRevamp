from pdb import set_trace

import numpy as np

from waterworld import parallel_env as Waterworld


def main():
    env = Waterworld(render_mode="human", speed_features=False)
    env.reset()
    max_cycles = 500

    for step in range(max_cycles):
        env.render()
        actions = {agent: 0.01 * (2 * np.random.random(2) - 1) for agent in env.agents}
        res = env.step(actions)

        # # Check if all observation is contained within limit
        # p_obstacle_pos_max = np.amax(res[0]["pursuer_0"][:30]) <= 1.0
        # p_food_pos_max = np.amax(res[0]["pursuer_0"][60:90]) <= 1.0
        # p_food_vel_max = np.amax(
        #     np.absolute(res[0]["pursuer_0"][90:120])
        # ) <= 2 * np.sqrt(2)
        # p_poison_pos_max = np.amax(res[0]["pursuer_0"][120:150]) <= 1.0
        # p_poison_vel_max = np.amax(
        #     np.absolute(res[0]["pursuer_0"][150:180])
        # ) <= 2 * np.sqrt(2)
        # p_pursuer_pos_max = np.amax(res[0]["pursuer_0"][180:210]) <= 1.0
        # p_pursuer_vel_max = np.amax(
        #     np.absolute(res[0]["pursuer_0"][210:240])
        # ) <= 2 * np.sqrt(2)

        # if not (
        #     p_obstacle_pos_max
        #     & p_food_pos_max
        #     & p_food_vel_max
        #     & p_poison_pos_max
        #     & p_poison_vel_max
        #     & p_pursuer_pos_max
        #     & p_pursuer_vel_max
        # ):
        #     set_trace()
        #     base_env = env.aec_env.env.env.env

        #     for food in base_env.evaders:
        #         print(food.body.velocity)

        #     for poison in base_env.poisons:
        #         print(poison.body.velocity)

        #     for pursuer in base_env.pursuers:
        #         print(pursuer.body.velocity)

        # Check if all observation is contained within limit
        p_obstacle_pos_max = np.amax(res[0]["pursuer_0"][:30]) <= 1.0
        p_food_pos_max = np.amax(res[0]["pursuer_0"][60:90]) <= 1.0
        p_poison_pos_max = np.amax(res[0]["pursuer_0"][90:120]) <= 1.0
        p_pursuer_pos_max = np.amax(res[0]["pursuer_0"][120:150]) <= 1.0

        if not (
            p_obstacle_pos_max & p_food_pos_max & p_poison_pos_max & p_pursuer_pos_max
        ):
            set_trace()
            base_env = env.aec_env.env.env.env


if __name__ == "__main__":
    main()

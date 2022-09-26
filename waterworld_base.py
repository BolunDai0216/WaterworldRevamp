import math
import random
from pdb import set_trace

import numpy as np
import pygame
import pymunk
from gym.utils import seeding
from scipy.spatial import distance as ssd

from waterworld_models import Evaders, Obstacle, Poisons, Pursuers


class WaterworldBase:
    def __init__(
        self,
        n_pursuers=5,
        n_evaders=5,
        n_poisons=10,
        n_obstacles=2,
        n_coop=1,
        n_sensors=30,
        sensor_range=0.2,
        radius=0.015,
        obstacle_radius=0.1,
        obstacle_coord=[(0.5, 0.5)],
        pursuer_max_accel=0.01,
        evader_speed=0.1,
        poison_speed=0.1,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        local_ratio=1.0,
        speed_features=True,
        max_cycles=500,
    ):
        """
        ╭──────────────────────────────────────────────────────────────────────────────────────────────────────╮
        │Input keyword arguments                                                                               │
        ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │n_pursuers: number of pursuing archea (agents)                                                        │
        │n_evaders: number of evader archea (food)                                                             │
        │n_poisons: number of poison archea                                                                    │
        │n_obstacles: number of obstacles                                                                      │
        │n_coop: number of pursuing archea (agents) that must be touching food at the same time to consume it  │
        │n_sensors: number of sensors on each of the pursuing archea (agents)                                  │
        │sensor_range: length of sensor dendrite on all pursuing archea (agents)                               │
        │radius: archea base radius. Pursuer: radius, evader: 2 x radius, poison: 3/4 x radius                 │
        │obstacle_radius: radius of obstacle object                                                            │
        │evader_speed: evading archea speed                                                                    │
        │poison_speed: poison archea speed                                                                     │
        │obstacle_coord: list of coordinate of obstacle object. Can be set to `None` to use a random location  │
        │speed_features: toggles whether pursuing archea (agent) sensors detect speed of other archea          │
        │pursuer_max_accel: pursuer archea maximum acceleration (maximum action size)                          │
        │thrust_penalty: scaling factor for the negative reward used to penalize large actions                 │
        │local_ratio: proportion of reward allocated locally vs distributed globally among all agents          │
        │food_reward: reward for pursuers consuming an evading archea                                          │
        │poison_reward: reward for pursuer consuming a poison object (typically negative)                      │
        │encounter_reward: reward for a pursuer colliding with an evading archea                               │
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
        """
        pygame.init()
        self.pixel_scale = 30 * 25

        self.display = pygame.display.set_mode((self.pixel_scale, self.pixel_scale))
        self.clock = pygame.time.Clock()
        self.FPS = 15  # Frames Per Second

        self.handlers = []
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poisons = n_poisons
        self.n_obstacles = n_obstacles
        self.n_coop = n_coop
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range
        self.base_radius = radius
        self.obstacle_radius = obstacle_radius
        self.evader_speed = evader_speed * self.pixel_scale
        self.poison_speed = poison_speed * self.pixel_scale
        self.speed_features = speed_features
        self.pursuer_max_accel = pursuer_max_accel
        self.thrust_penalty = thrust_penalty
        self.local_ratio = local_ratio
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.encounter_reward = encounter_reward
        self.max_cycles = max_cycles

        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]

        if obstacle_coord is not None and len(obstacle_coord) != self.n_obstacles:
            raise ValueError("obstacle_coord does not have same length as n_obstacles")
        else:
            self.initial_obstacle_coord = obstacle_coord

        self.frames = 0

        self.seed()
        self.add_obj()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        self.pursuers = []
        self.evaders = []
        self.poisons = []
        self.obstacles = []

        for i in range(self.n_pursuers):
            x, y = self._generate_coord(self.base_radius)
            self.pursuers.append(
                Pursuers(
                    x,
                    y,
                    self.pursuer_max_accel,
                    radius=self.base_radius,
                    collision_type=i + 1,
                    n_sensors=self.n_sensors,
                    sensor_range=self.sensor_range,
                )
            )

        for i in range(self.n_evaders):
            x, y = self._generate_coord(2 * self.base_radius)
            self.evaders.append(
                Evaders(
                    x,
                    y,
                    radius=2 * self.base_radius,
                    collision_type=i + 1000,
                    max_speed=self.evader_speed,
                )
            )

        for i in range(self.n_poisons):
            x, y = self._generate_coord(0.75 * self.base_radius)
            self.poisons.append(
                Poisons(
                    x,
                    y,
                    radius=0.75 * self.base_radius,
                    collision_type=i + 2000,
                    max_speed=self.poison_speed,
                )
            )

        for _ in range(self.n_obstacles):
            self.obstacles.append(
                Obstacle(
                    self.pixel_scale / 2,
                    self.pixel_scale / 2,
                    radius=self.obstacle_radius,
                )
            )

    def convert_coordinates(self, value, option="position"):
        """
        This function converts coordinates in pymunk into pygame coordinates.

        The coordinate system in pygame is:

                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
                        |       |                           │
                        |       |                           │
        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↓ y

        The coordinate system in pymunk is:

        (0, WINDOWSIZE) +-------+ (WINDOWSIZE, WINDOWSIZE)  ↑ y
                        |       |                           │
                        |       |                           │
                 (0, 0) +-------+ (WINDOWSIZE, 0)           + ──── → x
        """

        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])

        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self, radius):
        """
        radius: radius of the object
        """
        coord = self.np_random.rand(2) * self.pixel_scale

        # Create random coordinate that avoids obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.body.position
            while (
                ssd.cdist(coord[None, :], np.array([[x, y]]))
                <= radius * 2 + obstacle.radius
            ):
                coord = self.np_random.rand(2)

        return coord

    def _generate_speed(self, speed):
        _speed = (self.np_random.rand(2) - 0.5) * 2 * speed

        return _speed[0], _speed[1]

    def add(self):
        self.space = pymunk.Space()

        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)

    def add_bounding_box(self):
        # Bounding Box
        pts = [
            (-100, -100),
            (self.pixel_scale + 100, -100),
            (self.pixel_scale + 100, self.pixel_scale + 100),
            (-100, self.pixel_scale + 100),
        ]

        self.barriers = []

        for i in range(4):
            self.barriers.append(
                pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 100)
            )
            self.barriers[-1].elasticity = 0.999
            self.space.add(self.barriers[-1])

    def draw(self):
        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.draw(self.display, self.convert_coordinates)

    def add_handlers(self):
        # Collision handlers for pursuers v.s. evaders & poisons
        for pursuer in self.pursuers:
            for obj_list in [self.evaders, self.poisons]:
                for obj in obj_list:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            pursuer.shape.collision_type, obj.shape.collision_type
                        )
                    )

        for i in range(self.n_pursuers):
            for j in range(self.n_evaders):
                idx = i * (self.n_evaders + self.n_poisons) + j
                self.handlers[idx].begin = self.pursuer_evader_begin_callback
                self.handlers[idx].separate = self.pursuer_evader_separate_callback

            for k in range(self.n_poisons):
                idx = i * (self.n_evaders + self.n_poisons) + self.n_evaders + k
                self.handlers[idx].begin = self.pursuer_poison_begin_callback

        # Collision handlers for poisons v.s. evaders
        for poison in self.poisons:
            for evader in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        poison.shape.collision_type, evader.shape.collision_type
                    )
                )
                self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for evaders v.s. evaders
        for i in range(self.n_evaders):
            for j in range(i, self.n_evaders):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.evaders[i].shape.collision_type,
                            self.evaders[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for poisons v.s. poisons
        for i in range(self.n_poisons):
            for j in range(i, self.n_poisons):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.poisons[i].shape.collision_type,
                            self.poisons[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

        # Collision handlers for pursuers v.s. pursuers
        for i in range(self.n_pursuers):
            for j in range(i, self.n_pursuers):
                if not i == j:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            self.pursuers[i].shape.collision_type,
                            self.pursuers[j].shape.collision_type,
                        )
                    )
                    self.handlers[-1].begin = self.return_false_begin_callback

    def reset(self):
        self.frames = 0

        # Initialize obstacles positions
        if self.initial_obstacle_coord is None:
            for i, obstacle in enumerate(self.obstacles):
                obstacle_position = (
                    self.np_random.rand(self.n_obstacles, 2) * self.pixel_scale
                )
                obstacle.body.position = (
                    obstacle_position[0, 0],
                    obstacle_position[0, 1],
                )
        else:
            for i, obstacle in enumerate(self.obstacles):
                obstacle.body.position = (
                    self.initial_obstacle_coord[i][0] * self.pixel_scale,
                    self.initial_obstacle_coord[i][1] * self.pixel_scale,
                )

        # Add objects to space
        self.add()
        self.add_handlers()
        self.add_bounding_box()

        # Get observation
        obs_list = self.observe_list()

        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.behavior_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = obs_list

        return obs_list[0]

    def step(self, action, agent_id, is_last):
        action = np.asarray(action)
        action = action.reshape(2)
        speed = np.linalg.norm(action)
        if speed > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action / speed * self.pursuer_max_accel

        p = self.pursuers[agent_id]
        _velocity = p.body.velocity + action * self.pixel_scale
        p.reset_velocity(_velocity[0], _velocity[1])

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        # Average thrust penalty among all agents, and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (
            (accel_penalty / self.n_pursuers)
            * np.ones(self.n_pursuers)
            * (1 - self.local_ratio)
        )

        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        if is_last:
            # Step environment
            self.display.fill((255, 255, 255))
            self.draw()
            pygame.display.update()
            self.clock.tick(self.FPS)
            self.space.step(1 / self.FPS)

            for id in range(self.n_pursuers):
                p = self.pursuers[agent_id]

                # reward for food caught, encountered and poison
                self.behavior_rewards[id] = (
                    self.food_reward * p.shape.food_indicator
                    + self.encounter_reward * p.shape.food_touched_indicator
                    + self.poison_reward * p.shape.poison_indicator
                )

                p.shape.food_indicator = 0
                p.shape.food_touched_indicator = 0
                p.shape.poison_indicator = 0

            rewards = np.array(self.behavior_rewards) + np.array(self.control_rewards)

            obs_list = self.observe_list()
            self.last_obs = obs_list

            local_reward = rewards
            global_reward = local_reward.mean()

            # # Distribute local and global rewards according to local_ratio
            self.last_rewards = local_reward * self.local_ratio + global_reward * (
                1 - self.local_ratio
            )

            self.frames += 1

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observe_list = []

        for i, pursuer in enumerate(self.pursuers):
            obstacle_distances = []

            evader_distances = []
            evader_velocities = []

            poison_distances = []
            poison_velocities = []

            _pursuer_distances = []
            _pursuer_velocities = []

            for obstacle in self.obstacles:
                obstacle_distance, _ = pursuer.get_sensor_reading(
                    obstacle.body.position, obstacle.radius, obstacle.body.velocity
                )
                obstacle_distances.append(obstacle_distance)

            obstacle_sensor_vals = (
                np.amin(np.concatenate(obstacle_distances, axis=1), axis=1)
                / pursuer.sensor_range
            )

            barrier_distances = pursuer.get_sensor_barrier_readings()

            for evader in self.evaders:
                evader_distance, evader_velocity = pursuer.get_sensor_reading(
                    evader.body.position, evader.radius, evader.body.velocity
                )
                evader_distances.append(evader_distance)
                evader_velocities.append(evader_velocity)

            evader_distance_vals = np.concatenate(evader_distances, axis=1)
            evader_velocity_vals = np.concatenate(evader_velocities, axis=1)

            evader_min_idx = np.argmin(evader_distance_vals, axis=1)
            evader_sensor_distance_vals = (
                np.amin(evader_distance_vals, axis=1) / pursuer.sensor_range
            )
            evader_sensor_velocity_vals = evader_velocity_vals[
                np.arange(self.n_sensors), evader_min_idx
            ]

            for poison in self.poisons:
                poison_distance, poison_velocity = pursuer.get_sensor_reading(
                    poison.body.position, poison.radius, poison.body.velocity
                )
                poison_distances.append(poison_distance)
                poison_velocities.append(poison_velocity)

            poison_distance_vals = np.concatenate(poison_distances, axis=1)
            poison_velocity_vals = np.concatenate(poison_velocities, axis=1)

            poison_min_idx = np.argmin(poison_distance_vals, axis=1)
            poison_sensor_distance_vals = (
                np.amin(poison_distance_vals, axis=1) / pursuer.sensor_range
            )
            poison_sensor_velocity_vals = poison_velocity_vals[
                np.arange(self.n_sensors), poison_min_idx
            ]

            for j, _pursuer in enumerate(self.pursuers):
                if i == j:
                    continue

                _pursuer_distance, _pursuer_velocity = pursuer.get_sensor_reading(
                    _pursuer.body.position, _pursuer.radius, _pursuer.body.velocity
                )
                _pursuer_distances.append(_pursuer_distance)
                _pursuer_velocities.append(_pursuer_velocity)

            _pursuer_distance_vals = np.concatenate(_pursuer_distances, axis=1)
            _pursuer_velocity_vals = np.concatenate(_pursuer_velocities, axis=1)

            _pursuer_min_idx = np.argmin(_pursuer_distance_vals, axis=1)
            _pursuer_sensor_distance_vals = (
                np.amin(_pursuer_distance_vals, axis=1) / pursuer.sensor_range
            )
            _pursuer_sensor_velocity_vals = _pursuer_velocity_vals[
                np.arange(self.n_sensors), _pursuer_min_idx
            ]

            if self.speed_features:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        evader_sensor_velocity_vals,
                        poison_sensor_distance_vals,
                        poison_sensor_velocity_vals,
                        _pursuer_sensor_distance_vals,
                        _pursuer_sensor_velocity_vals,
                        np.array([pursuer.shape.food_indicator]),
                        np.array([pursuer.shape.poison_indicator]),
                    ]
                )
            else:
                pursuer_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        poison_sensor_distance_vals,
                        _pursuer_sensor_distance_vals,
                        np.array([pursuer.shape.food_indicator]),
                        np.array([pursuer.shape.poison_indicator]),
                    ]
                )

            observe_list.append(pursuer_observation)

        return observe_list

    def pursuer_poison_begin_callback(self, arbiter, space, data):
        """
        Called when a collision between a pursuer and a poison occurs.

        The poison indicator of the pursuer becomes 1, the pursuer gets
        a penalty for this step.
        """
        pursuer_shape, poison_shape = arbiter.shapes

        # For giving reward to pursuer
        pursuer_shape.poison_indicator += 1

        # Reset poision position & velocity
        x, y = self._generate_coord(poison_shape.radius)
        vx, vy = self._generate_speed(poison_shape.max_speed)

        poison_shape.reset_position(x, y)
        poison_shape.reset_velocity(vx, vy)

        return False

    def pursuer_evader_begin_callback(self, arbiter, space, data):
        """
        Called when a collision between a pursuer and an evader occurs.

        The counter of the evader increases by 1, if the counter reaches
        n_coop, then the pursuer catches the evader and gets a reward.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        # Add one collision to evader
        evader_shape.counter += 1

        # Indicate that food is touched by pursuer
        pursuer_shape.food_touched_indicator = 1

        if evader_shape.counter >= self.n_coop:
            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1

        return False

    def pursuer_evader_separate_callback(self, arbiter, space, data):
        """
        Called when a collision between a pursuer and a poison ends.

        If at this moment there are greater or equal then n_coop pursuers
        that collides with this evader, the evader's position gets reset
        and the pursuers involved will be rewarded.
        """
        pursuer_shape, evader_shape = arbiter.shapes

        if evader_shape.counter < self.n_coop:
            # Remove one collision from evader
            evader_shape.counter -= 1
        else:
            evader_shape.counter = 0

            # For giving reward to pursuer
            pursuer_shape.food_indicator = 1

            # Reset evader position & velocity
            x, y = self._generate_coord(evader_shape.radius)
            vx, vy = self._generate_speed(evader_shape.max_speed)

            evader_shape.reset_position(x, y)
            evader_shape.reset_velocity(vx, vy)

    def return_false_begin_callback(self, arbiter, space, data):
        """
        Callback function that simply returns False.
        """
        return False

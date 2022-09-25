import math
import random
from pdb import set_trace

import numpy as np
import pygame
import pymunk
from gym.utils import seeding
from scipy.spatial import distance as ssd


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
                    radius=self.base_radius,
                    collision_type=i + 1,
                    _n_sensors=self.n_sensors,
                    _sensor_range=self.sensor_range,
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
            (0, 0),
            (self.pixel_scale, 0),
            (self.pixel_scale, self.pixel_scale),
            (0, self.pixel_scale),
        ]
        self.barriers = []

        for i in range(4):
            self.barriers.append(
                pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 2)
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

        # Initial reward is zero
        reward = np.zeros(self.n_pursuers)

        # Add objects to space
        self.add()
        self.add_handlers()
        self.add_bounding_box()

        # Get observation
        obs_list = self.observe_list()

        return 0.0

    def step(self, action, agent_id, is_last):
        action = np.asarray(action)
        action = action.reshape(2)
        speed = np.linalg.norm(action)
        if speed > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action / speed * self.pursuer_max_accel

        p = self.pursuers[agent_id]
        _velocity = p.velocity + action
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
            # TODO: add actual rewards
            rewards = 0.0

            obs_list = self.observe_list()
            self.last_obs = obs_list

            local_reward = rewards
            global_reward = local_reward.mean()

            # Distribute local and global rewards according to local_ratio
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
                / pursuer._sensor_range
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
                np.amin(evader_distance_vals, axis=1) / pursuer._sensor_range
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
                np.amin(poison_distance_vals, axis=1) / pursuer._sensor_range
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
                np.amin(_pursuer_distance_vals, axis=1) / pursuer._sensor_range
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


class Obstacle:
    def __init__(self, x, y, pixel_scale=750, radius=0.1):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.body.position = x, y
        self.body.velocity = 0.0, 0.0

        self.shape = pymunk.Circle(self.body, pixel_scale * 0.1)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.custom_value = 1

        self.radius = radius * pixel_scale
        self.color = (120, 176, 178)

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class MovingObject:
    def __init__(self, x, y, pixel_scale=750, radius=0.015):
        self.pixel_scale = 30 * 25
        self.body = pymunk.Body()
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity

        self.radius = radius * pixel_scale

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy


class Evaders(MovingObject):
    def __init__(self, x, y, radius=0.03, collision_type=2, max_speed=100):
        super().__init__(x, y, radius=radius)

        vx = max_speed * random.uniform(-1, 1)
        vy = max_speed * random.uniform(-1, 1)
        self.body.velocity = vx, vy

        self.color = (238, 116, 106)
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.max_speed = max_speed


class Poisons(MovingObject):
    def __init__(self, x, y, radius=0.015 * 3 / 4, collision_type=3, max_speed=100):
        super().__init__(x, y, radius=radius)

        vx = max_speed * random.uniform(-1, 1)
        vy = max_speed * random.uniform(-1, 1)
        self.body.velocity = vx, vy

        self.color = (145, 250, 116)
        self.shape.collision_type = collision_type
        self.shape.max_speed = max_speed


class Pursuers(MovingObject):
    def __init__(
        self, x, y, radius=0.015, _n_sensors=30, _sensor_range=0.2, collision_type=1
    ):
        super().__init__(x, y, radius=radius)

        self.color = (101, 104, 249)
        self.shape.collision_type = collision_type
        self.sensor_color = (0, 0, 0)
        self._n_sensors = _n_sensors
        self._sensor_range = _sensor_range * self.pixel_scale
        self.shape.food_indicator = 0  # 1 if food caught this step, 0 otherwise
        self.shape.poison_indicator = 0  # 1 if poisoned this step, 0 otherwise

        # Generate self._n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self._n_sensors + 1)[:-1]
        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors
        self.shape.custom_value = 1

    def set_velocity(self, velocity):
        assert velocity.shape == (2,)
        self.body.velocity = velocity

    def draw(self, display, convert_coordinates):
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self._sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_barrier_readings(self):
        sensor_vectors = self._sensors * self._sensor_range
        position_vec = np.array([self.body.position.x, self.body.position.y])
        sensor_endpoints = position_vec + sensor_vectors

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, 0.0, self.pixel_scale)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - position_vec

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio will limit the end of the vector to the barriers
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=np.abs(sensor_vectors) > 1e-8,
        )

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to infinity
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = np.inf

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0

        return sensor_values[0, :]

    def get_sensor_reading(self, object_coord, object_radius, object_velocity):
        # Get location and velocity of pursuer
        self.center = self.body.position
        self.velocity = self.body.velocity

        # Get distance of object in local frame as a 2x1 numpy array
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # Get relative velocity as a 2x1 numpy array
        relative_speed = np.array(
            [
                [object_velocity[0] - self.velocity[0]],
                [object_velocity[1] - self.velocity[1]],
            ]
        )

        # Project distance to sensor vectors
        sensor_distances = self._sensors @ distance_vec

        # Project velocity vector to sensor vectors
        sensor_velocities = self._sensors @ relative_speed

        # Check for valid detection criterions
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self._sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # Set not sensed sensor readings of position to sensor range
        sensor_distances[not_sensed_idx] = self._sensor_range

        # Set not sensed sensor readings of velocity to zero
        sensor_velocities[not_sensed_idx] = 0.0

        return sensor_distances, sensor_velocities


def main():
    for i in range(3):
        base = WaterworldBase(obstacle_coord=None)
        base.reset()

        for j in range(200):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            base.display.fill((255, 255, 255))
            base.draw()
            pygame.display.update()
            base.clock.tick(base.FPS)
            base.space.step(1 / base.FPS)

            # for p in base.pursuers:
            #     _p_sensor_vals = []
            #     for e in base.evaders:
            #         e_coord = base.convert_coordinates(e.body.position)
            #         e_vel = base.convert_coordinates(e.body.velocity, option="velocity")

            #         sensor_distance_e, sensor_velocity_e = p.get_sensor_reading(
            #             e_coord, e.radius, e_vel, base.convert_coordinates
            #         )
            #         _p_sensor_vals.append(sensor_distance_e)
            #     p_sensor_vals = np.amin(np.concatenate(_p_sensor_vals, axis=1), axis=1)
            obs_list = base.observe_list()

    pygame.quit()


if __name__ == "__main__":
    main()


# [x] Making food disappear on colliding with more than n_coop objects at once
# [x] Draw spokes on pursuers
# [ ] Set categories and masks for objects

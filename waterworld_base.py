import random
from pdb import set_trace

import numpy as np
import pygame
import pymunk
from gym.utils import seeding
from scipy.spatial import distance as ssd


class WaterworldBase:
    def __init__(self, n_pursuers=5, n_evaders=5, n_poisons=10, n_obstacles=1):
        pygame.init()
        self.pixel_scale = 30 * 25

        self.display = pygame.display.set_mode((self.pixel_scale, self.pixel_scale))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.FPS = 15  # Frames Per Second

        self.handlers = []
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poisons = n_poisons
        self.n_obstacles = n_obstacles
        self.n_coop = 1

        self.add_obj()
        self.add()
        self.add_handlers()

        # Bounding Box
        pts = [
            (0, 0),
            (self.pixel_scale, 0),
            (self.pixel_scale, self.pixel_scale),
            (0, self.pixel_scale),
        ]
        for i in range(4):
            seg = pymunk.Segment(self.space.static_body, pts[i], pts[(i + 1) % 4], 2)
            seg.elasticity = 0.999
            self.space.add(seg)

        for i in range(self.n_pursuers):
            for j in range(self.n_evaders):
                idx = i * (self.n_evaders + self.n_poisons) + j
                self.handlers[idx].begin = self.pursuer_evader_begin_callback
                self.handlers[idx].separate = self.pursuer_evader_separate_callback

            for k in range(self.n_poisons):
                idx = i * (self.n_evaders + self.n_poisons) + self.n_evaders + k
                self.handlers[idx].begin = self.pursuer_poison_begin_callback

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        self.pursuers = []
        self.evaders = []
        self.poisons = []
        self.obstacles = []

        for i in range(self.n_pursuers):
            self.pursuers.append(
                Pursuers(
                    random.randint(0, self.pixel_scale),
                    random.randint(0, self.pixel_scale),
                    collision_type=i + 1,
                )
            )

        for i in range(self.n_evaders):
            self.evaders.append(
                Evaders(
                    random.randint(0, self.pixel_scale),
                    random.randint(0, self.pixel_scale),
                    collision_type=i + 1000,
                )
            )

        for i in range(self.n_poisons):
            self.poisons.append(
                Poisons(
                    random.randint(0, self.pixel_scale),
                    random.randint(0, self.pixel_scale),
                    collision_type=i + 2000,
                )
            )

        for _ in range(self.n_obstacles):
            self.obstacles.append(Obstacle(self.pixel_scale / 2, self.pixel_scale / 2))

    def convert_coordinates(self, value, option="position"):
        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])
        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self, radius):
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
        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)

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
        pass

    def step(self):
        pass

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
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)

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

        self.color = (238, 116, 106)
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.max_speed = max_speed


class Poisons(MovingObject):
    def __init__(self, x, y, radius=0.015 * 3 / 4, collision_type=3, max_speed=100):
        super().__init__(x, y, radius=radius)

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

    def draw(self, display, convert_coordinates):
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self._sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_reading(
        self, object_coord, object_radius, object_velocity, convert_coordinates
    ):
        # Get location and velocity of pursuer
        self.center = convert_coordinates(self.body.position)
        self.velocity = convert_coordinates(self.body.velocity, option="velocity")

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

        # Set not sensed sensor readings of position to inf
        sensor_distances[not_sensed_idx] = np.inf

        # Set not sensed sensor readings of velocity to zero
        sensor_velocities[not_sensed_idx] = 0.0

        return sensor_distances, sensor_velocities


def main():
    base = WaterworldBase()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        base.display.fill((255, 255, 255))
        base.draw()
        pygame.display.update()
        base.clock.tick(base.FPS)
        base.space.step(1 / base.FPS)

        for p in base.pursuers:
            _p_sensor_vals = []
            for e in base.evaders:
                e_coord = base.convert_coordinates(e.body.position)
                e_vel = base.convert_coordinates(e.body.velocity, option="velocity")

                sensor_distance_e, sensor_velocity_e = p.get_sensor_reading(
                    e_coord, e.radius, e_vel, base.convert_coordinates
                )
                _p_sensor_vals.append(sensor_distance_e)
            p_sensor_vals = np.amin(np.concatenate(_p_sensor_vals, axis=1), axis=1)

    pygame.quit()


if __name__ == "__main__":
    main()


# [x] Making food disappear on colliding with more than n_coop objects at once
# [x] Draw spokes on pursuers
# [ ] Set categories and masks for objects

import random
from pdb import set_trace

import numpy as np
import pygame
import pymunk


class WaterworldBase:
    def __init__(self, n_pursuers=5, n_evaders=5, n_poisons=10, n_obstacles=1):
        pygame.init()
        self.pixel_scale = 30 * 25

        self.display = pygame.display.set_mode((self.pixel_scale, self.pixel_scale))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.FPS = 15  # Frames Per Second

        # Collision handler for collisions between Pursuers and Evaders
        # self.handler_PE = self.space.add_collision_handler(1, 2)
        # Collision handler for collisions between Pursuers and Poisons
        # self.handler_PP = self.space.add_collision_handler(1, 3)
        # Collision handler for collisions between Evaders and Poisons
        # self.handler_EP = self.space.add_collision_handler(2, 3)

        # self.handler_PE.begin = self.return_false
        # self.handler_EP.begin = self.return_false
        # self.handler_PP.begin = self.return_false
        # self.handler_PE.separate = self.Pursuer_Evader_Collision
        # self.handler_PP.separate = self.Pursuer_Poison_Collision

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

        for i, handler in enumerate(self.handlers):
            if i < self.n_pursuers * self.n_evaders:
                handler.begin = self.pursuer_evader_begin_callback
                handler.separate = self.pursuer_evader_separate_callback
            else:
                handler.begin = self.return_false
                handler.separate = self.Pursuer_Poison_Collision

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

    def convert_coordinates(self, point):
        return int(point[0]), self.pixel_scale - int(point[1])

    def add(self):
        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)

    def draw(self):
        for obj_list in [self.pursuers, self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.draw(self.display, self.convert_coordinates)

    def add_handlers(self):
        for pursuer in self.pursuers:
            for obj_list in [self.evaders, self.poisons]:
                for obj in obj_list:
                    self.handlers.append(
                        self.space.add_collision_handler(
                            pursuer.shape.collision_type, obj.shape.collision_type
                        )
                    )
        for poison in self.poisons:
            for evader in self.evaders:
                self.handlers.append(
                    self.space.add_collision_handler(
                        poison.shape.collision_type, evader.shape.collision_type
                    )
                )

    def reset(self):
        pass

    def step(self):
        pass

    def return_false(self, arbiter, space, data):
        return False

    def Pursuer_Evader_Collision(self, arbiter, space, data):
        # print("Ta Daa")
        pass

    def Pursuer_Poison_Collision(self, arbiter, space, data):
        # print("Oh no")
        pass

    def pursuer_evader_begin_callback(self, arbiter, space, data):
        set_trace()
        pursuer_shape, evader_shape = arbiter.shapes

        # Add one collision to evader
        evader_shape.counter += 1

        return False

    def pursuer_evader_separate_callback(self, arbiter, space, data):
        pursuer_shape, evader_shape = arbiter.shapes

        if evader_shape.counter < self.n_coop:
            # Remove one collision to evader
            evader_shape.counter -= 1
        else:
            evader_shape.counter = 0
            print(" === Eaten === ")


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

        self.radius = radius * pixel_scale

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


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
        # Get location of pursuer
        self.center = convert_coordinates(self.body.position)

        # Get distance of object in local frame as a 2x1 numpy array
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # Project distance to sensor vectors
        sensor_distances = self._sensors @ distance_vec

        # Check for valid detection criterions
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self._sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # Set not sensed sensor readings to inf
        sensor_distances[not_sensed_idx] = np.inf

        return sensor_distances


class Evaders(MovingObject):
    def __init__(self, x, y, radius=0.03, collision_type=2):
        super().__init__(x, y, radius=radius)

        self.color = (238, 116, 106)
        self.shape.collision_type = collision_type
        self.shape.counter = 0

        # w, h = 1, 40
        # vertices = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        # t = pymunk.Transform(tx=w / 2, ty=h / 2)
        # self.sensor = pymunk.Poly(self.body, vertices, transform=t)
        # self.sensor.sensor = True

    def add(self, space):
        # space.add(self.body, self.shape, self.sensor)
        space.add(self.body, self.shape)

    # def draw(self, display, convert_coordinates):
    # pygame.draw.circle(
    # display, self.color, convert_coordinates(self.body.position), self.radius
    # )

    # world_vertices = []
    # for v in self.sensor.get_vertices():
    # world_coord = v + self.body.position
    # world_vertices.append(convert_coordinates(world_coord))
    # pygame.draw.polygon(display, self.color, world_vertices)


class Poisons(MovingObject):
    def __init__(self, x, y, radius=0.015 * 3 / 4, collision_type=3):
        super().__init__(x, y, radius=radius)

        self.color = (145, 250, 116)
        self.shape.collision_type = collision_type


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
                p_coord = base.convert_coordinates(p.body.position)

                sensor_distance_e = p.get_sensor_reading(
                    e_coord, e.radius, 0.0, base.convert_coordinates
                )
                _p_sensor_vals.append(sensor_distance_e)
            p_sensor_vals = np.amin(np.concatenate(_p_sensor_vals, axis=1), axis=1)
        # set_trace()

    pygame.quit()


if __name__ == "__main__":
    main()


# [ ] Making food disappear on colliding with more than n_coop objects at once
# [x] Draw spokes on pursuers
# [ ] Set categories and masks for objects

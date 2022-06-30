import random
from pdb import set_trace

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

        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poisons = n_poisons
        self.n_obstacles = n_obstacles

        self.add_obj()
        self.add()

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
                )
            )

        for i in range(self.n_evaders):
            self.evaders.append(
                Evaders(
                    random.randint(0, self.pixel_scale),
                    random.randint(0, self.pixel_scale),
                )
            )

        for i in range(self.n_poisons):
            self.poisons.append(
                Poisons(
                    random.randint(0, self.pixel_scale),
                    random.randint(0, self.pixel_scale),
                )
            )

        for i in range(self.n_obstacles):
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

    def reset(self):
        pass

    def step(self):
        pass


class Obstacle:
    def __init__(self, x, y, category=1, mask=1, pixel_scale=750, radius=0.1):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, pixel_scale * 0.1)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.filter = pymunk.ShapeFilter(categories=category)

        self.radius = radius * pixel_scale
        self.color = (120, 176, 178)

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class MovingObject:
    def __init__(self, x, y, category=1, mask=1, pixel_scale=750, radius=0.015):
        self.body = pymunk.Body()
        self.body.position = x, y
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)

        self.shape = pymunk.Circle(self.body, pixel_scale * radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.filter = pymunk.ShapeFilter()

        self.radius = radius * pixel_scale

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class Pursuers(MovingObject):
    def __init__(self, x, y, category=1, mask=1, radius=0.015):
        super().__init__(x, y, category, mask, radius=radius)

        self.color = (101, 104, 249)


class Evaders(MovingObject):
    def __init__(self, x, y, category=1, mask=1, radius=0.03):
        super().__init__(x, y, category, mask, radius=radius)

        self.color = (238, 116, 106)

        w, h = 1, 40
        vertices = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        t = pymunk.Transform(tx=w / 2, ty=h / 2)
        self.sensor = pymunk.Poly(self.body, vertices, transform=t)
        self.sensor.sensor = True

    def add(self, space):
        space.add(self.body, self.shape, self.sensor)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

        world_vertices = []
        for v in self.sensor.get_vertices():
            world_coord = v + self.body.position
            world_vertices.append(convert_coordinates(world_coord))
        pygame.draw.polygon(display, self.color, world_vertices)


class Poisons(MovingObject):
    def __init__(self, x, y, category=1, mask=1, radius=0.015 * 3 / 4):
        super().__init__(x, y, category, mask, radius=radius)

        self.color = (145, 250, 116)


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
        # set_trace()

    pygame.quit()


if __name__ == "__main__":
    main()

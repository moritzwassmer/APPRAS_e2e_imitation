import numpy as np
from collections import deque
from copy import deepcopy

from PIL import Image

def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    """Scales and crops image to same format as Transfuser trainingdata

    Args:
        image: rgb image
        scale: index of route which is run from leaderboard
        start_x: pixel x from where to start cropping
        crop_x: how much to crop starting from x
        start_y: pixel y from where to start cropping
        crop_y: how much to crop starting from y
    """
    (width, height) = (image.width // scale, image.height // scale)
    if scale != 1:
        image = image.resize((width, height))
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height

    image = np.asarray(image)
    cropped_image = image[start_y:start_y + crop_y, start_x:start_x + crop_x]

    return cropped_image


class RoutePlanner(object):
    """ Defines a class for navigation

    Taken from Learning By Cheating: https://arxiv.org/abs/1912.12294
    Repository: https://github.com/dotchen/LearningByCheating
    """
    def __init__(self, min_distance, max_distance):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False

        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route

    def save(self):
        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False


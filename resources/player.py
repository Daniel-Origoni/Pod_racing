from gymnasium.envs.classic_control.pod_racing.settings import *
import math

class Player():
    def __init__(self, id, x, y, t_x, t_y, target = 1, color = RED):
        self.id = id
        self.color = tuple(color)
        self.x = x
        self.y = y
        self.target = target
        self.angle = math.atan2((t_y - y),(t_x - x))

        self.thrust = 0
        self.x_speed = 0
        self.y_speed = 0
        self.x_acceleration = 0
        self.y_acceleration = 0

        self.checked = 0

    def pos(self):
        return [int(self.x), int(self.y)]
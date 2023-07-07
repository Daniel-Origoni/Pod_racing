import gymnasium as gym
from gymnasium import spaces
from resources import *

import pygame
import numpy as np
import math
import sys

class RaceTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, num_players = 2):

        # The observation space consits of the location of the Agent, the Opponent and the Target
        # Each location is a 2d int array in range (0 to WIDTH, 0 to HEIGHT)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), dtype=int),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([WIDTH, HEIGHT]), dtype=int),
            }
        )

        # The action space is a 3d int array, representing the x value, the y value and the thurst
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([WIDTH, HEIGHT, 101]), dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.num_players = num_players
        
    def step(self, action):
        
        # Get the commands for the player from the action
        x, y, thrust = action
        last_distance = checkDistance(*self.players[0].pos(), *self.checkpoints[self.players[0].target])

        for player in self.players:
            if player.id == 1:
                player.thrust = thrust
                theta = normalizeAngle(checkAngle(*player.pos(), x, y))
            else:
                player.thrust = 100
                theta = normalizeAngle(checkAngle(*player.pos(), *self.checkpoints[player.target]))

            if theta != player.angle:
                player.angle = updateAngle(theta, player.angle)

            if player.thrust == 101:
                player.thrust = 661
                
            x_power = round(math.cos(player.angle) * player.thrust,0)
            y_power = round(math.sin(player.angle) * player.thrust,0)
            player.x_acceleration = math.floor(player.x_speed * 0.85)
            player.y_acceleration = math.floor(player.y_speed * 0.85)
            player.x_speed = player.x_acceleration + x_power
            player.y_speed = player.y_acceleration + y_power

            player.x += (player.x_speed)
            player.y += (player.y_speed)
            
            if checkDistance(*player.pos(), *self.checkpoints[player.target]) < 800:
                player.target = (player.target + 1) % NUM_CHECKPOINTS
                player.checked += 1
                print("player checked = " + str(player.checked))

            if player.checked > 4:
                self.terminated = True

        reward = -0.05

        if last_distance > checkDistance(*self.players[0].pos(), *self.checkpoints[self.players[0].target]):
            reward = (last_distance - checkDistance(*self.players[0].pos(), *self.checkpoints[self.players[0].target]) + 800)/(last_distance)

        '''if self.players[0].checked == 1:
            reward = 1
            self.players[0].checked = 0'''
        
        self.total_reward += reward

        

        if self.render_mode == "human":
            self._render_frame(x, y)

        observation = get_obs(self.players, self.checkpoints)
        info = get_info(self.players)

        return observation, reward, self.terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self, *args):
        x, y = args

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH/10,HEIGHT/10))
            pygame.display.set_caption(TITLE)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WIDTH/10,HEIGHT/10))
        canvas.fill((25, 25, 25))

        for checkpoint in self.checkpoints:
            pygame.draw.circle(canvas, BLUE, (checkpoint[0]/10, checkpoint[1]/10), RADIUS)

        for player in self.players:
            modifiers = getTriangle(player.angle)
            result = tuple((mod_x + player.x/10, mod_y + player.y/10) for mod_x, mod_y in modifiers)
            pygame.draw.polygon(canvas, player.color, result)

        pos = ((self.players[0].x + math.cos(self.players[0].angle) * 1000) / 10, (self.players[0].y + math.sin(self.players[0].angle) * 1000)/10)
        pygame.draw.line(canvas,WHITE, (self.players[0].x /10, self.players[0].y /10), (x/10, y/10))
        pygame.draw.line(canvas,GREEN, (self.players[0].x /10, self.players[0].y /10), pos)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                
                # checking if key "A" was pressed
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
            
    def reset(self, **kwargs):
        self.checkpoints = []
        self.players = []
        self.total_reward = 0
        self.terminated = False

        for i in range(NUM_CHECKPOINTS):
            if i == 0:
                pos = [12000, 1990]
            if i == 1:
                pos = [10680, 4990]
            if i == 2:
                pos = [14020, 3010]
            if i == 3:
                pos = [3990, 7780]
            self.checkpoints.append(pos)

        self.players = [Player(i + 1, *self.checkpoints[0], *self.checkpoints[1], 1, np.random.randint(256, size=3)) for i in range(self.num_players)]
        self.players[0].color = RED

        observation = get_obs(self.players, self.checkpoints)
        info = get_info(self.players)

        if self.render_mode == "human":
            self._render_frame(0,0)

        return observation, info
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

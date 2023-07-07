from gymnasium.envs.classic_control.pod_racing.settings import *
import math
import numpy as np


# functions used by the environment
def checkAngle(a, b, c, d):
    return math.atan2((d - b),(c - a))

def checkDistance(a, b, c, d):
    return math.floor(math.sqrt((a - c)**2 + (b - d)**2))

def getTriangle(theta):
    p1 = (36.22, 0)
    p2 = (-28.38, 22.5)
    p3 = (-28.38, -22.5)

    rp1x = p1[0] * math.cos(theta) - p1[1] * math.sin(theta)
    rp1y = p1[0] * math.sin(theta) + p1[1] * math.cos(theta)                 
    rp2x = p2[0] * math.cos(theta) - p2[1] * math.sin(theta)
    rp2y = p2[0] * math.sin(theta) + p2[1] * math.cos(theta)                        
    rp3x = p3[0] * math.cos(theta) - p3[1] * math.sin(theta)                         
    rp3y = p3[0] * math.sin(theta) + p3[1] * math.cos(theta)
    rp1 = ( rp1x, rp1y )
    rp2 = ( rp2x, rp2y )
    rp3 = ( rp3x, rp3y )

    return (rp1, rp2, rp3)


def normalizeAngle(angle):
    normal = 2 * math.pi
    return (angle + normal) % normal

def updateAngle(theta, angle):
    if angle < math.pi and theta > angle:
        if theta < angle + math.pi:
            if (angle + ROTATION_SPEED) > theta:
                angle = theta
            else:
                angle += ROTATION_SPEED
        else:
            if (angle - ROTATION_SPEED) > 0:
                angle -= ROTATION_SPEED
            elif normalizeAngle(angle - ROTATION_SPEED) < theta:
                angle = theta
            else:
                angle = normalizeAngle(angle - ROTATION_SPEED)

    elif theta < angle:
        if theta > angle - math.pi:
            if (angle - ROTATION_SPEED) < theta:
                angle = theta
            else:
                angle -= ROTATION_SPEED
        else:
            if (angle + ROTATION_SPEED) < (math.pi * 2):
                angle += ROTATION_SPEED
            elif normalizeAngle(angle + ROTATION_SPEED) > theta:
                angle = theta
            else:
                angle = normalizeAngle(angle + ROTATION_SPEED)

    else: 
        if (angle + ROTATION_SPEED) > theta:
                angle = theta
        else:
                angle += ROTATION_SPEED

    return angle

def epsilon_decay(n_epoch, epoch, epsilon):
    min_epsilon = 0.1
        
    decayed_epsilon = epsilon - ((epsilon - min_epsilon) / (n_epoch * 0.8)) * epoch
    
    return decayed_epsilon

def get_obs(players, checkpoints):
    
    agent_location = np.array(players[0].pos(), dtype=int)
    target_location = np.array(checkpoints[players[0].target], dtype=int)
    obs = {"agent": agent_location, "target": target_location}
    return obs

def get_info(players):
    info = {}
    for i in range(1, len(players)):
        info["Opponent_{id}".format(id = str(i))] = np.array(players[i].pos(), dtype=int)
    return info

#!/usr/bin/env python3
import sys
import gym
import numpy as np
#import rbx1

"""Data generation for the case of a single block pick and place in rbx1 Env"""

actions = []
observations = []
infos = []
env = None

def main():
    env = gym.make('rbx1-env:Rbx1GymEnv-v0', renders=True, exposeCoords=True)        # calls reset.  exposeCoords because I want to add the extra metadata like Fetchpick andPlace
    numItr = 100
    initStateSpace = "random"
#    env.reset()                                     # calls reset
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()                           # calls reset
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def get_obj_rel_pos(ee_pos, obj_pos):       # returns halfway point between ee and target
#    interim_goal = (obj_pos + ee_pos) / 2.0
    # TODO:  need to fix for width of object, should really do this in ENV where we have ref to ball itself
    ball_diam = 0.05
    obj_pos[0] += ball_diam if obj_pos[0] > 0 else -1.0 * ball_diam
    obj_pos[1] += ball_diam if obj_pos[1] > 0 else -1.0 * ball_diam
    interim_goal = ((obj_pos * 75) + (ee_pos * 25)) / 100
    print("ee={}, obj={}, interim_goal={}".format(ee_pos, obj_pos, interim_goal))
    return interim_goal


def goToGoal(env, lastObs):

    obj_z_offset = 0.1        # height above object that gripper should initially target

    env._max_episode_steps = 100    # NOTE:  put this here for now instead of on the ENV itself

    goal = lastObs['destination']
    targetPos = lastObs['target']      #[3:6]
    eePos = lastObs['ee_pos']
#    object_rel_pos = lastObs['observation'][6:9]        # NOTE:  This is the difference between the object_pos and the grip_pos!! 
    object_rel_pos = get_obj_rel_pos(eePos, targetPos)       # NOTE:  was just straight subtraction
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy() # object_rel_pos.copy()

    object_oriented_goal[2] += obj_z_offset         # first make the gripper go slightly above the object

    timeStep = 0        #count the total number of timesteps
    episodeObs.append(lastObs)

    print("*** Moving gripper above object (ball)")
    while np.linalg.norm(eePos[:1] - targetPos[:1]) >= 0.02 and timeStep <= env._max_episode_steps:     # basically checking on X and Y distance, disregard Z
        env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy() # object_rel_pos.copy()
        object_oriented_goal[2] += obj_z_offset # was 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i] # *0.1     # was 6

        action[len(action)-1] = 0.75 #open

        print("* ACTION = {}".format(action))

        obsDataNew, reward, done, info = env.step(action)

        print("* OBSERVATION = {}".format(obsDataNew))

        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        targetPos = obsDataNew['target']      #[3:6]
        eePos = obsDataNew['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos)


    print("\n*** Moving gripper down to grasp object (ball)\n")
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]      # *0.6       # was 6

        action[len(action)-1] = -0.05

        print(action)

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        targetPos = lastObs['target']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos)

    print("\n*** Moving gripper to next thing \n")
    while np.linalg.norm(goal - targetPos) >= 0.01 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - targetPos)):
            action[i] = (goal - targetPos)[i]*6

        action[len(action)-1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        targetPos = lastObs['target']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos)


    while True: #limit the number of timesteps in the episode to a fixed duration
        env.render()
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        targetPos = lastObs['target']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos)


        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
import gym
import numpy as np
#import rbx1

"""Data generation for the case of a single block pick and place in rbx1 Env"""

actions = []
observations = []
infos = []

def main():
    env = gym.make('rbx1-env:Rbx1GymEnv-v0', renders=True, exposeCoords=True)        # calls reset.  exposeCoords because I want to add the extra metadata like Fetchpick andPlace
    numItr = 100
    initStateSpace = "random"
#    env.reset()                                     # calls reset
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()                           # calls reset
        print("\n\nOBSERVATION={}".format(obs))

        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file

def goToGoal(env, lastObs):

    env._max_episode_steps = 100    # NOTE:  put this here for now instead of on the ENV itself

    print("LAST OBS = {}".format(lastObs))

    print(type(lastObs))

    goal = lastObs['desired_goal']
    objectPos = lastObs['observation']      #[3:6]
    eePos = lastObs['ee_pos']
#    object_rel_pos = lastObs['observation'][6:9]        # NOTE:  This is the difference between the object_pos and the grip_pos!! 
    object_rel_pos = objectPos - eePos
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()

    print("OBJ ORIENTED GOAL = {}".format(object_oriented_goal))

    object_oriented_goal[2] += 0.03         # first make the gripper go slightly above the object

    timeStep = 0        #count the total number of timesteps
    episodeObs.append(lastObs)

    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            action[i] = object_oriented_goal[i]*6

        action[len(action)-1] = 0.05 #open

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        objectPos = lastObs['observation']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = objectPos - eePos


    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        objectPos = lastObs['observation']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = objectPos - eePos



    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        objectPos = lastObs['observation']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = objectPos - eePos


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
        objectPos = lastObs['observation']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = objectPos - eePos


        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()

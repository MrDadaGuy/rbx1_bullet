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

def get_obj_rel_pos(ee_pos, obj_pos, obj_bias_pct):       # returns a point between ee and target, modified by bias to object position
    obj_bias_pct = np.clip(obj_bias_pct, 0.0, 1.0)     # needs to be between zero and one.  obj_bias_pct is the amount that obj should be preferred in the calculation over EE pos
#    interim_goal = (obj_pos + ee_pos) / 2.0
    # TODO:  need to fix for width of object, should really do this in ENV where we have ref to ball itself
#    ball_radius = 0.03  # radius of urdf sphere_small is 0.03
#    newXpos = obj_pos[0] + ball_radius if obj_pos[0] > 0 else oj_pos[0] -ball_radius
#    newYpos = obj_pos[1] + ball_radius if obj_pos[1] > 0 else obj_pos[1] -ball_radius
    interim_goal = ((obj_pos * obj_bias_pct) + (ee_pos * (1 - obj_bias_pct))) 
    print(">>>>>get_obj_rel_pos: EE={}, obj={}, interim_goal={}, bias={}".format(ee_pos, obj_pos, interim_goal, obj_bias_pct))
    return interim_goal


def goToGoal(env, lastObs):

    obj_z_offset = 0.2        # height above object that gripper should initially target
    grip_open = 0.8

    env._max_episode_steps = 100    # NOTE:  put this here for now instead of on the ENV itself

    destination = lastObs['destination']
    targetPos = lastObs['target']       #[3:6]
    targetPos[2] += obj_z_offset
    eePos = lastObs['ee_pos']
#    object_rel_pos = lastObs['observation'][6:9]        # NOTE:  This is the difference between the object_pos and the grip_pos!! 
    object_rel_pos = get_obj_rel_pos(eePos, targetPos, 0.5)       # NOTE:  was just straight subtraction
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

#    object_oriented_goal = object_rel_pos.copy() # object_rel_pos.copy()
#    object_oriented_goal[2] += obj_z_offset         # first make the gripper go slightly above the object

    timeStep = 0        #count the total number of timesteps
    episodeObs.append(lastObs)

    step1_done = False

    print("\n\n*** STEP 1:  Moving gripper above object (ball)\n\n")
    while step1_done == False:
#    while np.linalg.norm(eePos[:2] - targetPos[:2]) >= 0.005 and np.linalg.norm(eePos[2] - (targetPos[2] + obj_z_offset)) >= 0.005 and timeStep <= env._max_episode_steps:     # basically checking on X and Y distance, disregard Z
#    while np.linalg.norm(object_rel_pos) >= 0.05 and timeStep <= env._max_episode_steps:     # basically checking on X and Y distance, disregard Z
        env.render()
        action = [0, 0, 0, 0]
#        object_oriented_goal = object_rel_pos.copy() # object_rel_pos.copy()
#        object_oriented_goal[2] += obj_z_offset # was 0.03

#        object_rel_pos = get_obj_rel_pos(eePos, targetPos, 0.75) 

        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i] # *0.1     # was 6

        action[len(action)-1] = grip_open    #open

        print("* ACTION = {}".format(action))

        obsDataNew, reward, done, info = env.step(action)

        print("* OBSERVATION = {}".format(obsDataNew))

        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
        eePos = obsDataNew['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos, 0.5 + timeStep * 0.01)       # NOTE: trying to change the obj bias increaes over time

        if timeStep > env._max_episode_steps:
            raise RuntimeError("Too long on step 1")
        if abs(eePos[0] - targetPos[0]) < 0.03:
            if abs(eePos[1] - targetPos[1]) < 0.03:
                if abs(eePos[2] - (targetPos[2])) < 0.03:
                    print("\n\n MOVING TO NEXT STEP!!!!!\n")
                    step1_done = True


    print("\n*** STEP 2:  Moving gripper down to grasp object (ball)\n")
    targetPos = obsDataNew['target']      #[3:6]
    targetPos[2] += 0.175     # offset for radius of ball, and to account for gripper height 
    lower_amt = -0.05        # start with Z position what it was last loop, then decrement lower_amt from there
    grip_amt = grip_open
    object_rel_pos = get_obj_rel_pos(eePos, targetPos, 1)

    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]      # *0.6       # was 6

        grip_amt += -0.05
        action[2] += lower_amt      # lower arm on Z axis
        action[3] += grip_amt       # close grip incrementally

        print("~~~ Move2 action = {}".format(action))

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
#        targetPos = lastObs['target']      #[3:6]
        eePos = lastObs['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, targetPos, 1 + grip_amt)

        if 'ball_collision' in info:
            if info['ball_collision'] == True:
                break



    print("\n*** STEP 2.5:  Moving gripper up a bit\n")
    eePos = lastObs['ee_pos']
    object_rel_pos = get_obj_rel_pos(eePos, targetPos, 0)
    bias = 0.5
#    while np.linalg.norm(goal - destination) >= 0.01 and timeStep <= env._max_episode_steps :
    while True:
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = (object_rel_pos)[i] #*6

#        action[len(action)-1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
#        destinationPos = lastObs['destination']      #[3:6]
#        destination[2] += 0.15      # NOTE:  adding some Z so it's actually above the thingy

        bias += 0.05
        eePos = lastObs['ee_pos']
        targetPos[2:] = eePos[2:]
        targetPos[2] == eePos[2] + 0.05       # NOTE: just move EE upwards a bit
        object_rel_pos = get_obj_rel_pos(eePos, targetPos, bias)


    print("\n*** STEP 3:  Moving gripper to destination\n")
    destination = lastObs['destination']
    destination[2] += 0.15      # NOTE:  adding some Z so it's actually above the thingy
    object_rel_pos = get_obj_rel_pos(eePos, destination, 1)

#    while np.linalg.norm(goal - destination) >= 0.01 and timeStep <= env._max_episode_steps :
    while True:
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = (object_rel_pos)[i] #*6

#        action[len(action)-1] = -0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

#        objectPos = obsDataNew['observation'][3:6]
#        object_rel_pos = obsDataNew['observation'][6:9]
#        destinationPos = lastObs['destination']      #[3:6]
#        destination[2] += 0.15      # NOTE:  adding some Z so it's actually above the thingy

        eePos = lastObs['ee_pos']
        object_rel_pos = get_obj_rel_pos(eePos, destination, 1)

    print("\n*** STEP 4:  Open gripper, drop ball \n")
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
        object_rel_pos = get_obj_rel_pos(eePos, targetPos, .75)


        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import rbx1
import random
import pybullet_data
from pkg_resources import parse_version
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from rbx1_driver.srv import ResetAction,ResetActionResponse
from rbx1_driver.srv import StepAction,StepActionResponse

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class Rbx1GymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False, 
               exposeCoords=False):
    self._exposeCoords = exposeCoords
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._width = 341
    self._height = 256
    self._isDiscrete = isDiscrete
    self.terminated = 0
    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    
#    p.setRealTimeSimulation(True)  # no effect in DIRECT mode

    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rbx1Timings.json")
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([np.finfo(np.float32).max] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(self._height, self._width, 4),
                                        dtype=np.uint8)
    self.viewer = None

  def reset(self):
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    self.plane_id = p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, 0])

#    self.table_id = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)

    self.ball_id = p.loadURDF(os.path.join(self._urdfRoot, "sphere2.urdf"), [.3, .3, .025], globalScaling=0.05)

    xpos = 0.5 + 0.2 * random.random()
    ypos = 0 + 0.25 * random.random()
    ang = 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -9.8)

    # load ROBOT MODEL
    self._rbx1 = rbx1.Rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()   # TODO:  this is odd, it gets set in the called function !?
#    return np.array(self._observation)

    print("***************** NUM CONSTRAINTS ====  {}".format(p.getNumConstraints()))
    for constraintIdx in range(p.getNumConstraints()):
      print("Constraint idx: {} \nInfo:  {} \nState: {} \n".format(constraintIdx, p.getConstraintInfo(constraintIdx), p.getConstraintState(constraintIdx)))

    print("***************** NUM JOINTS ====  {}".format(p.getNumJoints(self._rbx1.rbx1Uid)))
    for jointIdx in range(p.getNumJoints(self._rbx1.rbx1Uid)):
      print("Joint idx: {} \nInfo:  {} \nState: {} \n".format(jointIdx, p.getJointInfo(self._rbx1.rbx1Uid, jointIdx), p.getJointState(self._rbx1.rbx1Uid, jointIdx)))


    return self._observation


  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):

    viewMat = [
        -0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
        -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
        0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
    ]
    projMatrix = [
        0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
        -0.02000020071864128, 0.0
    ]

    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=viewMat,
                               projectionMatrix=projMatrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))

    if self._exposeCoords:
      boxPos, boxOrn = p.getBasePositionAndOrientation(self.blockUid)
      ballPos, ballOrn = p.getBasePositionAndOrientation(self.ball_id)
      eePos = p.getLinkState(self._rbx1.rbx1Uid, self._rbx1.rbx1EndEffectorIndex)

      self._observation = {"rgb_image" : np_img_arr, "desired_goal" :  np.array(boxPos) , "observation" : np.array(ballPos) , "ee_pos" : np.array(eePos[0])}
    else:
      self._observation = np.array(np_img_arr)

    return self._observation

  def step(self, action):

    print("\nSTEPPING:  {}\n".format(action))
    pose = Pose()
    pose.position = Point(action[0], action[1], action[2])
    pose.orientation = Quaternion(0.5, 0.5, -0.5, -0.5)

    return self.step2(pose, action[3])

  def step2(self, pose, gripper_pos):
    for i in range(self._actionRepeat):
#      self._rbx1.applyAction(action)       # NOTE:  don't need to apply action, the ros subscriber will do this
      rospy.wait_for_service('rbx1/step')
      moveit_step = rospy.ServiceProxy("rbx1/step", StepAction)
      moveit_step(pose, gripper_pos)

      p.stepSimulation()
      if self._termination():
        print("termination? uh why")
        break
      #self._observation = self.getExtendedObservation()
      self._envStepCounter += 1

    self._observation = self.getExtendedObservation()
    if self._renders:
      time.sleep(self._timeStep)

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    reward = self._reward()
    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    #print (self._rbx1.endEffectorPos[2])
    state = p.getLinkState(self._rbx1.rbx1Uid, self._rbx1.rbx1EndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self.blockUid, self._rbx1.rbx1Uid, maxDist)   # was self._rbx1.trayUid

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._rbx1.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._rbx1.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._rbx1.rbx1Uid, self._rbx1.rbx1EndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._rbx1.rbx1Uid, 1000, -1,
                                       self._rbx1.rbx1EndEffectorIndex)

    reward = -1000
    numPt = len(closestPoints)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      #print("grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      reward = reward + 1000

    #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import rospy
from sensor_msgs.msg import JointState

class Rbx1:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):

    rospy.init_node('rbx1_bullet_client', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

    self.old_position = None
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.rbx1EndEffectorIndex = 6
    self.rbx1GripperIndex = 7

    self.start_pos = [0,0,0]
    self.start_orientation = p.getQuaternionFromEuler([0,0,0])

    #lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    #upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):
    self.rbx1Uid = p.loadURDF("/home/ubuntu/src/rbx1/rbx1_urdf/urdf/rbx1_urdf.urdf", self.start_pos, self.start_orientation, useFixedBase=True)
    self.num_joints = p.getNumJoints(self.rbx1Uid)

    p.resetBasePositionAndOrientation(self.rbx1Uid, self.start_pos, self.start_orientation)
#    self.jointPositions = [0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    self.jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

    self.numJoints = p.getNumJoints(self.rbx1Uid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.rbx1Uid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.rbx1Uid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

    self.endEffectorPos = [0.537, 0.0, 0.5]
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.rbx1Uid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def joint_state_callback(self, data):
    if data.position == self.old_position:
        return
    self.old_position = data.position

    print("*** NUM JOINTS = {}, LEN DATA POS = {}".format(self.num_joints, len(data.position)))
#    print("CHANGE, moving... {}".format(data.position))
    joint_states = list(data.position) 
    joint_states.insert(0, 0.0)
    joint_states.insert(7, 0.0)

    print("Doing JOINT STATES ... {}".format(joint_states))

    # get the joint states position and move rbx1
    p.setJointMotorControlArray(self.rbx1Uid, list(range(self.num_joints)), p.POSITION_CONTROL, targetPositions=joint_states)
    p.stepSimulation()

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.rbx1Uid, self.rbx1GripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands):
    raise RuntimeError("This should not be called, you naughty person!")
    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]

      state = p.getLinkState(self.rbx1Uid, self.rbx1EndEffectorIndex)
      actualEndEffectorPos = state[0]
      #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.17):
        self.endEffectorPos[1] = -0.17
      if (self.endEffectorPos[1] > 0.22):
        self.endEffectorPos[1] = 0.22

      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz

      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.rbx1Uid, self.rbx1EndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.rbx1Uid,
                                                    self.rbx1EndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.rbx1Uid,
                                                    self.rbx1EndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.rbx1Uid, self.rbx1EndEffectorIndex, pos)

      #print("jointPoses")
      #print(jointPoses)
      #print("self.kukaEndEffectorIndex")
      #print(self.kukaEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.rbx1EndEffectorIndex + 1):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.rbx1Uid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.rbx1Uid, i, jointPoses[i])
      #fingers
      p.setJointMotorControl2(self.rbx1Uid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)
      p.setJointMotorControl2(self.rbx1Uid,
                              8,
                              p.POSITION_CONTROL,
                              targetPosition=-fingerAngle,
                              force=self.fingerAForce)
      p.setJointMotorControl2(self.rbx1Uid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

      p.setJointMotorControl2(self.rbx1Uid,
                              10,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
      p.setJointMotorControl2(self.rbx1Uid,
                              13,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)

    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.rbx1Uid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)

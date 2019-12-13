#!/usr/bin/env python3

import pybullet as bullet
import pybullet_data
import rospy
from sensor_msgs.msg import JointState

from signal import signal, SIGINT
from sys import exit


bullet.connect(bullet.GUI)    #or DIRECT for non-graphical version
bullet.setAdditionalSearchPath(pybullet_data.getDataPath())   #used by loadURDF
bullet.setGravity(0,0,-10)
bullet.setRealTimeSimulation(True)  # no effect in DIRECT mode

rbx1_start_pos = [0,0,1]
rbx1_start_orientation = bullet.getQuaternionFromEuler([0,0,0])

rbx1_id = bullet.loadURDF("src/rbx1/rbx1_urdf/urdf/rbx1_urdf.urdf", rbx1_start_pos, rbx1_start_orientation, useFixedBase=True)

rbx1_num_joints = bullet.getNumJoints(rbx1_id)
rbx1_start_joint_states = [0] * rbx1_num_joints

#print("NUM OF JOINTS = {}".format(rbx1_num_joints))
#for j in range(rbx1_num_joints):
#    print(bullet.getJointInfo(rbx1_id, j))
#exit()



def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.position)

    # get the joint states position and move rbx1
    bullet.setJointMotorControlArray(rbx1_id, list(range(1, rbx1_num_joints -1)), bullet.POSITION_CONTROL, targetPositions=data.position)


def listener():

    rospy.init_node('pybullet_client', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, callback)
    rospy.spin()


    #boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
#    bullet.stepSimulation()
    #cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    #print(cubePos, cubeOrn)

    #p.disconnect()
#    bullet.stepSimulation()

def reset():
    bullet.resetBasePositionAndOrientation(rbx1_id, [0, 0, 0], [0, 0, 0, 1])
    bullet.setJointMotorControlArray(rbx1_id, list(range(1, rbx1_num_joints -1)), bullet.POSITION_CONTROL, targetPositions=[0]* rbx1_num_joints)


if __name__ == '__main__':
    listener()


""" def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    setup()

    print('Running. Press CTRL-C to exit.')
    while True:
        # Do nothing and hog CPU forever until SIGINT received.
        pass """
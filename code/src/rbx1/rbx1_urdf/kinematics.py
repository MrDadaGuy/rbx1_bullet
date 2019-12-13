#!/usr/bin/env python

import time, math
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import ikpy 
from ikpy import geometry_utils
from ikpy.chain import Chain

from scipy.spatial.transform import Rotation as R

class Rbx1_Kinematics(object):

    def __init__(self):
        rospy.init_node('kinematics')
        rospy.loginfo("Kinematic thingy initializin' here...")
        self.chain = Chain.from_urdf_file("urdf/rbx1_urdf.urdf", active_links_mask=[False, True, True, True, True, True, True, True, False, False])
        self.publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.seq = 0

        print(self.chain)


    def move(self, x, y, z, orientation_matrix = np.eye(3)):        

        joint_states = self.chain.inverse_kinematics(geometry_utils.to_transformation_matrix(
            [x, y, z],
            orientation_matrix))

#        joint_states = self.chain.inverse_kinematics([  [-.5, 0, 0, x],
#                                                        [0, 0.5, 0, y],
#                                                        [0, 0, 0.5, z],
#                                                        [0, 0, 0, -.5]    ])



        print(joint_states)

        hdr = Header()
        hdr.seq = self.seq = self.seq + 1
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = "My-Kinematic-Thingy"
        js = JointState()
        js.header = hdr
        js.name = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6", "Joint_Gripper"] #, ,  
#            "Joint_Grip_Servo", "Joint_Tip_Servo", "Joint_Grip_Servo_Arm", 
#            "Joint_Grip_Idle", "Joint_Tip_Idle", "Joint_Grip_Idle_Arm"]
        js.position = joint_states[1:8]
        js.velocity = []
        js.effort = []
        self.publisher.publish(js)


    def rpy_to_orientation_matrix(self, roll, pitch, yaw):

        yawMatrix = np.matrix([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
            ])
        
        pitchMatrix = np.matrix([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
            ])
        
        rollMatrix = np.matrix([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
            ])
        
        R = yawMatrix * pitchMatrix * rollMatrix

        print(R)
        
        theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
        multi = 1 / (2 * math.sin(theta))
        
        rx = multi * (R[2, 1] - R[1, 2]) * theta
        ry = multi * (R[0, 2] - R[2, 0]) * theta
        rz = multi * (R[1, 0] - R[0, 1]) * theta
        
#        print (rx, ry, rz)


if __name__ == "__main__":
    kin = Rbx1_Kinematics()
    x = 0.5
    y = 0
    z = 0.1
    kin.move(x, y, z)                                                     # identity matrix
    time.sleep(2)
    kin.move(x, y, z, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))      # 180 around X
    time.sleep(2)
    kin.move(x, y, z, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))      # 180 around Y
    time.sleep(2)
    kin.move(x, y, z, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))      # 180 around Z



#    kin.move(0,3,0)
#    time.sleep(2)
#    kin.move(-3,0,0)
#    time.sleep(2)
#    kin.move(0,-3,0)
#    time.sleep(2)
#    kin.move(0,0,3)
#    time.sleep(2)

#    kin.rpy_to_orientation_matrix(180, 0, 0)

#    kin.move(1,2,2)
#    time.sleep(1)
#    kin.move(2,12,1)
#    time.sleep(1)
#    kin.move(12,2,0)


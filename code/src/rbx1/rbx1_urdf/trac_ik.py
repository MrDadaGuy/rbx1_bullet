#!/usr/bin/env python

from trac_ik_python.trac_ik import IK

import time, math
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class Rbx1_Kinematics(object):
    def __init__(self):         # gets URDF from /robot_description parameter on param server
        rospy.init_node('trac_ik_py')
        rospy.loginfo("Kinematic thingy initializin' here...")
        self.ik_solver = IK("base_link", "Link_6")
        self.seed_state = [0.0] * self.ik_solver.number_of_joints
        self.publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.seq = 0
        self.z_offset = 0.1575
        self.scale = 1
        self.x = self.y = self.z = self.qx = self.qy = self.qz = self.qw = 0        # keep track of EE pose



    def move(self, x, y, z, qx, qy, qz, qw):

        print("{} {} {} {} {} {} {}".format(x, y, z, qx, qy, qz, qw))

        joint_states = self.ik_solver.get_ik(self.seed_state, x, y, z, qx, qy, qz, qw)

        print(joint_states)

        if joint_states == None:
            raise ValueError("Can't reach target.")

        hdr = Header()
        hdr.seq = self.seq = self.seq + 1
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = "trac-ik-py"
        js = JointState()
        js.header = hdr
        js.name = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"] #, ,  , "Joint_Gripper"
        js.position = joint_states
        js.velocity = []
        js.effort = []
        self.publisher.publish(js)

        # since we got moved the robot, let's keep track of current EE pose
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def callback(self, data):
#        print(data)
        self.move(data.pose.position.x + (self.x * self.scale), 
            data.pose.position.y + (self.y * self.scale), 
            data.pose.position.z + (self.z * self.scale), 
            self.qx, self.qy, self.qz, self.qw)     
            # NOTE:  ignoring orientation because I want EE to stay straight down!
            # data.pose.orientation.x + (self.qx * self.scale), 
            # data.pose.orientation.y + (self.qy * self.scale), 
            # data.pose.orientation.z + (self.qz * self.scale), 
            # data.pose.orientation.w + (self.qw * self.scale))


    def listener(self):
        rospy.Subscriber("/unity/controllers/left", PoseStamped, self.callback)
        rospy.spin()


if __name__ == "__main__":
    kin = Rbx1_Kinematics()

    x = 0.25
    y = 0.0
    z = 0.0 + kin.z_offset
    qx =  0.7071068
    qy = 0
    qz = 0
    qw = -qx

    kin.move(x, y, z, qx, qy, qz, qw)                                                     # identity matrix
    time.sleep(2)
    kin.move(x, y, z, qx, qy, qz, qw)                                                     # identity matrix



#    print("do what i say damit")
#    kin.move(0.2556989193, -0.009014785, 0.1694626522, 0.309258015, -0.176831156, 0.193536893, 0.172092049)
#    print("hooray?")
    kin.listener()




# #    kin.move(0, 0, .68, 0, 0, 0, 1)        
# #    time.sleep(2)


# #    kin.move(0, 0, .67, 1, 0, 0, 0)    
# #    time.sleep(2)



#     kin.move(-x, y, z, qx, qy, qz, qw)                                                     # identity matrix
#     time.sleep(2)

# #    kin.move(-y, x, z, qx, qy, qz, qw)                                                     # identity matrix
# #    time.sleep(2)

# #    kin.move(y, -x, z, qx, qy, qz, qw)                                                     # identity matrix
#     kin.move(y, -x, z, qx, qy, qz, qw)                                                     # identity matrix
#     time.sleep(2)

# #    kin.move(0, 0, .5, 0, 0, 0, 1)                                                     # identity matrix
# #    time.sleep(2)
# #    kin.move(x, y, z, 0, 0, 0, 1)                                                     # identity matrix
# #    time.sleep(2)
# #    kin.move(x, y, z,  0.7071068, 0, 0, -0.7071068)      # 90 around X
# #    time.sleep(2)
# #    kin.move(x, y, z, 0, 0.7071068, 0, 0.7071068)      # 90 around Y
# #    time.sleep(2)
# #    kin.move(x, y, z,  0, 0, 0.7071068, 0.7071068)      # 90 around Z

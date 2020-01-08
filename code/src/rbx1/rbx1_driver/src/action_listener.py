#!/usr/bin/env python2

import ik_tool
import rospy
from geometry_msgs.msg import Pose
from rbx1_driver.srv import ResetAction,ResetActionResponse
from rbx1_driver.srv import StepAction,StepActionResponse

def take_reset_action(req):
    print("Taking Reset Action")
    return ResetActionResponse(ikTool.reset())

def take_step_action(req):
    x = req.pose.position.x
    y = req.pose.position.y
    z = req.pose.position.z
    ox = req.pose.orientation.x
    oy = req.pose.orientation.y
    oz = req.pose.orientation.z
    ow = req.pose.orientation.w
    print("Taking Step Action:", x, y, z, ox, oy, oz, ow, req.gripper_pct)
    return StepActionResponse(ikTool.step(x, y, z, ox, oy, oz, ow, req.gripper_pct))

if __name__ == "__main__":
    rospy.init_node('action_listener')

    # sim = GHER.gmgym.ros_unity_sim.Sim()
    ikTool = ik_tool.IK_Tool()
    reset_service = rospy.Service('rbx1/reset', ResetAction, take_reset_action)
    step_service  = rospy.Service('rbx1/step', StepAction, take_step_action)

    print("Ready to take actions")
    rospy.spin()

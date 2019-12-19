#!/usr/bin/env python2

import ik_tool

from rbx1_driver.srv import ResetAction,ResetActionResponse
from rbx1_driver.srv import StepAction,StepActionResponse
import rospy

def take_reset_action(req):
    print("Taking Reset Action")
    return ResetActionResponse(ikTool.reset())

def take_step_action(req):
    print("Taking Step Action:",req.x, req.y, req.z, req.gripper_pct)
    return StepActionResponse(ikTool.step(req.x, req.y, req.z, req.gripper_pct))

if __name__ == "__main__":
    rospy.init_node('action_listener')

    # sim = GHER.gmgym.ros_unity_sim.Sim()
    ikTool = ik_tool.IK_Tool()
    reset_service = rospy.Service('rbx1/take_reset_action', ResetAction, take_reset_action)
    step_service  = rospy.Service('rbx1/take_step_action', StepAction, take_step_action)

    print("Ready to take actions")
    rospy.spin()

#! /usr/bin/env python

import rospy

import actionlib
import actionlib.simple_action_server
#from control_msgs.msg import *
import trajectory_msgs.msg._JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import control_msgs.msg._GripperCommandActionFeedback
import control_msgs.msg._GripperCommandActionResult
import control_msgs.msg._GripperCommandAction

class RobotAction(object):
    # create messages that are used to publish feedback/result
    _feedback = control_msgs.msg.GripperCommandActionFeedback()
    _result = control_msgs.msg.GripperCommandActionResult()

    js_pub = rospy.Publisher('/joint_states_grip_cmd', JointState, queue_size=10)

    def __init__(self, name):
        rospy.loginfo("Grip Controller initializin' here...")
        self._action_name = name
        self._as = actionlib.ActionServer(self._action_name, control_msgs.msg.GripperCommandAction, self.on_goal, self.on_cancel, auto_start = False)    # execute_cb=self.execute_cb,
        self._as.start()

    def on_cancel(self, goal):
        rospy.loginfo("Called 'on_cancel'.  Boo.  Not cool.")
      
    def on_goal(self, goal):
        r = rospy.Rate(50)
        
        goal.set_accepted()

        # publish info to the console for the user
        rospy.loginfo('%s: Executing callback - goal:  ' % self._action_name)

        goal = goal.get_goal()
        cmd = goal.command
        desired_position = cmd.position

        hdr = Header()
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = "MoveIt! Grip Ctl"
        js = JointState()
        js.header = hdr
        js.name = ["Joint_Grip_Servo", ] #, "Joint_Tip_Servo", "Joint_Grip_Servo_Arm", "Joint_Grip_Idle", "Joint_Tip_Idle", "Joint_Grip_Idle_Arm"]
        js.position = [desired_position, ]

        print(js)

        self.js_pub.publish(js)

#        goal.publish_feedback(self._feedback)                   # publish the feedback
        r.sleep()               # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes

        status = self._as.status
        print(status)

#        status = goal.get_goal_status().status

#        if status == actionlib.GoalStatus.ACTIVE:
#            rospy.loginfo('%s: Succeeded' % self._action_name)
#            goal.set_succeeded(self._result)
#        elif status == actionlib.GoalStatus.PREEMPTED:
#            rospy.loginfo('%s: Preempted' % self._action_name)
#            goal.set_preempted(self._result)
#        else:
#            rospy.loginfo('%s: Cancelled' % self._action_name)
#            goal.set_canceled(self._result)


if __name__ == '__main__':
    rospy.init_node('rbx1_grip_controller')
    server = RobotAction(rospy.get_name())
    rospy.spin()
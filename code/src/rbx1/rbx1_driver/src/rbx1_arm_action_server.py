#! /usr/bin/env python

import rospy

import actionlib
import actionlib.simple_action_server
#from control_msgs.msg import *
import trajectory_msgs.msg._JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

import control_msgs.msg._FollowJointTrajectoryActionFeedback
import control_msgs.msg._FollowJointTrajectoryActionResult
import control_msgs.msg._FollowJointTrajectoryAction

class RobotAction(object):
    # create messages that are used to publish feedback/result
    _feedback = control_msgs.msg.FollowJointTrajectoryFeedback()
    _result = control_msgs.msg.FollowJointTrajectoryResult()

    js_pub = rospy.Publisher('/joint_states_arm_cmd', JointState, queue_size=10)

    def __init__(self, name):
        rospy.loginfo("Arm Controller initializin' here...")
        self._action_name = name
        self._as = actionlib.ActionServer(self._action_name, control_msgs.msg.FollowJointTrajectoryAction, self.on_goal, self.on_cancel, auto_start = False)    # execute_cb=self.execute_cb,
        self._as.start()

    def on_cancel(self, goal):
        rospy.loginfo("Called 'on_cancel'.  Boo.  Not cool.")
      
    def on_goal(self, goal):
        r = rospy.Rate(50)
        counter = 0
        
        goal.set_accepted()

        # publish info to the console for the user
        rospy.loginfo('%s: Executing callback - goal:  ' % self._action_name)

        points = goal.get_goal().trajectory.points

        for i in range(1, len(points)):         # this is the number of motion steps provided by MoveIt!
            hdr = Header()
            hdr.stamp = rospy.Time.now()
            hdr.frame_id = "MoveIt! Arm Ctl"
            js = JointState()
            js.header = hdr
            js.name = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]
            js.position = points[i].positions
            js.velocity = points[i].velocities
            js.effort = points[i].accelerations
#            rospy.loginfo(str(js))
            self.js_pub.publish(js)

            counter += 1

            goal.publish_feedback(self._feedback)                   # publish the feedback

        r.sleep()               # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes

        status = goal.get_goal_status().status

        if status == actionlib.GoalStatus.ACTIVE:
            rospy.loginfo('%s: Succeeded' % self._action_name)
            goal.set_succeeded(self._result)
        elif status == actionlib.GoalStatus.PREEMPTED:
            rospy.loginfo('%s: Preempted' % self._action_name)
            goal.set_preempted(self._result)
        else:
            rospy.loginfo('%s: Cancelled' % self._action_name)
            goal.set_canceled(self._result)


if __name__ == '__main__':
    rospy.init_node('rbx1_arm_controller')
    server = RobotAction(rospy.get_name())
    rospy.spin()
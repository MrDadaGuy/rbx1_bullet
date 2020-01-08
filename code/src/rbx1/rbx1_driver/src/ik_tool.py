#!/usr/bin/env python2


import sys, math, time, random
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import RobotState

import actionlib_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from std_msgs.msg import Header, String, Empty
from sensor_msgs.msg import JointState

#import generate_samples

ARM_FAR_REACH = 0.44412
ARM_CLOSE_REACH = 0.21

moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_arm = moveit_commander.MoveGroupCommander("arm")
#group_gripper = moveit_commander.MoveGroupCommander("gripper")  # dont think we need this - we'll handle open/close without planner

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)

class IK_Tool:
    def __init__(self):

        # rospy.init_node("moveit_ik_node") ### TC ### this class does not need a node of it's own since it is running in a service
        self.goal_pub = rospy.Publisher("move_group/goal", moveit_msgs.msg.MoveGroupActionGoal, queue_size=1)
        self.goal_update_pub = rospy.Publisher("/rviz/moveit/update_goal_state", Empty, queue_size=1)
        self.ros_time = rospy.Time()
        self.grip_joints = ['Joint_Grip_Idle', 'Joint_Tip_Idle', 'Joint_Grip_Idle_Arm', 'Joint_Grip_Servo', 'Joint_Tip_Servo', 'Joint_Grip_Servo_Arm']    # 'Joint_Grippper', 

        self.grip_goal = moveit_msgs.msg.MoveGroupActionGoal()
        self.grip_goal.goal.request.workspace_parameters.header.frame_id = "base_footprint"
        self.grip_goal.goal.request.workspace_parameters.min_corner.x = -1.0
        self.grip_goal.goal.request.workspace_parameters.min_corner.y = -1.0
        self.grip_goal.goal.request.workspace_parameters.min_corner.z = -1.0
        self.grip_goal.goal.request.workspace_parameters.max_corner.x = 1.0
        self.grip_goal.goal.request.workspace_parameters.max_corner.y = 1.0
        self.grip_goal.goal.request.workspace_parameters.max_corner.z = 1.0
        self.grip_goal.goal.request.group_name = "gripper"
        self.grip_goal.goal.request.num_planning_attempts = 10
        self.grip_goal.goal.request.allowed_planning_time = 0.5  # was 5.0
        self.grip_goal.goal.request.max_velocity_scaling_factor = 1.0
        self.grip_goal.goal.request.max_acceleration_scaling_factor = 1.0
        self.grip_goal.goal.request.start_state.is_diff = True
        self.grip_goal.goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self.grip_goal.goal.planning_options.planning_scene_diff.is_diff = True
        self.grip_goal.goal.planning_options.replan_delay = 0.5 # was 2.0

    def _move_arm(self, x, y, z=0.2, ox=0.5, oy=0.5, oz=-0.5, ow=-0.5):
        assert self._is_inside_circle(x, y, ARM_FAR_REACH), "Target is outside of RBX1 arm reach of {} m".format(ARM_FAR_REACH)
        assert not self._is_inside_circle(x, y, ARM_CLOSE_REACH), "Target is too close to RBX1 base, needs to be greater than {} m".format(ARM_CLOSE_REACH)

        group_arm.set_start_state_to_current_state()

        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = ow
        pose_target.orientation.x = ox
        pose_target.orientation.y = oy
        pose_target.orientation.z = oz
        pose_target.position.x = x
        pose_target.position.y = y 
        pose_target.position.z = z
        group_arm.set_pose_target(pose_target)

        success = group_arm.go(wait=True)
        
        group_arm.stop()
        group_arm.clear_pose_targets()

        return success

    def _move_gripper(self, pct_open=1.0):
        assert pct_open >= 0.0 and pct_open <= 1.0

        joint_pos = (pct_open if pct_open < 1.0 else 0.999) * (math.pi / 2)

        joint_positions = [joint_pos, joint_pos*-1, joint_pos, joint_pos*-1, joint_pos, joint_pos*-1]

        self.grip_goal.goal.request.workspace_parameters.header.stamp = self.ros_time.now()

        # remove all items from goal constraint list
        del self.grip_goal.goal.request.goal_constraints[:]

        constraints = moveit_msgs.msg.Constraints()

        for i in range(len(self.grip_joints)):
            jc = moveit_msgs.msg.JointConstraint() 
            jc.joint_name = self.grip_joints[i]
            jc.position = joint_positions[i]
            jc.tolerance_above = 0.0001
            jc.tolerance_below = 0.0001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        self.grip_goal.goal.request.goal_constraints.append(constraints)

        self.grip_goal.header.stamp = self.ros_time.now() # = header
        self.grip_goal.goal_id.stamp = self.ros_time.now() #  = goal_id
        self.grip_goal.goal_id.id = "ik_tools-{}".format(self.ros_time.now())

        while self.goal_pub.get_num_connections == 0:
            rospy.sleep(0.1)

        try:
            self.goal_pub.publish(self.grip_goal)
        except Exception as err:
            print(err)

    def _is_inside_circle(self, x, y, rad, circle_x = 0, circle_y = 0):
        if ((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad):
            return True
        return False

    def reset(self):
        print("Taking Reset Action")
        group_arm.set_start_state_to_current_state()

        # Reset robot to original pose and regenerate new world, new random position of world objects in unity
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.grip_joints
        joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # moveit_robot_state = RobotState()
        # moveit_robot_state.joint_state = joint_state
        # group_arm.set_start_state(moveit_robot_state)
        success = group_arm.go(joint_state)
        group_arm.stop()
        
        self._move_gripper(0.0)
        self.goal_update_pub.publish()

        return success

    def step(self, x, y, z, ox=0.5, oy=0.5, oz=-0.5, ow=-0.5, gripper_pct_open=1.0):
        print("Taking Step Action")
        success = self._move_arm(x, y, z, ox, oy, oz, ow)
        
        self._move_gripper(gripper_pct_open)
        self.goal_update_pub.publish()

        return success

#!/usr/bin/env python
 
import rospy
from nav_msgs.msg import Odometry
from obstacles import publish_obstacle_msg_moving

t=0
 
def callback(msg):
    print(msg.pose.pose)
    if(t%1000000 == 0):
        print('calling publisher for updated position')
        publish_obstacle_msg_moving(msg.pose.pose)
        r.sleep()

rospy.init_node('check_odometry')
odom_sub = rospy.Subscriber('/husky_velocity_controller/odom', Odometry, callback)
rospy.spin()
#!/usr/bin/env python
import rospy, math
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32, Point
import time

def publish_obstacle_msg(points):
	rospy.init_node("test_obstacle_msg")
	pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)
	obstacle_msg = ObstacleArrayMsg()
	obstacle_msg.header.stamp = rospy.Time.now()
	obstacle_msg.header.frame_id = "odom" # CHANGE HERE: odom/map
	lines = [[1, 2, 3], [4, 5, 6]]
	scale_x = 190
	scale_y = 260
	shift_x = 4.5
	shift_y = 1
	xcoords = points[0]/scale_x - shift_x
	ycoords = points[1]/scale_y - shift_y

	# xcoords = lines[0]
	# ycoords = lines[1]

	# obstacle_msg.obstacles[0].id = 0
	j=0
	for idx, xcoord in enumerate(xcoords):
		if(idx%50 == 0):
			obstacle_msg.obstacles.append(ObstacleMsg())
			# print('point: x= ', xcoord, ' y= ', ycoords[idx], ' idx: ', idx)
			print(type(xcoord))
			# Add point obstacle
			# v1 = Point32()
			# v1.x = xcoord
			# v1.y = ycoords[idx]
			# obstacle_msg.obstacles[idx].id = idx

			x = xcoord
			y = ycoords[idx]

			# rotate coordinates


			obstacle_msg.obstacles[j].polygon.points = [Point()]
			obstacle_msg.obstacles[j].polygon.points[0].x = y
			obstacle_msg.obstacles[j].polygon.points[0].y = -x
			obstacle_msg.obstacles[j].polygon.points[0].z = 0
			j = j+1


		# obstacle_msg.obstacles.append(ObstacleMsg())
		# obstacle_msg.obstacles[1].id = 2
		# v1 = Point32()
		# v1.x = -1
		# v1.y = -1
		# v2 = Point32()
		# v2.x = -0.5
		# v2.y = -1.5
		# v3 = Point32()
		# v3.x = 0
		# v3.y = -1
		# obstacle_msg.obstacles[2].polygon.points = [v1, v2, v3]

	r = rospy.Rate(10) # 10hz
	t = 0.0
	timeout = time.time() + 5 # 10 seconds from now
	while True and not rospy.is_shutdown():
		test = 0
		if test == 5 or time.time() > timeout:
			print('timer about to break')
			break
		pub.publish(obstacle_msg)
		r.sleep()
		test - test - 1
	# while not rospy.is_shutdown():
	# 	pub.publish(obstacle_msg)
	# 	r.sleep()



def publish_obstacle_msg_moving(pose):
	points = unit_test2()

	position = pose.position
	print('postiions: ', position)

	rospy.init_node("test_obstacle_msg")
	pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)
	obstacle_msg = ObstacleArrayMsg()
	obstacle_msg.header.stamp = rospy.Time.now()
	obstacle_msg.header.frame_id = "odom" # CHANGE HERE: odom/map
	lines = [[1, 2, 3], [4, 5, 6]]
	scale_x = 250
	scale_y = 190
	shift_x = 4.5
	shift_y = 1
	xcoords = points[0]/scale_x
	ycoords = points[1]/scale_y

	j=0
	for idx, xcoord in enumerate(xcoords):
		if(idx%50 == 0):
			obstacle_msg.obstacles.append(ObstacleMsg())
			print(type(xcoord))

			x = xcoord       - position.x
			y = ycoords[idx] - position.y

			# rotate coordinates
			obstacle_msg.obstacles[j].polygon.points = [Point()]
			obstacle_msg.obstacles[j].polygon.points[0].x = y
			obstacle_msg.obstacles[j].polygon.points[0].y = -x
			obstacle_msg.obstacles[j].polygon.points[0].z = 0
			j = j+1

	r = rospy.Rate(10) # 10hz
	t = 0.0
	timeout = time.time() + 10 # 2 seconds from now
	while True and not rospy.is_shutdown():
		test = 0
		if test == 5 or time.time() > timeout:
			print('timer about to break')
			break
		pub.publish(obstacle_msg)
		r.sleep()
		test - test - 1
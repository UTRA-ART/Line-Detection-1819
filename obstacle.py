#!/usr/bin/env python
import rospy, math
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32

def publish_obstacle_msg(points):
  rospy.init_node("test_obstacle_msg")

  pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)

  obstacle_msg = ObstacleArrayMsg()
  obstacle_msg.header.stamp = rospy.Time.now()
  obstacle_msg.header.frame_id = "odom" # CHANGE HERE: odom/map

  lines = [[1, 2, 3], [4, 5, 6]]
  xcoords = lines[0]
  ycoords = lines[1]

  for idx, xcoord in enumerate(xcoords):
    print('point: x= ', xcoord, ' y= ', ycoords[idx])
    # Add point obstacle
    obstacle_msg.obstacles.append(ObstacleMsg())
    obstacle_msg.obstacles[idx].id = idx
    obstacle_msg.obstacles[idx].polygon.points = [Point32()]
    obstacle_msg.obstacles[idx].polygon.points[0].x = xcoord
    obstacle_msg.obstacles[idx].polygon.points[0].y = ycoords[idx]
    obstacle_msg.obstacles[idx].polygon.points[0].z = 0

  r = rospy.Rate(10) # 10hz
  t = 0.0
  while not rospy.is_shutdown():
    pub.publish(obstacle_msg)
    r.sleep()

if __name__ == '__main__':
  try:
    publish_obstacle_msg()
  except rospy.ROSInterruptException:
    pass

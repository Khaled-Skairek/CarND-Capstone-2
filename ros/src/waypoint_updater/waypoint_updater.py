#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial import KDTree
from styx_msgs.msg import Lane, Waypoint
from nav_msgs.msg import Path
from std_msgs.msg import Int32
import time
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# Implentation parameters
LOOKAHEAD_WPS = 200    # Number of waypoints we will publish
PUBLISH_RATE = 10      # Publishing rate (Hz) Default: 50

max_wp_distance = 20.0
debugging = False


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_wp_sub = rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.base_waypoints_viz_pub = rospy.Publisher('base_waypoints_viz', Path, queue_size=2)

        # State variables
        self.base_waypoints = []
        self.base_vels = []
        self.next_waypoint = None
        self.pose = None
        self.red_light_waypoint = None
        self.msg_seq = 0
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.accel = max(rospy.get_param('/dbw_node/decel_limit') * 0.5, -1.0)
        # Distance (m) where car will stop before red light
        self.stop_distance = rospy.get_param('~stop_distance', 5.0)

        self.loop()

    def loop(self):
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            final_waypoints = self.get_final_waypoints()
            if final_waypoints:
                lane = self.generate_lane(final_waypoints)
                self.final_waypoints_pub.publish(lane)
                self.msg_seq += 1
            rate.sleep()

    def get_closest_waypoint_idx(self, x, y):
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_next_waypoint(self, x, y):
        closest_idx = self.get_closest_waypoint_idx(x, y)
        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_final_waypoints(self):
        """Get final waypoints"""
        if self.pose:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            # Find next_waypoint based on ego position & orientation
            next_waypoint = self.get_next_waypoint(x, y)

            # Generate the list of next LOOKAHEAD_WPS waypoints
            num_base_wp = len(self.base_waypoints)
            last_base_wp = num_base_wp - 1
            waypoint_idx = [idx % num_base_wp
                            for idx in range(next_waypoint, next_waypoint + LOOKAHEAD_WPS)]
            final_waypoints = [self.base_waypoints[wp] for wp in waypoint_idx]

            # Start from original velocities
            for idx in waypoint_idx:
                self.set_waypoint_velocity(self.base_waypoints, idx, self.base_vels[idx])
            try:
                red_idx = waypoint_idx.index(self.red_light_waypoint)
                self.decelerate(final_waypoints, red_idx, self.stop_distance)
            except ValueError:
                # No red light available: self.red_light_waypoint is None or not in final_waypoints
                red_idx = None

            # If we are close to the end of the circuit, make sure that we stop there
            if self.base_vels[-1] < 1e-5:
                try:
                    last_wp_idx = waypoint_idx.index(last_base_wp)
                    self.decelerate(final_waypoints, last_wp_idx, 0)
                except ValueError:
                    pass
            total_vel = 0
            for fwp in final_waypoints:
                total_vel+=fwp.twist.twist.linear.x

            return final_waypoints

    def generate_lane(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        return lane

    def decelerate(self, waypoints, stop_index, stop_distance):
        """Decelerate a list of wayponts so that they stop on stop_index"""
        if stop_index <= 0:
            return
        dist = self.distance(waypoints, 0, stop_index)
        step = dist / stop_index

        v = 0.
        d = 0.
        for idx in reversed(range(len(waypoints))):
            if idx < stop_index:
                d += step
                if d > self.stop_distance:
                    v = math.sqrt(2*abs(self.accel)*(d-stop_distance))
                    if v < 0.01:
                        v = 0
            if v < self.get_waypoint_velocity(waypoints, idx):
                self.set_waypoint_velocity(waypoints, idx, v)

    def get_waypoint_velocity(self, waypoints, waypoint):
        return waypoints[waypoint].twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2) + pow((a.z - b.z), 2))
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # Callback functions
    def pose_cb(self, msg):
        self.pose = msg

    def base_waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in msg.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

        self.base_vels = [self.get_waypoint_velocity(msg.waypoints, i) for i in range(len(msg.waypoints))]
        self.base_wp_sub.unregister()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def traffic_cb(self, msg):
        self.red_light_waypoint = msg.data if msg.data >= 0 else None


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
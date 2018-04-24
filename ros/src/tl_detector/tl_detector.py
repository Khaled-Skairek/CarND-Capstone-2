#!/usr/bin/env python

import yaml
import numpy as np
from scipy.spatial import KDTree, distance

import rospy
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Int32
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from light_classification.tl_classifier import TLClassifier
import tf as rostf

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.waypoints_2d = None
        self.waypoint_tree = None
        self.lights_tree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = rostf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.lights_tree:
            lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y]
                         for light in msg.lights]
            self.lights_tree = KDTree(lights_2d)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x: x coordinate of a waypoint
            y: y coordinate of a waypoint

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        # stop_line_positions = self.config['stop_line_positions']
        # if self.pose:
        #     x = self.pose.pose.position.x
        #     y = self.pose.pose.position.y
        #     car_wp_idx = self.get_closest_waypoint(x, y)
        #
        #     diff = len(self.waypoints.waypoints)
        #     for i, light in enumerate(self.lights):
        #         # Get stop line waypoint index
        #         line = stop_line_positions[i]
        #         temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
        #         # Find the closest stop line waypoint index
        #         d = temp_wp_idx - car_wp_idx
        #         if 0 <= d < diff:
        #             diff = d
        #             closest_light = light
        #             line_wp_idx = temp_wp_idx
        closest_light = self.get_closest_light()

        if closest_light:
            state = self.get_light_state(closest_light)
            rospy.loginfo("Closest light state: {}".format(state))
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

    def get_closest_light(self):
        """Get the closest traffic light in front of the car.

        :return: The closest traffic light in front of the car
        """
        if self.pose:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y

            # Check if closest is ahead or behind vehicle
            closest_idx = self.lights_tree.query([x, y], 1)[1]
            closest_light = self.lights[closest_idx]
            prev_light = self.lights[closest_idx - 1]

            # Equation for hyperplane through closest_coords
            closest_vec = np.array([closest_light.pose.pose.position.x,
                                    closest_light.pose.pose.position.y])
            prev_vec = np.array([prev_light.pose.pose.position.x,
                                 prev_light.pose.pose.position.y])
            pos_vect = np.array([x, y])
            dot_product = np.dot(closest_vec - prev_vec, pos_vect - closest_vec)
            if dot_product > 0:
                closest_idx = (closest_idx + 1) % len(self.lights)

            closest_light = self.lights[closest_idx]
            # rospy.loginfo("pose.x={},pose.y={}".format(x, y))
            # rospy.loginfo("light.x={},light.y={}".format(closest_light.pose.pose.position.x,
            #                                              closest_light.pose.pose.position.y))
            return closest_light


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

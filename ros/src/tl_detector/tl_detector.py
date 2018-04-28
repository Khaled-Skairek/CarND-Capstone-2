#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
from scipy.spatial import distance, KDTree
import threading

DEBUG = True              # get printout

USE_SIMULATOR_STATE = False # For testing: use simulator provided topic /vehicle/traffic_lights
STATE_COUNT_THRESHOLD = 3

NO_LIGHT = -10000000

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        self.bridge = CvBridge()

        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.is_classifier_available = True
        self.async_light_state = TrafficLight.UNKNOWN

        self.lights_2d = None
        self.lights_tree = None

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

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
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        # store base_waypoints
        self.waypoints = msg
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in msg.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # store lights from styx_msgs as well
        self.lights = msg.lights
        if not self.lights_2d:
            self.lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y]
                              for light in msg.lights]
            self.lights_tree = KDTree(self.lights_2d)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
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
            if light_wp != NO_LIGHT:
                if state != TrafficLight.RED:
                    light_wp = -light_wp  # >0: RED; <0: NOT RED!

            self.last_wp = light_wp
            rospy.logdebug("tl_detector: publishing new light_wp = %s",str(light_wp))
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            rospy.logdebug("tl_detector: publishing last light_wp = %s",str(self.last_wp))
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        
        # only increase count if we are looking at a light
        if light_wp != NO_LIGHT:
            self.state_count += 1

    def get_closest_waypoint_idx(self, x, y):
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        return closest_idx

    def get_next_waypoint_idx(self, x, y):
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

    def get_closest_light_idx(self, x, y):
        closest_idx = self.lights_tree.query([x, y], 1)[1]
        return closest_idx

    def get_next_light_idx(self, x, y):
        closest_idx = self.get_closest_light_idx(x, y)
        # Check if closest is ahead or behind vehicle
        closest_coord = self.lights_2d[closest_idx]
        prev_coord = self.lights_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.lights_2d)
        return closest_idx

    def get_light_state_async(self):
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        self.async_light_state = self.light_classifier.get_classification(cv_image)

        self.is_classifier_available = True
    
    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.is_classifier_available:
            self.is_classifier_available = False
            thread = threading.Thread(target=self.get_light_state_async, args=[])
            thread.start()
        return self.async_light_state
            
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        if self.pose:
            #TODO find the closest visible traffic light (if one exists)

            # Get the closest light
            car_x = self.pose.pose.position.x
            car_y = self.pose.pose.position.y
            closest_light_idx = self.get_next_light_idx(car_x, car_y)
            if closest_light_idx is not None:
                closest_light = self.lights[closest_light_idx]

                # get waypoint of stopping line in front of light
                stop_line_position = stop_line_positions[closest_light_idx]
                stop_line_position_x = stop_line_position[0]
                stop_line_position_y = stop_line_position[1]
                stopline_wp = self.get_closest_waypoint_idx(stop_line_position_x, stop_line_position_y)

                if USE_SIMULATOR_STATE:
                    state = self.lights_state[closest_light_idx]
                    return stopline_wp, state
                else:
                    state = self.get_light_state(closest_light)
                    return stopline_wp, state
        
        self.waypoints = None
        return NO_LIGHT, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
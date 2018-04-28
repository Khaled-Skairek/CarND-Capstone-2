from pid import PID
from yaw_controller import YawController
import math
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MAX_BRAKE = 400


class Controller(object):
    def __init__(self, *args, **kwargs):

        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        min_speed = 0

        self.velocity_pid = PID(kp=0.8, ki=0, kd=0.05, mn=self.decel_limit, mx=0.5 * self.accel_limit)
        self.steering_pid = PID(kp=0.16, ki=0.001, kd=0.1, mn=-self.max_steer_angle, mx=self.max_steer_angle)
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, min_speed, self.max_lat_accel, self.max_steer_angle)
        
    def reset(self):
        self.velocity_pid.reset()
        self.steering_pid.reset()

    def control(self, target_linear_vel, target_angular_vel, current_linear_vel,
                cross_track_error, sample_time):
        # Difference between target velocity and current velocity.
        vel_error = target_linear_vel - current_linear_vel
        throttle = self.velocity_pid.step(vel_error, sample_time)

        # N*m - to hold the car in place if we are stopped at a light. Acceleration ~ 1m/s^2
        brake = 0.0
        # Stop
        if target_linear_vel == 0.0 and current_linear_vel < 0.1:
            throttle = 0.0
            brake = MAX_BRAKE
        # Decelerate
        elif throttle < 0:
            deceleration = abs(throttle)
            if deceleration > self.brake_deadband:
                total_weight = self.vehicle_mass + (self.fuel_capacity * GAS_DENSITY)
                brake = total_weight * self.wheel_radius * deceleration
            throttle = 0

        # without Steering PID car will wiggle around the waypoint path
        predictive_steering = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_linear_vel)
        corrective_steering = self.steering_pid.step(cross_track_error, sample_time)
        steering = predictive_steering + corrective_steering

        return throttle, brake, steering

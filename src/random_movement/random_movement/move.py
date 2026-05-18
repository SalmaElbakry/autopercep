import asyncio
import math
import re
import os
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import qtm
import rclpy
import json

# ================================= define parameters =================================

with open("parameters.json", 'r') as f:
    parameters = json.load(f)

class Options():
    def __init__(self):
        self.corner1 = (float(parameters["corner1"][0]), float(parameters["corner1"][1]))  # limit moving area x, y
        self.corner2 = (float(parameters["corner2"][0]), float(parameters["corner2"][1]))
        self.rot_index = int(parameters["rot_index"])  # message index of rotation angle
        self.update_time = float(parameters["update_time"])  # time for updating current position in seconds
        self.minimal_dist = float(parameters["minimal_dist"])  # distance defining two car too close
        self.away_scale = float(parameters["away_scale"])  # amount of going away when two cars are too close
        self.angular_speed = float(parameters["angular_speed"])
        self.rotate_scale = float(parameters['rotate_scale'])

opt = Options()

# ================================= picar control ================================= 
import RPi.GPIO as GPIO
import Adafruit_PCA9685

from geometry_msgs.msg import Twist
#from perlin_noise import PerlinNoise
from rclpy.node import Node
from turtlesim.msg import Pose

from random_movement.reverse_speed import SpeedSolver

self_id = os.environ['SELF_ID']
qtm_ip = os.environ['QTM_IP']

pos_message = {}
# define motor control
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


IN1 = 23
IN2 = 24
IN3 = 27
IN4 = 22

ENA = 0
ENB = 1

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

def changespeed_left(speed):
    pwm.set_pwm(ENA, 0, speed)

def changespeed_right(speed):
    pwm.set_pwm(ENB, 0, speed)

def stopcar():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm.set_pwm(ENA, 0, 0)
    pwm.set_pwm(ENB, 0, 0)

def custom_speed(speed_left, speed_right):
    #make all motors moving forward at the speed of variable move_speed
    if speed_left < 0:
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN1, GPIO.HIGH)
        speed_left = -speed_left
    else:
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN1, GPIO.LOW)

    if speed_right < 0:
        GPIO.output(IN4, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        speed_right = -speed_right
    else:
        GPIO.output(IN4, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)

    speed_left += 500
    speed_right += 500

    changespeed_left(speed_left)
    changespeed_right(speed_right)

# ================================= random move ================================= 

def decode_xyz(x,y,z):
    return (x,y,z)

# change a1 a2 a3 order according to calibration
def decode_rot(a1, a2, a3):
    return (a1, a2, a3)

def calculate_speed(v_lin, v_ang):
    # left_motor = 4.899 * v_lin - 384.44 * v_ang
    # right_motor = 4.892 * v_lin + 427.5061 * v_ang

    right_motor = 11.81 * v_ang + 8.12307 * v_lin - 182.898 
    left_motor = -11.81 * v_ang + 8.12307 * v_lin - 164.878

    return left_motor, right_motor

def move(steps, map_func, move_func):
    for i in range(steps):
        next_x = np.random.rand(1)
        next_y = np.random.rand(1)

        next_x = map_func(next_x)
        next_y = map_func(next_y)

        move_func(next_x, next_y)

def create_body_index(xml_string):
    """ Extract a name to index dictionary from 6dof settings xml """
    xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index

def angle_diff(a1, a2):
    if isinstance(a1, np.ndarray):
        diff = a1 - a2
        diff[diff > 180] = diff[diff > 180] - 360
        diff[diff < -180] = diff[diff < -180] + 360
    else:
        diff = a1 - a2
        if diff > 180:
            return diff - 360
        if diff < -180:
            return 360 + diff

    return diff

async def pos_receiver(qtm_address="192.168.2.28"):
    global pos_message

    connection = await qtm.connect(qtm_ip)
    if connection is None:
        print("Failed to connect")
        return

    async with qtm.TakeControl(connection, ""):
        await connection.new()

    # Get 6dof settings from qtm
    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    # tracked body name
    wanted_body = f"car_{self_id}"

    def on_packet(packet):
        global pos_message
        timestamp = packet.timestamp
        info, bodies = packet.get_6d_euler()

        if wanted_body is None or wanted_body not in body_index:
            print(f"target '{wanted_body}' not detected")

        for body in body_index.keys():
            # Extract one specific body
            current_index = body_index[body]
            position, rotation = bodies[current_index]

            x, y = position.x, position.y
            
            if opt.rot_index == 1:
                a1 = rotation.a1
            elif opt.rot_index == 2:
                a1 = rotation.a2
            elif opt.rot_index == 3:
                a1 = rotation.a3

            if body == wanted_body:
                pos_message['self'] = (float(x), float(y), float(a1))
            else:
                pos_message[body] = (float(x), float(y), float(a1))

        # if wanted_body is not None and wanted_body in body_index:
        #     # Extract one specific body
        #     wanted_index = body_index[wanted_body]
        #     position, rotation = bodies[wanted_index]

        #     # update global pos there
        #     #pos = re.findall(r'\(.*?\)', position)[0]
        #     #rot = re.findall(r'\(.*?\)', rotation)[0]
        #     #x,y,z = eval('decode_xyz' + pos)
        #     #a1, a2, a3 = eval('decode_rot' + rot)

        #     x, y = position.x, position.y
            
        #     if opt.rot_index == 1:
        #         a1 = rotation.a1
        #     elif opt.rot_index == 2:
        #         a1 = rotation.a2
        #     elif opt.rot_index == 3:
        #         a1 = rotation.a3

        #     pos_message = [(float(x), float(y), float(a1))]
        # else:
        #     # Print all bodies
        #     print(f"target '{wanted_body}' not detected")

    # Start streaming frames
    while True:
        await connection.stream_frames(components=["6deuler"], on_packet=on_packet)
        await asyncio.sleep(0.01)
        await connection.stream_frames_stop()

class PicarControl(Node):
    def __init__(self, move_period=3):
        super().__init__('picar_control')
        self.step = 0.01

        # map to arena size
        self.x_low = min(opt.corner1[0], opt.corner2[0])
        self.x_high = max(opt.corner1[0], opt.corner2[0])
        self.y_low = min(opt.corner1[1], opt.corner2[1])
        self.y_high = max(opt.corner1[1], opt.corner2[1])

        self.center = ((self.x_low + self.x_high) / 2., (self.y_low + self.y_high) / 2.)
        self.x_span = abs(self.x_high - self.x_low)
        self.y_span = abs(self.y_high - self.y_low)

        linear_func_file = '/home/pi/ros2_projects/track_1000-3500_function.csv'
        rot_func_file = '/home/pi/ros2_projects/track_rotate_function.csv'
        self.speed_rk = SpeedSolver(linear_func_file, rot_func_file)

        self.publisher = self.create_publisher(Pose, f'/picar_{self_id}/goal', 10)
        self.timer = self.create_timer(move_period, self.move_callback)

    def move_callback(self):
        # random sample move destination
        next_x = np.random.rand(1)
        next_y = np.random.rand(1)
        # next_x = self.map_position(next_x)
        # next_y = self.map_position(next_y)
        next_x, next_y = self.map_position(next_x, next_y)

        self.move_to(next_x, next_y)

    def move_to(self, next_x, next_y, sleep_time=None):
        global pos_message

        goal_message = Pose()
        goal_message.x = float(next_x)
        goal_message.y = float(next_y)

        self.publisher.publish(goal_message)

        if len(pos_message.keys()) == 0:
            print("position not received")
            return
        else:
            target_vec = (next_x - pos_message['self'][0], next_y - pos_message['self'][1])

            target_dist = math.sqrt( (next_x - pos_message['self'][0]) ** 2 \
                                   + (next_y - pos_message['self'][1]) ** 2 )

            # check if it's angular or radius
            target_angle = np.sign(target_vec[1]) * \
                           math.acos(target_vec[0] / target_dist)
            target_angle = target_angle * 180 / np.pi

            angle_to_move = -angle_diff(pos_message['self'][2], target_angle)
            start_angle = pos_message['self'][2]

            print(f"target angle: {target_angle}, current angle: {pos_message['self'][2]}, angle to move: {angle_to_move}")
            if type(angle_to_move) is type(np.array([0])):
                angle_to_move = angle_to_move[0]
            else:
                angle_to_move = angle_to_move

            # rotate
            if angle_to_move > 0:
                rot_l, rot_r = calculate_speed(0, opt.angular_speed) # self.speed_rk.solve_rotate_reverse(np.pi * 2)
            else:
                rot_l, rot_r = calculate_speed(0, -opt.angular_speed) # self.speed_rk.solve_rotate_reverse(-np.pi * 2)

            #rot_l, rot_r = 1500, -2000
            ## set wheel speed
            rot_l, rot_r = int(rot_l), int(rot_r)
            one_round = 360 / opt.angular_speed - opt.rotate_scale

            # move_time = abs(angle_to_move / opt.angular_speed)
            move_time = abs(angle_to_move / 360 * one_round) + 1
            custom_speed(rot_l, rot_r)

            # time.sleep(move_time + opt.rotate_scale * (360 / opt.angular_speed)) # 1.22  # wait for rotate while let other thread run
            # time.sleep(move_time)
            time.sleep(opt.rotate_scale)

            while abs(angle_diff(pos_message['self'][2], target_angle)) > 5:
                # print(f"current angle {pos_message['self'][2]}, target: {target_angle}, diff {angle_diff(pos_message['self'][2], target_angle)}")
                time.sleep(opt.update_time)

            # stop car
            stopcar()

            print(f"angle moved: {angle_diff(pos_message['self'][2], start_angle)}")

            # translate
#            lin_l = self.speed_rk.solve_speed_reverse("left", 500)
#            lin_r = self.speed_rk.solve_speed_reverse("right", 500)

            lin_l, lin_r = calculate_speed(400,0)
            #lin_l, lin_r = 2000, 2500 
            ## set wheel speed
            lin_l, lin_r = int(lin_l), int(lin_r)
            move_time = target_dist / 400.
            custom_speed(lin_l, lin_r)

            turn_back = False
            # wait for move while let other thread run
            # time.sleep(move_time)  
            if sleep_time is not None:
                time.sleep(sleep_time)
                move_time = max(move_time - sleep_time, 0)

            for i in range(int(move_time / opt.update_time)):
                time.sleep(opt.update_time)
                # global pos_message
                near_dist, near_pos, near_key = self.calculate_near(pos_message)
                
                self_x, self_y = pos_message['self'][0], pos_message['self'][1]

                # print(f"nearst car: {near_key}, dist {near_dist}")

                if self.close_to_border(self_x, self_y):

                    print(f"close to border {self_x} {self_y}")
                    turn_back = True
                    next_x, next_y = self.center
                    break
                # avoid hitting another car
                elif near_dist < opt.minimal_dist:
                    print(f"cars too close")
                    print(f"nearst car: {near_key}, dist {near_dist}")
                    turn_back = True
                    other_x, other_y = near_pos

                    next_x = self_x + opt.away_scale * (self_x - other_x)
                    next_y = self_y + opt.away_scale * (self_y - other_y)
                    break

            # stop car
            stopcar()

            pos_message = {}

            time.sleep(1)

            if turn_back:
                self.move_to(next_x, next_y, 1)

    def close_to_border(self, self_x, self_y):
        if (self_x - self.x_low < opt.minimal_dist) or \
           (self_y - self.y_low < opt.minimal_dist) or \
           (self.x_high - self_x < opt.minimal_dist) or \
           (self.y_high - self_y < opt.minimal_dist):
            return True
        else:
            return False

    def calculate_near(self, positions):
        self_x = positions['self'][0]
        self_y = positions['self'][1]

        near_dist = 1e7
        near_pos = None
        near_key = None
        for k in positions.keys():
            if k == 'self':
                continue

            other_x, other_y = positions[k][0], positions[k][1]
            if math.isnan(other_x) or math.isnan(other_y):
                continue

            dist = np.linalg.norm(np.array([self_x, self_y]) - np.array([other_x, other_y]))

            if dist < near_dist:
                near_dist = dist
                near_pos = (other_x, other_y)
                near_key = k

        return near_dist, near_pos, near_key

    def map_position(self, x, y):
        return (self.center[0] + (x - 0.5) * self.x_span,
                self.center[1] + (y - 0.5) * self.y_span)

def main(args=None):
    global pos_message
    pos_message = {}
    rclpy.init(args=args)
    # turtle = TurtleControl()
    # rclpy.spin(turtle)
    # turtle.destroy_node()
    # rclpy.shutdown()

    controller = PicarControl()
    thread1 = threading.Thread(target=rclpy.spin, args=(controller,))
    thread1.start()
    thread2 = threading.Thread(target=asyncio.run, args=(pos_receiver(),))
    thread2.start()

if __name__ == '__main__':
    main()

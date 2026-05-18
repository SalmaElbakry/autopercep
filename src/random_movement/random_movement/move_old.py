import asyncio
import math
import re
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import qtm
import rclpy
from geometry_msgs.msg import Twist
from perlin_noise import PerlinNoise
from rclpy.node import Node
from turtlesim.msg import Pose

from reverse_speed import SpeedSolver

pos_message = None

def decode_xyz(x,y,z):
    return (x,y,z)

# change a1 a2 a3 order according to calibration
def decode_rot(a1, a2, a3):
    return (a1, a2, a3)

def control_speed(left_v, right_v):
    # placeholder for picar speed control
    # do no sleep in this block
    print(f'speed set to {left_v}, {right_v}')

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

    connection = await qtm.connect(qtm_address)
    if connection is None:
        print("Failed to connect")
        return

    async with qtm.TakeControl(connection, ""):
        await connection.new()

    # Get 6dof settings from qtm
    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)

    # tracked body name
    wanted_body = "car"

    def on_packet(packet):
        timestamp = packet.timestamp
        info, bodies = packet.get_6d_euler()

        if wanted_body is not None and wanted_body in body_index:
            # Extract one specific body
            wanted_index = body_index[wanted_body]
            position, rotation = bodies[wanted_index]

            # update global pos there
            pos = re.findall(r'\(.*?\)', position)[0]
            rot = re.findall(r'\(.*?\)', rotation)[0]
            x,y,z = eval('decode_xyz' + pos)
            a1, a2, a3 = eval('decode_rot' + rot)

            pos_message = (float(x), float(y), float(a1))
        else:
            # Print all bodies
            print(f"target '{wanted_body}' not detected")

    # Start streaming frames
    await connection.stream_frames(components=["6deuler"], on_packet=on_packet)
    await asyncio.sleep(1)
    await connection.stream_frames_stop()

class PicarControl(Node):
    def __init__(self, move_period=3):
        super().__init__('turtle_control')
        self.step = 0.01

        linear_func_file = '/home/ruiheng/Documents/picar/track_1000-3500_function.csv'
        rot_func_file = '/home/ruiheng/Documents/picar/track_rotate_function.csv'
        self.speed_rk = SpeedSolver(linear_func_file, rot_func_file)

        self.publisher = self.create_publisher(Pose, '/picar/goal', 10)
        self.timer = self.create_timer(move_period, self.move_callback)

    def move_callback(self):
        global pos_message
        next_x = np.random.rand(1)
        next_y = np.random.rand(1)
        # next_x = self.map_position(next_x)
        # next_y = self.map_position(next_y)
        next_x, next_y = self.map_position(next_x, next_y)

        goal_message = Pose()
        goal_message.x = next_x
        goal_message.y = next_y

        self.publisher.publish(goal_message)

        if pos_message is None:
            print("position not received")
            return
        else:
            target_vec = (next_x - pos_message[0], next_y - pos_message[1])

            target_dist = math.sqrt( (next_x - pos_message[0]) ** 2 \
                                   + (next_y - pos_message[1]) ** 2 )

            # check if it's angular or radius
            target_angle = np.sign(target_vec[1]) * \
                           math.acos(target_vec[0] / target_dist)
            target_angle = target_angle * 180 / np.pi

            angle_to_move = angle_diff(pos_message[2], target_angle)
            # rotate
            rot_l, rot_r = self.speed_rk.solve_speed_reverse(np.pi * 2)
            ## set wheel speed
            move_time = angle_to_move / 360.
            control_speed(rot_l, rot_r)

            time.sleep(move_time)  # wait for rotate while let other thread run
            # stop car

            # translate
            lin_l = self.speed_rk.solve_speed_reverse("left", 500)
            lin_r = self.speed_rk.solve_speed_reverse("right", 500)
            ## set wheel speed
            move_time = target_dist / 500.
            control_speed(lin_l, lin_r)

            time.sleep(move_time)  # wait for move while let other thread run
            # stop car

            pos_message = None

    def map_position(self, x, y):
        # map to arena size
        center = (0., 0.)
        x_max = 700
        y_max = 700

        return (center[0] + (x - 0.5) * 2 * (x_max - center[0]),
                center[1] + (y - 0.5) * 2 * (y_max - center[1]))

class TurtleControl(Node):
    def __init__(self):
        super().__init__('turtle_control')
        self.step = 0.01

        self.pose = (0.,0.)
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.listener = self.create_subscription(Pose, '/turtle1/pose', self.store_pose_callback, 10)
        # self.random_move()
        self.timer = self.create_timer(2, self.random_move)

    def store_pose_callback(self, msg):
        # print(f"received pose: {msg}")
        self.pose = (msg.x, msg.y)

    def random_move(self):
        # while True:
        next_x = np.random.rand(1)
        next_y = np.random.rand(1)

        print(f"x: {next_x[0]}, y: {next_y[0]}")
        print(f"position: {self.pose[0]}, {self.pose[1]}")

        next_x = self.map_position(next_x)
        next_y = self.map_position(next_y)

        vel_message = Twist()
        vel_message.linear.x = (next_x - self.pose[0])[0]
        vel_message.linear.y = (next_y - self.pose[1])[0]

        self.publisher.publish(vel_message)
        time.sleep(1)

        vel_message.linear.x = 0.
        vel_message.linear.y = 0.
        vel_message.angular.z = 2 * np.pi

        self.publisher.publish(vel_message)
        time.sleep(1)

        vel_message.angular.z = 0.
        self.publisher.publish(vel_message)

        self.step += 0.1


    def map_position(self, position):
        return position * 5 + 5
        # return (position - 0.5) * 5 + 5
        # return position * 10

def main(args=None):
    rclpy.init(args=args)
    # turtle = TurtleControl()
    # rclpy.spin(turtle)
    # turtle.destroy_node()
    # rclpy.shutdown()

    controller = PicarControl()
    thread1 = threading.Thread(target=rclpy.spin, args=(controller,))
    thread1.start()
    thread2 = threading.Thread(target=asyncio.run, args=(pos_receiver,))
    thread2.start()

if __name__ == '__main__':
    main()

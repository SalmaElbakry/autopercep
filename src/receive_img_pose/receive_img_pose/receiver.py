import asyncio
import csv
import json
import os
import threading
from datetime import datetime
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import qtm
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

pose_message_buffer = {}
qtm_ip = os.environ['QTM_IP']
self_id = os.environ['SELF_ID']
with open("parameters.json", 'r') as f:
    parameters = json.load(f)

# asyncio get incoming message
async def handle_echo(reader, writer):
    global pose_message_buffer
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info('peername')

    timestamp = message.split(' ')[-1]
    pose_message_buffer = (' '.join(message.split(' ')[:-1]), timestamp)

    print(f"Received {message!r} from {addr!r}")

    # print(f"Send: {message!r}")
    writer.write(data)
    await writer.drain()

    # print("Close the connection")
    writer.close()

async def start_server():
    server = await asyncio.start_server(
        handle_echo, '127.0.0.1', 8888)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

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

def create_body_index(xml_string):
    """ Extract a name to index dictionary from 6dof settings xml """
    xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index

async def pos_receiver(qtm_address="192.168.2.28"):
    global pose_message_buffer
    global parameters

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
    wanted_list = parameters['wanted_bodies']

    def on_packet(packet):
        global pose_message_buffer
        timestamp = packet.timestamp
        info, bodies = packet.get_6d_euler()

        for wanted_body in wanted_list:
            if wanted_body is not None and wanted_body in body_index:
                # Extract one specific body
                wanted_index = body_index[wanted_body]
                position, rotation = bodies[wanted_index]

                # update global pos there
                #pos = re.findall(r'\(.*?\)', position)[0]
                #rot = re.findall(r'\(.*?\)', rotation)[0]
                #x,y,z = eval('decode_xyz' + pos)
                #a1, a2, a3 = eval('decode_rot' + rot)

                x, y = position.x, position.y

                if int(parameters["rot_index"]) == 1:
                    a1 = rotation.a1
                elif int(parameters["rot_index"]) == 2:
                    a1 = rotation.a2
                elif int(parameters["rot_index"]) == 3:
                    a1 = rotation.a3

                pose_message_buffer[wanted_body] = (float(x), float(y), float(a1), timestamp)
            else:
                # Print all bodies
                print(f"target '{wanted_body}' not detected")

    # Start streaming frames
    while True:
        await connection.stream_frames(components=["6deuler"], on_packet=on_packet)
        await asyncio.sleep(0.01)
        await connection.stream_frames_stop()

class Receiver(Node):
    def __init__(self):
        super().__init__('picar_receiver_node')

        # ros2 parameter to adjust behavior
        # self.declare_parameter('save_folder', '/home/pi/360_images')
        # self.declare_parameter('save_interval', 0.5)

        with open("parameters.json", 'r') as f:
            self.parameters = json.load(f)

        save_folder = parameters["save_folder"]  # self.get_parameter('save_folder').get_parameter_value().string_value
        save_interval = parameters["save_interval"]  # self.get_parameter('save_interval').get_parameter_value().integer_value

        current_time = datetime.now()
        csv_out = open(os.path.join(save_folder, f'records_{current_time.strftime("%Y%m%d-%H%M")}.csv'), 'w', newline='')
        fields = ['image', 'image_timestamp']

        for body_name in self.parameters["wanted_bodies"]:
            if body_name == 'car_' + self_id:
                body_name = 'self'

            fields.append(body_name + '_pose_x')
            fields.append(body_name + '_pose_y')
            fields.append(body_name + '_pose_angle')
            fields.append(body_name + '_pose_timestamp')

        self.writer = csv.DictWriter(csv_out, fieldnames=fields)
        self.writer.writeheader()

        self.timer = self.create_timer(save_interval, self.store_image_callback)
        self.subscription = self.create_subscription(
            CompressedImage, 'image_raw/compressed', self.reterive_image_callback, 10
            # Image, 'image_raw', self.reterive_image_callback, 10
        )
        self.bridge = CvBridge()
        self.buffer = None
        self.img_folder = os.path.join(save_folder, current_time.strftime("%Y%m%d"))
        os.makedirs(self.img_folder, exist_ok=True)

    def store_image_callback(self):
        """Save image and pos in current buffer."""
        global pose_message_buffer
        # pose_message_buffer = None

        if self.buffer is None:
            return
        img_data = self.buffer[0]
        # print(img_data.header)
        pose_data = None
        for body_name in self.parameters["wanted_bodies"]:
            if body_name not in pose_message_buffer:
                pose_data = ('pose not updated', 1000000)
                self.get_logger().info('pose not updated')
                return

        pose_data = pose_message_buffer
        pose_message_buffer = {}
        # pos_data = self.buffer[1]

        # for body_name in self.parameters["wanted_bodies"]:
            # self.get_logger().info(f'got pos {pose_data[body_name][0]} from timestamp {pose_data[body_name][1]}')
            # self.get_logger().info(f'storing image from timestamp {img_data.header.stamp.sec} {img_data.header.stamp.nanosec}')

        image = self.bridge.compressed_imgmsg_to_cv2(img_data)
        # image = self.bridge.imgmsg_to_cv2(img_data)
        self.save_img(image, str(img_data.header.stamp.sec) + str(img_data.header.stamp.nanosec), pose_data)
        # cv2.imshow("camera", image)
        # cv2.waitKey(1)

        buffer = None

    def reterive_image_callback(self, data):
        """
        Get image and store it in buffer (overwrite).

        :param data: image data from pi car
        """
        #
        # self.get_logger().info('')
        self.buffer = (data, None)  # image and position

    def calculate_angle(self, self_pose, other_pose):
        self_xy = np.array([self_pose[0], self_pose[1]])
        other_xy = np.array([other_pose[0], other_pose[1]])

        angle_to_other = np.arctan2(other_pose[1] - self_pose[1], other_pose[0] - self_pose[0])
        angle_to_other = angle_to_other / np.pi * 180
        angle_between = angle_diff(angle_to_other, self_pose[2])

        return angle_between

    def save_img(self, img, stamp, pose_data):
        csv_row = {'image': os.path.join(self.img_folder, f'image_{stamp}.png'),
                   'image_timestamp': stamp}

        self_pose = (0,0,0)
        for body_name in self.parameters["wanted_bodies"]:
            if body_name == 'car_' + self_id:
                field_name = 'self'

                self_pose = (pose_data[body_name][0], pose_data[body_name][1], pose_data[body_name][2])

                csv_row[field_name + '_pose_x'] = pose_data[body_name][0]
                csv_row[field_name + '_pose_y'] = pose_data[body_name][1]
                csv_row[field_name + '_pose_angle'] = pose_data[body_name][2]
                csv_row[field_name + '_pose_timestamp'] = pose_data[body_name][3]

        # pose filter
        lower_angle = float(self.parameters["angle_range"][0])
        upper_angle = float(self.parameters["angle_range"][1])
        # print(f"angle_range: {lower_angle} to {upper_angle}")

        save_this_img = True

        for body_name in self.parameters["wanted_bodies"]:
            if body_name == 'car_' + self_id:
                continue
            else:
                field_name = body_name

            other_pose = (pose_data[body_name][0], pose_data[body_name][1], pose_data[body_name][2])

            target_angle = self.calculate_angle(self_pose, other_pose)
            

            csv_row[field_name + '_pose_x'] = pose_data[body_name][0]
            csv_row[field_name + '_pose_y'] = pose_data[body_name][1]
            csv_row[field_name + '_pose_angle'] = pose_data[body_name][2]
            csv_row[field_name + '_pose_timestamp'] = pose_data[body_name][3]

            if (target_angle > upper_angle) or (target_angle < lower_angle):
                save_this_img = False
                # csv_row[field_name + '_pose_angle'] = "out range"

        self.writer.writerow(csv_row)

        if save_this_img:
            print(f"saving angle between self and {body_name} is {target_angle}, image name: image_{stamp}")
            cv2.imwrite(os.path.join(self.img_folder, f'image_{stamp}.png'), img)

        # self.writer.writerow({'image': os.path.join(self.img_folder, f'image_{stamp}.png'),
        #                       'pose_x': pose_data[0],
        #                       'pose_y': pose_data[1],
        #                       'pose_angle': pose_data[2],
        #                       'image_timestamp': stamp,
        #                       'pose_timestamp': pose_data[3]})

def main(args=None):
    rclpy.init(args=args)

    receiver = Receiver()
    thread1 = threading.Thread(target=rclpy.spin, args=(receiver,))
    thread1.start()
    thread0 = threading.Thread(target=asyncio.run, args=(pos_receiver(),))
    thread0.start()

    # receiver.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

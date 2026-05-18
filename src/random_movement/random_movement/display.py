import asyncio
import threading
import xml.etree.ElementTree as ET
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import qtm
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose

pos_message = None

def create_body_index(xml_string):
    """ Extract a name to index dictionary from 6dof settings xml """
    xml = ET.fromstring(xml_string)

    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index

    return body_to_index

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
        global pos_message
        timestamp = packet.timestamp
        info, bodies = packet.get_6d_euler()

        if wanted_body is not None and wanted_body in body_index:
            # Extract one specific body
            wanted_index = body_index[wanted_body]
            position, rotation = bodies[wanted_index]

            # update global pos there
            # change rotation accordingto calibration
            x, y, a1 = position.x, position.y, rotation.a3

            pos_message = (float(x), float(y), float(a1))
        else:
            # Print all bodies
            print(f"target '{wanted_body}' not detected")

    # Start streaming frames
    while True:
        await connection.stream_frames(components=["6deuler"], on_packet=on_packet)
        await asyncio.sleep(0.01)
        await connection.stream_frames_stop()


class DrawPath(Node):
    def __init__(self):
        super().__init__('draw_path')
        self.listener = self.create_subscription(Pose, '/picar/goal',
                                                 self.draw_goal_callback, 10)
        self.timer = self.create_timer(0.02, self.draw_pos_callback)

        # creating initial data values
        # of x and y
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
       # ======================================

        self.position = (0., 0.)
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(12, 12))

        #self.ax.plot(np.array([0, -500]),
                     #np.array([0, 0]),)
                     # np.array([500, 500]),
                     # np.array([500, -300]))
        # self.line1, = self.ax.plot(x, y)

        plt.xlim([-1500, 1500])
        plt.ylim([-1500, 1500])

        # plt.show()

        # for i in range(20):
        #     message = Pose()

        #     message.x = float((np.random.rand(1) - 0.5) * 700 * 2)
        #     message.y = float((np.random.rand(1) - 0.5) * 700 * 2)
        #     self.draw_goal_callback(message)
        #     sleep(1)

    def draw_pos_callback(self):
        global pos_message
        if pos_message is None:
            print('pose not received')
            return

        x = [pos_message[0]]
        y = [pos_message[1]]

        self.ax.plot(x, y, marker="o", markersize=2, markerfacecolor="red")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #pos_message = None


    def draw_goal_callback(self, msg):
        global pos_message
        if pos_message is not None:
            self.position = (pos_message[0], pos_message[1])
        goal_x = msg.x
        goal_y = msg.y

        # x = np.linspace(0, 500, 1000)
        # new_y = np.sin(x-0.5*goal_x)
        # self.ax.plot(x, new_y)

        self.ax.plot(np.array([self.position[0], goal_x]),
                     np.array([self.position[1], goal_y]))

        # self.line1.set_xdata(x)
        # self.line1.set_ydata(new_y)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #self.position = (goal_x, goal_y)

def main(args=None):
    rclpy.init(args=args)

    #thread1 = threading.Thread(target=rclpy.spin, args=(drawer,))
    #thread1.start()
    thread2 = threading.Thread(target=asyncio.run, args=(pos_receiver(),))
    thread2.start()
    
    drawer = DrawPath()
    rclpy.spin(drawer)

    # drawer.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()

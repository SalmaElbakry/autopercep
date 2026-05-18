import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

import RPi.GPIO as GPIO
import Adafruit_PCA9685
import os
import time

self_id = os.environ['SELF_ID']

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

    # set left speed
    pwm.set_pwm(ENA, 0, int(speed_left))
    
    # set right speed
    pwm.set_pwm(ENB, 0, int(speed_right))

def stopcar():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm.set_pwm(ENA, 0, 0)
    pwm.set_pwm(ENB, 0, 0)

def calculate_speed(v_lin, v_ang):
    left_motor = 4.899 * v_lin - 384.44 * v_ang
    right_motor = 4.892 * v_lin + 427.5061 * v_ang

    return left_motor, right_motor

class SpeedReceiver(Node):
    def __init__(self):
        super().__init__("speed_receiver")

        self.last_time = time.perf_counter()
        self.subscription = self.create_subscription(
            Twist,
            f"cmd_vel_{self_id}",
            self.listener_callback,
            10
        )
        print(f"receiving speed from topic: cmd_vel_{self_id}")

    def listener_callback(self, msg):
        current_time = time.perf_counter()
        print(f'Received command at time {current_time}.')
        print(f'Time elapsed: {current_time - self.last_time} s')
        v_lin = 1000 * msg.linear.x  # m/s to mm/s
        v_ang = msg.angular.z

        v_left, v_right = calculate_speed(v_lin, v_ang)

        print(f'receive speed: linear {v_lin}, angular {v_ang}')    
        print(f'output motor: left {v_left}, right {v_right}')

        custom_speed(v_left, v_right)
        self.last_time = time.perf_counter()

    def stop_car(self):
        stopcar()


def main(args=None):
    rclpy.init(args=args)

    subscriber = SpeedReceiver()

    rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Twist, 'speed_test', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.lin_speeds = range(5, 12)
        self.i = 0

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 50.1 * self.lin_speeds[self.i % 7]
        msg.angular.z = 2.5
        self.publisher_.publish(msg)
        self.get_logger().info(f'linear speed: {msg.linear.x}, angular speed: {msg.angular.z}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

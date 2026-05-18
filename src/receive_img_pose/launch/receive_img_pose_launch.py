from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='receive_img_pose',
            executable='receive_image',
            name='imageReceiver',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'save_folder': '/home/ruiheng/Documents/ros2_projects/picar_images/',
                 'save_interval': 5}
            ]
        )
    ])

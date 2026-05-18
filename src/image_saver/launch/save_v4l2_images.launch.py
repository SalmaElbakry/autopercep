from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_saver',
            executable='save_image_node',
            name='image_saver',
            output='screen',
            parameters=[
                #{'image_topic': '/image_raw'},          # set to your v4l2_camera image topic
                #{'output_dir': '/tmp/v4l2_images'},     # change as needed
                {'image_format': 'png'},                # 'jpg' or 'png'
                {'save_every_n': 2},                    # save every frame
                {'use_header_stamp': True},
                #{'csv_name': 'images_log.csv'},
                {'overwrite_existing_csv': False},
            ],
        )
    ])


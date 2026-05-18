from setuptools import setup

package_name = 'image_saver_csv'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/save_v4l2_images.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Subscribe to a v4l2 image topic, save images to disk, and log timestamps to CSV.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'save_image_node = image_saver.save_image_node:main',
        ],
    },
)


from setuptools import setup

package_name = 'speed_receiver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruiheng',
    maintainer_email='preon7@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'receive_speed = speed_receiver.receiver:main',
        	'test_speed_publish = speed_receiver.test_speed_publisher:main',
        ],
    },
)

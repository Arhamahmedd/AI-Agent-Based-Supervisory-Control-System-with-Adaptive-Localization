from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lstm_localizer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arham Ahmed',
    maintainer_email='arham@example.com',
    description='LSTM-based adaptive EKF covariance predictor',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lstm_covariance_node = lstm_localizer.lstm_covariance_node:main',
            'ai_covariance_updater = lstm_localizer.ai_covariance_updater:main',
        ],
    },
)

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ai_supervisor'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arham Ahmed',
    maintainer_email='arham@example.com',
    description='AI Supervisory Control Agents',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_agent    = ai_supervisor.safety_agent:main',
            'flow_agent      = ai_supervisor.flow_agent:main',
            'override_agent  = ai_supervisor.override_agent:main',
            'decision_engine = ai_supervisor.decision_engine:main',
        ],
    },
)

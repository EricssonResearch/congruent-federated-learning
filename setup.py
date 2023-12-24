from setuptools import setup, find_packages

setup(
    name='cfl',
    version='0.1.0',
    packages=find_packages(include=['cfl', 'cfl.*']),
    package_data={'fg_traces' :['fg_traces/*.pickle']},
    include_package_data=True,
    install_requires=['torch>=2.0.1'],
)


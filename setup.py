from setuptools import setup, find_packages

setup(
    name='cfl',
    version='0.1.0',
    packages=find_packages(include=['cfl', 'cfl.*']),
    package_data={'fg_traces' :['fg_traces/*.pickle']},
    include_package_data=True,
    install_requires=['numpy~=1.24.4', 
                      'matplotlib~=3.7.3', 
                      'torch>=2.0.1', 
                      'pandas~=2.1.1', 
                      'scikit-learn~=1.2.2',
                      'torchvision'],
)


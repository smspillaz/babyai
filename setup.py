from setuptools import setup, find_packages

setup(
    name='babyai',
    version='1.1',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.6',
        'numpy==1.15.4', # Temporary: fix numpy version because of bug introduced in 1.16
        "torch>=0.4.1",
        'blosc>=1.5.1',
        'gym_minigrid @ https://github.com/maximecb/gym-minigrid/archive/master.zip'
    ],
)

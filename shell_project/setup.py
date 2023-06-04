from setuptools import setup, find_packages

setup(
    name='shell_project',
    version='1.0.0',
    author='Lian Fu',
    description='little shell project',
    packages=find_packages(),
    install_requires=[
        'click',
        # Other dependencies
    ],
    entry_points='''
        [console_scripts]
        find_details = main:cli
    '''
)

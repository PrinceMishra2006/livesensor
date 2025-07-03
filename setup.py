from setuptools import setup, find_packages

from typing import List


def get_requirements()-> list[str]:

    requirements_list = list[str] =[] # type: ignore

    return requirements_list




setup(
    name='sensor',
    version='0.0.1',
    author='prince',
    author_email='mishrajii1476@gmail.com',
    packages=get_requirements(),  # pymongo==4.2.0
     

)
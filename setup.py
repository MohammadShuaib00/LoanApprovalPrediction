import os
from setuptools import setup, find_packages
from typing import List


def get_requirements(filename: str = "requirements.txt") -> List[str]:
    requirement_list = []
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()

                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)
    except Exception as e:
        print(f"Requirements file not found! Exception: {e}")

    return requirement_list


setup(
    name="loanapprovalprediction",
    version="0.0.1",
    author="Mohammad Shuaib",
    author_email="mohammadshuaib3455@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)

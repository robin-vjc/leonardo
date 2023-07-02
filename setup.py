from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="leonardo",
    version="1.1",
    description="Leonardo - Image processing using Stable Difussion Models",
    author="Robin Vujanic",
    author_email="vjc.robin@gmail.com",
    packages=find_packages(),
    install_requires=required,
)

from setuptools import setup, find_packages

setup(
    name="p3_lib",
    version="0.1.0",
    packages=find_packages(include=["p3_lib", "p3_lib.*"]),
)

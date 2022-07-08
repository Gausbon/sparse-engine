from setuptools import setup, find_packages

setup(
    name="sparse-engine",
    version="1.0.0",
    description="Arrange tinyspos on mcu",
    url="https://github.com/gausbon/sparse-engine",
    author="Gausbon",
    author_email="gausbon-private@hotmail.com",
    packages=["sparse-engine"],
    install_requires=['torch >= 1.11', 'torchvision'],
    python_requires='>=3.8',
)

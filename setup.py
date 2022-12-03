import os
import re
from setuptools import find_packages, setup

# get version string from module
with open(
    os.path.join(os.path.dirname(__file__), "src", "lightning_ssl", "__init__.py"),
    "r",
) as f:
    pattern = r"__version__ = ['\"]([^'\"]*)['\"]"
    version = re.search(pattern, f.read(), re.M).group(1)  # type: ignore

# Get package dependencies from requirement files
with open(os.path.join("src", "requirements.txt"), "r") as fin:
    reqs = fin.readlines()

setup(
    name="lightning_ssl",
    version=version,
    author="Riccardo Musmeci",
    author_email="riccardomusmeci92@gmail.com",
    description="Self Supervised Learning library with PyTorch-Lightning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"lightning_ssl": ["py.typed"]},
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <=3.11",
    zip_safe=False,
)

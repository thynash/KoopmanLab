import io
import setuptools

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="koopmanlab",
    version="1.0.4",
    author="Wei Xiong, Tian Yang",
    author_email="xiongw21@mails.tsinghua.edu.cn",
    description="A library for Koopman Neural Operator with Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koopman-Laboratory/KoopmanLab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.5',
    install_requires=[
    "torch>=2.2",
    "torchvision>=0.17",
    "matplotlib>=3.8",
    "numpy>=1.26",
    "einops>=0.7.0",
    "timm>=1.0.7",
    "scipy>=1.11",
    "h5py>=3.10",
]
)

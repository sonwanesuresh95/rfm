import setuptools


setuptools.setup(
    name="rfm",
    version="1.0.15",
    author="Suresh Sonwane",
    author_email="sonwanesuresh739@gmail.com",
    description="Package for RFM Analysis and Customer Segmentation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=['pandas>=1.2.4', 'numpy>=1.20.1', 'matplotlib>=3.3.4']
)
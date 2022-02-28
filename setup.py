import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rfm",
    version="1.0.7",
    author="Suresh Sonwane",
    author_email="sonwanesuresh739@gmail.com",
    description="Python Package for RFM Analysis and Customer Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sonwanesuresh95/rfm",
    project_urls={
        "Bug Tracker": "https://github.com/sonwanesuresh95/rfm/issues",
    },
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

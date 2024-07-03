import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optlnls",
    version="0.3.17",
    author="Sergio Lordano",
    author_email="sergiolordano2@gmail.com",
    description="X-ray optics utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oasys-lnls-kit/optlnls",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',

)

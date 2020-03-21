import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optlnls", # Replace with your own username
    version="0.0.2",
    author="Sergio Lordano",
    author_email="sergiolordano2@gmail.com",
    description="X-ray optics utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SergioLordano/optlnls",
    packages=setuptools.find_packages(),
	install_requires=[
		"numpy", "matplotlib", "scipy"
	],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optlnls", # Replace with your own username
    version="0.1.6",
    author="Sergio Lordano",
    author_email="sergiolordano2@gmail.com",
    description="X-ray optics utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oasys-lnls-kit/optlnls",
    packages=setuptools.find_packages(),
	install_requires=[
    	'numpy>=1.18.0',
    	'scipy>=1.4.1',
    	'matplotlib>=3.1.2',
    	'srxraylib>=1.0.28',
	],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.2',

)

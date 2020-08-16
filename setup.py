import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cronus",
    version="0.0.1",
    author="Minas Karamanis",
    author_email="minaskar@gmail.com",
    description="cronus: Large Scale MCMC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minaskar/cronus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=['numpy', 'tqdm'],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['cronus-run=cronus.run:run_script'],
    }
)
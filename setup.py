import os
from setuptools import find_packages, setup


# read README.md
with open("README.md") as f:
    readme = f.read()


# write the package version
def write_version():
    with open(os.path.join("cytoemd", "version.txt")) as f:
        version = f.read().strip()

    with open(os.path.join("cytoemd", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


version = write_version()


def do_setup():
    setup(
        name='cytoemd',
        version=version,
        description="CytoEMD: a python package to characterize samples in Mass/Flow Cytometry Datasets.",
        url="https://github.com/CompCy-lab/cytoemd",
        long_description=readme,
        python_requires='>=3.6',
        install_requires=[
            "numpy",
            "pandas",
            "anndata",
            "umap-learn",
            "pyemd",
            "meld",
            "scanpy",
            "tqdm",
            'typing_extensions; python_version<"3.8"'
        ],
        packages=find_packages(
            exclude=[
                "notebooks",
                "notebooks.*",
                "tests",
                "tests.*",
                "results",
                "resutls.*"
            ]
        ),
        test_suite="tests",
        classifiers=[
            "Intended Audience :: Science/Research :: Computational Biology",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Computational Biology",
        ],
        zip_safe=False
    )


if __name__ == '__main__':
    do_setup()

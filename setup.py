from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "cbernoulli",
        ["./cbernoulli.pyx"],
    )
]

setup(
    name="CapCurApprox",
    version="0.1",
    packages=['CapCurApprox'],
    ext_modules=cythonize(extensions),  # Compile the Cython file
    install_requires=[
        'Cython',
    ],
)

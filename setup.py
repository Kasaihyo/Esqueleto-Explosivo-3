from setuptools import find_packages, setup

setup(
    name="esqueleto-explosivo-3-simulator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
        "numba>=0.58.0",
        "pyarrow>=14.0.0",
        "matplotlib>=3.7.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
    description="Optimized simulator for Esqueleto Explosivo 3 slot game",
    author="Flux Gaming",
    author_email="info@fluxgaming.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

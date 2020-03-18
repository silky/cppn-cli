from setuptools import setup

setup(name="cppn-cli",
        version='1',
        description="CPPN CLI Tool",
        url="https://github.com/silky/cppn-cli",
        author="Noon van der Silk",
        author_email="noonsilk@gmail.com",
        license="MIT",
        install_requires=[
            "tensorflow==1.8.0",
            "numpy==1.14.3",
            "Pillow==5.1.0",
            "GPy==1.9.2",
            "GPyOpt==1.2.5",
            "scikit-image==0.14.0",
            "scipy==1.1.0",
        ],
        packages=["cppn"],
        scripts=["bin/cppn"])

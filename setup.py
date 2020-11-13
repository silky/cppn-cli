from setuptools import setup

setup(name="cppn-cli",
        version='1',
        description="CPPN CLI Tool",
        url="https://github.com/silky/cppn-cli",
        author="Noon van der Silk",
        author_email="noonsilk@gmail.com",
        license="MIT",
        install_requires=[
            "tensorflow==2.3.1",
            "numpy==1.14.3",
            "Pillow",
            "GPy==1.9.2",
            "GPyOpt==1.2.5",
            "scikit-image",
            "scipy",
        ],
        packages=["cppn"],
        scripts=["bin/cppn"])

import pathlib
from setuptools import setup


HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name='tacklebox',
    version='1.0.0',
    description='Improved handling of PyTorch module hooks',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/IsaacRe/tacklebox',
    author='Isaac Rehg',
    author_email='isaacrehg@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    packages=[
        'tacklebox',
    ],
    include_package_data=True,
    install_requires=[
        'torch',
        'inspect',
        'json',
        'tqdm',
        'warnings'
    ],
    entry_points={
        'console_scripts': [
            'tacklebox.__main__:main'
        ]
    }
)
from setuptools import setup

with open('requirements.txt', 'r') as f:
    requires = f.read().splitlines()


setup(
    name="herd",
    packages=['herd'],
    version="0.0.1",
    author="David Mykytyn and Alex Malz",
    author_email="dwm261@nyu.edu",
    description="herding binaries",
    url="https://github.com/mykytyn/gaia-herd",
    license="GPL3",
    install_requires=requires
)
    
    

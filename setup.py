try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import gri

setup(name='gri',
      version=gri.__version__,
      description='Grand Index (13GRI) and Adjusted Grand Index (13AGRI) implementations.',
      author='Victor Alexandre Padilha',
      author_email='victorpadilha.cc@gmail.com',
      url='https://github.com/padilha/gri',
      py_modules='gri',
      install_requires=['numpy'])

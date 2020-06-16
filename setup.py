from setuptools import setup

setup(name='ptitprince',
      version='0.2.4',
      description='A Python implementation of Rainclouds, originally on R, ggplot2. Written on top of seaborn.',
      url='http://github.com/pog87/PtitPrince',
      author='Davide Poggiali',
      author_email='davide.poggiali@unipd.it',
      license='MIT',
      packages=['ptitprince'],
      install_requires=[
          'seaborn>=0.10', 'matplotlib', 'numpy>=1.13', 'scipy',
          'PyHamcrest>=1.9.0', 'cython'
      ],
      zip_safe=False)

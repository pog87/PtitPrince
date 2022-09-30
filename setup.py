from setuptools import setup, find_packages

setup(name='ptitprince',
      version='0.2.6',
      description='A Python implementation of Rainclouds, originally on R, ggplot2. Written on top of seaborn.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',      
      url='https://github.com/pog87/PtitPrince',
      author='Davide Poggiali',
      author_email='davide.poggiali@unipd.it',
      license='MIT',
      packages=find_packages(),
      platform='any',
      keywords=[
          'data visualization',
          'raincloud plots',
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
      ],      
      install_requires=[
          'seaborn==0.11',
          'matplotlib',
          'numpy>=1.16',
          'scipy',
          'PyHamcrest>=1.9.0',
          'cython'
      ],
      zip_safe=False)

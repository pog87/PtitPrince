# PtitPrince

A python implementation of the Raincloud plot!
[https://github.com/RainCloudPlots/RainCloudPlots](https://github.com/RainCloudPlots/RainCloudPlots)

## Installation
You can install it via
```
pip install ptitprince
```

or via conda
```
conda install -c pog87 ptitprince
```

or cloning this repo in your working directory.

-----
## Academic use

To **cite Raincloud plots** please use the following information:

> Allen M, Poggiali D, Whitaker K et al. Raincloud plots: a multi-platform tool for robust data visualization [version 1; peer review: 2 approved]. Wellcome Open Res 2019, 4:63. DOI: [10.12688/wellcomeopenres.15191.1](https://doi.org/10.12688/wellcomeopenres.15191.1)



![output](output_4_0.png)


## History of this project

 
This is a python version of the Raincloud plot (or PetitPrince plot, depending on the orientation) from R (under ggplot2) to Python.  The Raincloud plot is a variant of the violin plot written in R ggplot2 by [Micah Allen](https://micahallen.org/2018/03/15/introducing-raincloud-plots/).

I found a tweet asking for a .py version of the RainCloud plot, and I agreed to give it a try. Alas, the py version for ggplot2 does not allow to create new styles in a confortable way. So I decided to write this package using the [seaborn](https://seaborn.pydata.org/) library as a requisite.

---

Then I replicated the plots from the original post by [Micah Allen](https://micahallen.org/2018/03/15/introducing-raincloud-plots/), using Jupyter.

### Changelog

#### v.0.2.x

    * PtitPrince now relies on seaborn 0.10 and numpy >= 1.13
    * kwargs can be passed to the [cloud (default), boxplot, rain/stripplot, pointplot]
                     by preponing [cloud_, box_, rain_, point_] to the argument name.
    * End of support for python2, now the support covers python>=3.6

## Plans for the future:

 * ~~ask seaborn mantainers to add this new plot type~~ (not gonna happen)
 * ~~add a "move" option in seabon to control the positioning of each plot, as in ggplot2.~~ (either, added in ptitprince)
 * ~~get RainCloud published~~ (done!)
 * add logarithic density estimate (LDE) to the options for the cloud

------
[![Binder](https://img.shields.io/badge/binder%20tutorial-python-fb62f6.svg)](https://mybinder.org/v2/gh/RainCloudPlots/RainCloudPlots/master?filepath=tutorial_python%2Fraincloud_tutorial_python.ipynb)
[![Downloads](http://pepy.tech/badge/ptitprince)](http://pepy.tech/project/ptitprince)

[List](https://github.com/thomasjpfan/awesome-python-data-science) that metions this package.

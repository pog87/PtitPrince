# PtitPrince

A python implementation of the Raincloud plot!
[https://github.com/RainCloudPlots/RainCloudPlots](https://github.com/RainCloudPlots/RainCloudPlots)

You can install it via
```
pip install ptitprince
```

or cloning this repo in your working directory.

-----
 
 I tried to port to python the Raincloud plot (or PetitPrince plot, depending on the orientation) from R (under ggplot2) to Python.  The Raincloud plot is a variant of the violin plot written in R ggplot2 by [Micah Allen](https://micahallen.org/2018/03/15/introducing-raincloud-plots/). Everithing started with  this xkcd comic, making fun of the violin plot.

![xkcd](https://imgs.xkcd.com/comics/violin_plots.png)

---

I found a [tweet](https://twitter.com/flxrue/status/974639616912478210) asking for a .py version, and I agreed. Alas, the py version for ggplot2 does not allow to create new styles in a confortable way. So I decided to use the [seaborn](https://seaborn.pydata.org/) library and to rewrite a bit the `violin_plot` function. [This](half_viol.py) is the rewitten version.

---

Then I tried to replicate the plots from the original post by [Micah Allen](https://micahallen.org/2018/03/15/introducing-raincloud-plots/), using Jupyter.

## Plans for the future:

 * ask seaborn mantainers to add this new plot type
 * add a "move" option in seabon to control the positioning of each plot, as in ggplot2.

------
[![Binder](https://img.shields.io/badge/binder%20tutorial-python-fb62f6.svg)](https://mybinder.org/v2/gh/RainCloudPlots/RainCloudPlots/master?filepath=tutorial_python%2Fraincloud_tutorial_python.ipynb)
[![Downloads](http://pepy.tech/badge/ptitprince)](http://pepy.tech/project/ptitprince)

[List](https://github.com/thomasjpfan/awesome-python-data-science) that metions this package.

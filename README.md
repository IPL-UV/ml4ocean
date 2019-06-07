2019_ocean
---
Using python for ocean applications. For now just includes a few useful demo machine learning notebooks using the sklearn library.

---
#### Installation

1. Make sure that [miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed.
2. Clone the git repository
   ```bash
   git clone https://github.com/IPL-UV/ml4ocean.git
   ```
3. Create a new environment from the `.yml` file and activate.
   ```bash
    conda env create -f environment.yml
    source activate 2019_ml_ocean
   ```

---

### TODO

* Upgrade to scalable GP methods ([GPy](https://sheffieldml.github.io/GPy/), [GPyTorch](https://gpytorch.ai/)).
* Look into scalable Bayesian models ([Pyro](http://pyro.ai/))
* Investigate Bayesian Neural Networks.

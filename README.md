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
4. (Optional) If you already have the sample `.yml` file then to save time whenever I upgrade it, just run these commands.
   ```bash
   conda env update -f environment.yml
   ```

---
### Algorithms

All algorithms within this repository will be based on the `fit`, `predict` API of scikit-learn. This is a design choice and will allow us to utilize some of sklearn capabilities like `TransformedTargetRegressor` and `MultiOutputRegressor`. Furthermore, it's a familiar API and should simplify when reading the code. I will wrap all other libraries used (**GPy**, **GPyTorch**) under this fit, predict framework.

#### scikit-learn

This is the baseline framework which we assess how well we can model and estimate uncertainties using GPs for multi-output. This baseline does not scale well but it can be used to get an idea of how well we can obtain predictions.

* PCA Target Transform 
* Standard GP
    * MultiOutput
    * Multi-Task

#### GPy

This is a second wave of methods that we can look at in order to see how well GPs can be used for this application. These methods scale fairly well t

* Sparse GP (FITC) (**TODO**)
* Sparse GP (VFE) 


#### PyTorch

* Standard GP (**TODO**)
  * [MultiOutput](https://gpytorch.readthedocs.io/en/latest/examples/11_ModelList_GP_Regression/ModelList_GP_Regression.html)
  * Multi-Task
* [Sparse GP](https://gpytorch.readthedocs.io/en/latest/examples/05_Scalable_GP_Regression_Multidimensional/SGPR_Example_CUDA.html) (**TODO**)


### TODO

* Upgrade to scalable GP methods ( [GPyTorch](https://gpytorch.ai/)).
* Look into scalable Bayesian models ([Pyro](http://pyro.ai/))
* Investigate Bayesian Neural Networks.

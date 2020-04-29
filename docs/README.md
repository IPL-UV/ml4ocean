---
title: Overview
description: Overview of my ML4OCN Research
authors:
    - J. Emmanuel Johnson
path: docs/projects/ML4OCN
source: README.md
---
# Overview

I present some of my findings for working on machine learning applied to ocean applications.

## API

You can find the Documentation for the Codes [here](src/index.html)

---

## ARGO Project

!!! summary "Summary"
    In this project, we are trying to predict profiles of $Bbp$. This is a multi-output ML problem with high dimensional inputs and high dimensional outputs.

!!! details
    You can find the repository with all of the reproducible code, specific functions and notebooks at [github.com/IPL-UV/ml4ocean](https://github.com/IPL-UV/ml4ocean).

---

### Subset Dataset

We look at the North Atlantic and SubTropical Gyre regions to try and get an idea about how some ML methods might perform in this scenarios. As a first pass, we did minimal preprocessing with some standard PCA components reduction and we were ambitious and attempted to predict all 273 depths for the profiles. We had some success as we were able to do a decent job with Random forests and MultiLayer Perceptrons.

!!! fire "Relevant Materials"
    * ISPRS 2020 Publication - **Pending**

!!! check "ISPRS"
    * 0.0 - [Full Walkthrough](notebooks/na_data/4_pipeline/0.0_full_walkthrough)
    * 1.0 - [Experiment (Refactored)](notebooks/na_data/4_pipeline/2.0_experiment_refactored)
    * 2.0 - [Results](notebooks/na_data/4_pipeline/3.0_results)
    * 3.0 - [Validation](notebooks/na_data/4_pipeline/4.0_validation)

??? todo "Todo"
    * 0.0 - Data Exploration Notebook
    * 1.0 - Data Preprocessing Notebook
    * 2.0 - Loading the Preprocessed Data
    * 3.0 - Baseline ML Algorithm
    * 4.0 - Visualization & Summary Statistics

---

### Global Dataset

After we found success in the subset data, we decided to go for the global dataset and see how we do. We did reduce the problem difficulty by predicting only 19 depth levels instead of $\sim$273.

!!! check "Notebooks"
    * 2.0 - [Loading the Preprocessed Data](notebooks/global_data/1_load_processed_data)
    * 2.1 - [Refactored](notebooks/global_data/1.1_load_processed_data_refactored)
    * 3.0 - [Baseline ML Algorithm](notebooks/global_data/2_ml_algorithms)

??? todo "Todo"
    * 0.0 - Data Exploration Notebook
    * 1.0 - Data Preprocessing Notebook
    * 4.0 - Visualization & Summary Statistics

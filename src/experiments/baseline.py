import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from src.features.build_features import CycleTransform, GeoCartTransform

# Datasets
from src.data.make_dataset import (
    DataLoader,
    load_standard_data,
    load_high_dim_data,
    load_labels,
)


class DataParams:
    region = "stg"


class ProcessParams:
    n_components = 5
    valid_split = 0.2
    standardize = "before"
    seed = 123


class ModelParams:
    model = "gp"

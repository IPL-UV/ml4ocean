import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_stats(ypred: np.ndarray, y: np.ndarray, each_level: bool = False):

    stats = pd.DataFrame()

    # Get total average statistics
    if each_level == False:
        multioutput = "uniform_average"
        mae = mean_absolute_error(y, ypred, multioutput=multioutput)
        mse = mean_squared_error(y, ypred, multioutput=multioutput)
        rmse = mse
        r2 = r2_score(y, ypred, multioutput=multioutput)
        stats = pd.DataFrame(
            data=[[mae, mse, rmse, r2]], columns=["mae", "mse", "rmse", "r2"]
        )
    elif each_level == True:
        multioutput = "raw_values"
        stats["mae"] = mean_absolute_error(y, ypred, multioutput=multioutput)
        stats["mse"] = mean_squared_error(y, ypred, multioutput=multioutput)
        stats["rmse"] = np.sqrt(stats["mse"])
        stats["r2"] = r2_score(y, ypred, multioutput=multioutput)
    else:
        raise ValueError(f"Unrecognized stat request: {each_level}")

    return stats

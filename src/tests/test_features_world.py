from src.data.world import get_full_data
from src.features.world import subset_independent_floats


def test_subset_soca2016():

    # extract full dataframe
    full_df = get_full_data()

    # subset
    _, soca2016_df = subset_independent_floats(full_df, "soca2016")
    # check number of samples (meta, inputs)
    n_samples = 378
    error_msg = f"Incorrect number of samples for soca2016 floats: {soca2016_df.shape[0]} =/= {n_samples}"
    assert soca2016_df.shape[0] == n_samples, error_msg


def test_subset_isprs2020():

    # extract full dataframe
    full_df = get_full_data()

    # subset
    _, isprs2020_df = subset_independent_floats(full_df, "isprs2020")
    # check number of samples (meta, inputs)
    n_samples = 331
    error_msg = f"Incorrect number of samples for isprs2020 floats: {isprs2020_df.shape[0]} =/= {n_samples}"
    assert isprs2020_df.shape[0] == n_samples, error_msg


def test_subset_independent_floats():

    # extract full dataframe
    full_df = get_full_data()

    # subset
    ml_df, indep_df = subset_independent_floats(full_df, "both")

    # check number of samples (meta, inputs)
    n_samples = 24704
    error_msg = f"Incorrect number of samples for non-independent floats: {ml_df.shape[0]} =/= {n_samples}"
    assert ml_df.shape[0] == n_samples, error_msg

    # check number of samples (meta, inputs)
    n_samples = 331 + 378
    error_msg = f"Incorrect number of samples for independent floats: {indep_df.shape[0]} =/= {n_samples}"
    assert indep_df.shape[0] == n_samples, error_msg

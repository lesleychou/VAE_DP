import numpy as np
from sdv.evaluation import evaluate
from scipy import stats
import pandas as pd

from sdv.metrics.tabular import NumericalMLP, CategoricalSVM

# Distributional metrics - Check distribution differences between synthetic & original dataset as well as how
# Easy it is to discriminate them i.e. svc detection
def distribution_metrics(
    distributional_metrics,
    data_supp,
    synthetic_supp,
    categorical_columns
):
    # Need the data in same column order
    synthetic_supp = synthetic_supp[data_supp.columns]

    # Now categorical columns need to be converted to objects as SDV infers data
    # types from the fields and integers/floats are treated as numerical not categorical
    # TODO: is converted to object, then it can NOT check the metric for categorical_columns
    synthetic_supp[categorical_columns] = synthetic_supp[categorical_columns].astype(float)
    data_supp[categorical_columns] = data_supp[categorical_columns].astype(float)

    # evaluate on each column
    evals = pd.DataFrame()
    col_list = []
    single_col_evals_list = []
    p_value_list = []

    for col in data_supp:
        col_list.append(col)
        single_col_evals = evaluate(
            synthetic_supp[[col]], data_supp[[col]], metrics=distributional_metrics, aggregate=False
        )
        # return of stats.ks_2samp: KstestResult(statistic=0.15579, pvalue=8.4e-92)
        p_value = stats.ks_2samp(synthetic_supp[col], data_supp[col])[1]
        p_value_list.append(p_value)

        single_col_evals = single_col_evals["raw_score"].tolist()[0]
        single_col_evals_list.append(single_col_evals)

    evals['dimension'] = col_list
    evals['KSTest_score'] = single_col_evals_list
    evals['KSTest_p_value'] = p_value_list

    return evals


# Build in some privacy metrics from SDV - TO DO!!!


def privacy_metrics(
    private_variable,
    data_supp,
    synthetic_supp,
    categorical_columns,
    continuous_columns,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    if private_variable in continuous_columns:

        continuous_columns = [
            column for column in continuous_columns if column != private_variable
        ]

        mlp_priv = NumericalMLP.compute(
            data_supp.fillna(0),
            synthetic_supp.fillna(0),
            key_fields=(continuous_columns),
            sensitive_fields=[private_variable],
        )

        return mlp_priv

    elif private_variable in categorical_columns:

        categorical_columns = [
            column for column in categorical_columns if column != private_variable
        ]

        svm_priv = CategoricalSVM.compute(
            data_supp.fillna(0),
            synthetic_supp.fillna(0),
            key_fields=(categorical_columns),
            sensitive_fields=[private_variable],
        )

        return svm_priv
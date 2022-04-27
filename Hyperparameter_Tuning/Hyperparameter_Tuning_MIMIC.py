#%% -------- Import Libraries -------- #

# Standard imports
from selectors import EpollSelector
from tokenize import String
import numpy as np
import pandas as pd
import torch

# VAE is in other folder
import sys

sys.path.append("../")

# Opacus support for differential privacy
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# For datetime columns we need a transformer
from rdt.transformers import datetime

# Utility file contains all functions required to run notebook
from utils import (
    set_seed,
    mimic_pre_proc,
    constraint_filtering,
    plot_elbo,
    plot_likelihood_breakdown,
    plot_variable_distributions,
    reverse_transformers,
)
from metrics import distribution_metrics

import optuna
import pickle

import warnings

warnings.filterwarnings("ignore")  # We suppress warnings to avoid SDMETRICS throwing unique synthetic data warnings (i.e.
# data in synthetic set is not in the real data set) as well as SKLEARN throwing convergence warnings (pre-processing uses
# GMM from sklearn and this throws non convergence warnings)

set_seed(0)

filepath = ".../Private MIMIC Data/table_one_synthvae.csv"

# Load in the MIMIC dataset
data_supp = pd.read_csv(filepath)

# Save the original columns

original_categorical_columns = [
    "ETHNICITY",
    "DISCHARGE_LOCATION",
    "GENDER",
    "FIRST_CAREUNIT",
    "VALUEUOM",
    "LABEL",
]
original_continuous_columns = ["SUBJECT_ID", "VALUE", "age"]
original_datetime_columns = ["ADMITTIME", "DISCHTIME", "DOB", "CHARTTIME"]

# Drop DOD column as it contains NANS - for now

# data_supp = data_supp.drop('DOD', axis = 1)

original_columns = (
    original_categorical_columns
    + original_continuous_columns
    + original_datetime_columns
)
#%% -------- Data Pre-Processing -------- #

pre_proc_method = "GMM"

(
    x_train,
    original_metric_set,
    reordered_dataframe_columns,
    continuous_transformers,
    categorical_transformers,
    datetime_transformers,
    num_categories,
    num_continuous,
) = mimic_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)

#%% -------- Create & Train VAE -------- #

# User defined parameters

# General training
batch_size = 32
n_epochs = 5
logging_freq = 1  # Number of epochs we should log the results to the user
patience = 5  # How many epochs should we allow the model train to see if
# improvement is made
delta = 10  # The difference between elbo values that registers an improvement
filepath = None  # Where to save the best model


# Privacy params
differential_privacy = False  # Do we want to implement differential privacy
sample_rate = 0.1  # Sampling rate
noise_scale = None  # Noise multiplier - influences how much noise to add
target_eps = 1  # Target epsilon for privacy accountant
target_delta = 1e-5  # Target delta for privacy accountant

# Define the metrics you want the model to evaluate

# Define distributional metrics required - for sdv_baselines this is set by default
distributional_metrics = [
    "SVCDetection",
    "GMLogLikelihood",
    "CSTest",
    "KSTest",
    "KSTestExtended",
    "ContinuousKLDivergence",
    "DiscreteKLDivergence",
]

gower = False

# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)

generator = None
sample_rate = batch_size / len(dataset)
data_loader = DataLoader(
    dataset,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(dataset), sample_rate=sample_rate, generator=generator
    ),
    pin_memory=True,
    generator=generator,
)


# -------- Define our Optuna trial -------- #


def objective(
    trial,
    gower,
    distributional_metrics,
    differential_privacy=False,
    target_delta=1e-3,
    target_eps=10.0,
    n_epochs=50,
):

    latent_dim = trial.suggest_int("Latent Dimension", 2, 128, step=2)  # Hyperparam
    hidden_dim = trial.suggest_int("Hidden Dimension", 32, 1024, step=32)  # Hyperparam

    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)

    lr = trial.suggest_float("Learning Rate", 1e-3, 1e-2, step=1e-5)
    vae = VAE(encoder, decoder, lr=1e-3)  # lr hyperparam

    C = trial.suggest_int("C", 10, 1e4, step=50)

    if differential_privacy == True:
        (
            training_epochs,
            log_elbo,
            log_reconstruction,
            log_divergence,
            log_categorical,
            log_numerical,
        ) = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=C,  # Hyperparam
            target_eps=target_eps,
            target_delta=target_delta,
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")

    else:

        (
            training_epochs,
            log_elbo,
            log_reconstruction,
            log_divergence,
            log_categorical,
            log_numerical,
        ) = vae.train(data_loader, n_epochs=n_epochs)

    # -------- Generate Synthetic Data -------- #

    synthetic_supp = constraint_filtering(
        n_rows=data_supp.shape[0],
        vae=vae,
        reordered_cols=reordered_dataframe_columns,
        data_supp_columns=data_supp.columns,
        cont_transformers=continuous_transformers,
        cat_transformers=categorical_transformers,
        date_transformers=datetime_transformers,
        pre_proc_method=pre_proc_method,
    )

    # -------- Datetime Handling -------- #

    # If the dataset has datetimes then we need to re-convert these to a numerical
    # Value representing seconds, this is so we can evaluate the metrics on them

    metric_synthetic_supp = synthetic_supp.copy()

    for index, column in enumerate(original_datetime_columns):

        # Fit datetime transformer - converts to seconds
        temp_datetime = datetime.DatetimeTransformer()
        temp_datetime.fit(metric_synthetic_supp, columns=column)

        metric_synthetic_supp = temp_datetime.transform(metric_synthetic_supp)

    # -------- SDV Metrics -------- #
    # Calculate the sdv metrics for SynthVAE

    metrics = distribution_metrics(
        gower_bool=gower,
        distributional_metrics=distributional_metrics,
        data_supp=data_supp,
        synthetic_supp=synthetic_supp,
        categorical_columns=original_categorical_columns,
        continuous_columns=original_continuous_columns,
        saving_filepath=None,
        pre_proc_method=pre_proc_method,
    )

    # Optuna wants a list of values in float form

    list_metrics = [metrics[i] for i in metrics.columns]

    print(list_metrics)

    return list_metrics


#%% -------- Run Hyperparam Optimisation -------- #

# If there is no study object in your folder then run and save the study so
# It can be resumed if needed

first_run = True  # First run indicates if we are creating a new hyperparam study

if first_run == True:

    if gower == True:
        directions = ["maximize" for i in range(distributional_metrics.shape[0] + 1)]
    else:
        directions = ["maximize" for i in range(distributional_metrics.shape[0])]

    study = optuna.create_study(directions=directions)

else:

    with open("no_dp_MIMIC.pkl", "rb") as f:
        study = pickle.load(f)

study.optimize(
    lambda trial: objective(
        trial,
        gower=gower,
        distributional_metrics=distributional_metrics,
        differential_privacy=differential_privacy,
        target_delta=target_delta,
        target_eps=target_eps,
        n_epochs=n_epochs,
    ),
    n_trials=3,
    gc_after_trial=True,
)  # GC to avoid OOM
#%%

study.best_trials
#%% -------- Save The  Study -------- #

# For a multi objective study we need to find the best trials and basically
# average between the 3 metrics to get the best trial

with open("no_dp_MIMIC.pkl", "wb") as f:
    pickle.dump(study, f)

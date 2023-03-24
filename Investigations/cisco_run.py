#%% -------- Import Libraries -------- #

# Standard imports
import numpy as np
import pandas as pd
import torch

# VAE is in other folder as well as opacus adapted library
import sys

sys.path.append("../")

# Opacus support for differential privacy
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For the SUPPORT dataset
from pycox.datasets import support

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# Utility file contains all functions required to run notebook
from utils import (
    set_seed,
    support_pre_proc,
    plot_elbo,
    plot_likelihood_breakdown,
    plot_variable_distributions,
    reverse_transformers,
)
from metrics import distribution_metrics, privacy_metrics

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("data/baseline_no_anomaly_500Gbps.csv")
# Find all rows that Producer==leaf4
tmp_df = df[(df['name']=='Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/data-rate') & (df['Producer']=='leaf5')]
tmp_df = tmp_df.dropna(axis=1, how='all')
tmp_df = tmp_df.select_dtypes(exclude=['object'])
tmp_df = tmp_df.drop(['time'], axis=1)
tmp_df = tmp_df[['input-load']]

categorical_column_name=[]
categorical_len_count=0
for col in tmp_df:
    if len(tmp_df[col].unique()) <= 3:
        categorical_column_name.append(col)
        categorical_len_count += len(tmp_df[col].unique())

# Column Definitions
original_continuous_columns = list(set(tmp_df.columns.values.tolist()) - set(categorical_column_name))
original_categorical_columns = categorical_column_name

#%% -------- Data Pre-Processing -------- #

pre_proc_method = "standard"

(
    x_train,
    tmp_df,
    reordered_dataframe_columns,
    continuous_transformers,
    categorical_transformers,
    num_categories,
    num_continuous,
) = support_pre_proc(data_supp=tmp_df,
                     continuous_columns=original_continuous_columns,
                     categorical_columns=original_categorical_columns,
                     all_possible_categories=categorical_len_count,
                     pre_proc_method=pre_proc_method)

#%% -------- Create & Train VAE -------- #

# User defined hyperparams
# General training
batch_size = 1
latent_dim = 512
hidden_dim = 1024
n_epochs = 5
logging_freq = 1  # Number of epochs we should log the results to the user
patience = 100  # How many epochs should we allow the model train to see if
# improvement is made
delta = 10  # The difference between elbo values that registers an improvement
filepath = None  # Where to save the best model


# Privacy params
differential_privacy = False  # Do we want to implement differential privacy
sample_rate = 0.1  # Sampling rate
C = 1e16  # Clipping threshold any gradients above this are clipped
noise_scale = 1.2  # Noise multiplier - influences how much noise to add
target_eps = 1  # Target epsilon for privacy accountant
target_delta = 1e-5  # Target delta for privacy accountant

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

# Create VAE
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(latent_dim, num_continuous, hidden_dim=hidden_dim, num_categories=num_categories)

vae = VAE(encoder, decoder)

print(vae)

if differential_privacy == False:
    (
        training_epochs,
        log_elbo,
        log_reconstruction,
        log_divergence,
        log_categorical,
        log_numerical,
    ) = vae.train(
        data_loader,
        n_epochs=n_epochs,
        logging_freq=logging_freq,
        patience=patience,
        delta=delta,
    )

elif differential_privacy == True:
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
        logging_freq=logging_freq,
        patience=patience,
        delta=delta,
        C=C,
        target_eps=target_eps,
        target_delta=target_delta,
        sample_rate=sample_rate,
        noise_scale=noise_scale,
    )
    print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
#%% -------- Plot Loss Features ELBO Breakdown -------- #

elbo_fig = plot_elbo(
    n_epochs=training_epochs,
    log_elbo=log_elbo,
    log_reconstruction=log_reconstruction,
    log_divergence=log_divergence,
    saving_filepath="",
)

#%% -------- Plot Loss Features Reconstruction Breakdown -------- #

likelihood_fig = plot_likelihood_breakdown(
    n_epochs=training_epochs,
    log_categorical=log_categorical,
    log_numerical=log_numerical,
    saving_filepath="",
    pre_proc_method=pre_proc_method,
)

#%% -------- Synthetic Data Generation -------- #

synthetic_sample = vae.generate(x_train.shape[0])

if torch.cuda.is_available():
    synthetic_sample = pd.DataFrame(
        synthetic_sample.cpu().detach().numpy(),
        columns=reordered_dataframe_columns
    )
else:
    synthetic_sample = pd.DataFrame(
        synthetic_sample.detach().numpy(),
        columns=reordered_dataframe_columns
    )

# Reverse the transformations

synthetic_supp = reverse_transformers(
    synthetic_set=synthetic_sample,
    data_supp_columns=tmp_df.columns,
    cont_transformers=continuous_transformers,
    cat_transformers=categorical_transformers,
    pre_proc_method=pre_proc_method,
)

#%% -------- Plot Histograms For All The Variable Distributions -------- #

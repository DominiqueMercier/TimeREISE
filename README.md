[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# TimeREISE
This repository provides all code necessary to reproduce the results reported in our paper `TimeREISE: Time-series Randomized Evolving Input Sample Explanation`[[Sensors](https://doi.org/10.3390/s22114084)][[arXiv](https://arxiv.org/abs/2202.07952)].

<strong>Abstract</strong>: Deep neural networks are one of the most successful classifiers across different domains. However, their use is limited in safety-critical areas due to their limitations concerning interpretability. The research field of explainable artificial intelligence addresses this problem. However, most interpretability methods align to the imaging modality by design. The paper introduces TimeREISE, a model agnostic attribution method that shows success in the context of time series classification. The method applies perturbations to the input and considers different attribution map characteristics such as the granularity and density of an attribution map. The approach demonstrates superior performance compared to existing methods concerning different well-established measurements. TimeREISE shows impressive results in the deletion and insertion test, Infidelity, and Sensitivity. Concerning the continuity of an explanation, it showed superior performance while preserving the correctness of the attribution map. Additional sanity checks prove the correctness of the approach and its dependency on the model parameters. TimeREISE scales well with an increasing number of channels and timesteps. TimeREISE applies to any time series classification network and does not rely on prior data knowledge. TimeREISE is suited for any usecase independent of dataset characteristics such as sequence length, channel number, and number of classes.

## Requirements

An appropriate Python environment can be set up using the `src/requirements.txt` file provided in the repo. The respective datasets can be downloaded from the [UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/dataset.php) and should be placed in the `data/` folder.

## Basic Usage

Results can be reproduced by running the corresponding bash scripts located in the subfolders of `src/bash_scripts/` as outlined in the table below. Models are saved in `models/` and resulting evaluation files are placed under `results/`.

Script - Description|File
---|:--
Script 1 - Train the baseline models|`execute_baseline.sh`
Script 2 - Compute the attributions|`execute_attribution.sh`
Script 3 - Compute the metrics|`execute_metrics.sh`

## Advanced Usage
Additional notebooks to compute figures and execute the approach in a notebook can be found in `src/notebooks`.

Notebook - Description|File
---|:--
Notebook 1 - Train baseline, compute attribution|`classification.ipynb`
Notebook 2 - Extract runtime from logs|`runtime_extract.ipynb`
Notebook 3 - Compute sanity of TimeREISE|`sanity_checker.ipynb`
Notebook 4 - Produces paper plots|`paper_plots.ipynb`

## Citation

Please consider citing our associated [paper](#):
```
    @Article{s22114084,
        AUTHOR = {Mercier, Dominique and Dengel, Andreas and Ahmed, Sheraz},
        TITLE = {TimeREISE: Time Series Randomized Evolving Input Sample Explanation},
        JOURNAL = {Sensors},
        VOLUME = {22},
        YEAR = {2022},
        NUMBER = {11},
        ARTICLE-NUMBER = {4084},
        URL = {https://www.mdpi.com/1424-8220/22/11/4084},
        ISSN = {1424-8220},
        DOI = {10.3390/s22114084}
    }
```

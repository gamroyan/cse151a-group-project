# Cse151A - Group Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pX_-QEuy3mVDzJt_kRWsYxMzv_V7c8IX?usp=sharing)

## Data and Environment Setup

Link to the data: https://huggingface.co/datasets/wykonos/movies

To get the dataset locally, we included imports from datasets to fetch the dataset from huggingface:

```python
from datasets import load_dataset
import pandas as pd
```

## Data Exploration

How many observations does your dataset have?

Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.

Do you have missing and duplicate values in your dataset?

## Data Plots

Plot your data with various types of charts like bar charts, pie charts, scatter plots etc. and clearly explain the plots.

How will you preprocess your data? Handle data imbalance if needed. You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo. 

You must also include in your Jupyter Notebook, a link for data download and environment setup requirements

# Cse151A - Group Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pX_-QEuy3mVDzJt_kRWsYxMzv_V7c8IX?usp=sharing)

## Data and Environment Setup

Link to the data: https://huggingface.co/datasets/wykonos/movies

To get the dataset locally, we included imports from datasets to fetch the dataset from huggingface:

```python
from datasets import load_dataset
import pandas as pd
load_dataset("wykonos/movies")
```

*Data Exploration and Data Plots included in Notebook

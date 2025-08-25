# Cse151A - Group Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pX_-QEuy3mVDzJt_kRWsYxMzv_V7c8IX?usp=sharing)

This project explores the [Wykonos Movies Dataset](https://huggingface.co/datasets/wykonos/movies) to perform data preprocessing and predictive modeling. We focus on analyzing attributes like genre, runtime, budget, revenue, popularity, and vote averages to build models that predict movie performance.

## Data and Environment Setup

### Option 1 – Run on Google Colab

Simply click the **Open in Colab** button above and run the notebooks in order.

### Option 2 – Run Locally

1. Clone this repository
  ```
  git clone https://github.com/gamroyan/cse151a-group-project.git
  cd cse151a-group-project
  ```
2. Install dependencies
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn datasets
  ```

4. Launch Jupyter Notebook
  ```
  jupyter notebook
  ```

## Dataset
- **Source:** [Hugging Face: wykonos/movies](https://huggingface.co/datasets/wykonos/movies)
- **Features:** id, title, genres, original_language, overview, popularity, budget, revenue, runtime, vote_average, vote_count, etc
- **Dataset size:** ~722,796 movies with 20 attributes.

To get the dataset locally, we included imports from datasets to fetch the dataset from huggingface:

```python
from datasets import load_dataset
import pandas as pd
load_dataset("wykonos/movies")
```

## Contributors
- **Team members:** Gayane Amroyan, Ethan Jenkins, Richenda Janowitz, Akshay Uppal
- **Course:** CSE151A at UCSD

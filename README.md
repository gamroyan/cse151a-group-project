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

## Milestone 2: Data Preprocessing
Our dataset offers a wide variety of attributes for each movie, like genre, runtime, budget, revenue, and status. We have many numerical attributes like popularity and date released but also many categorical features like original language and overview. 

Firstly, we realized that deleting rows isn't the most effective way to address missing data in this dataset. It would significantly reduce our dataset, as some of the attributes like revenue are consistently empty. We could replace missing numerical features with the mean or median and use a placeholder like "unknown" or "n/a" for the categorical ones instead of dropping null values. If possible, we'd also flag whether a value was missing to help the model find any patterns in the missing data and figure out ways to reduce bias.

After dropping the appropriate null values, we'd use standardization to scale numerical data (e.g. budget, revenue, runtime, popularity, vote_count) so mean = 0 and standard deviation = 1. We'd also use transforms to normalize skewed data (e.g. budget and revenue) before standardizing to make sure distributions are normalized. We can use many different label encoding methods for categorical features. For genres, we can use value replacement and take k number of categories and assign integer values from 0 to k-1 (drama -> 0, comedy -> 1, ... etc). We can one-hot encode the attributes original_language and status, as they have low-cardinality and there isn't much meaning between the classes. For categorical text-heavy fields (e.g. overview, tagline, keywords, credits), we can clean the strings like removing punctuation and either use vectorization methods or limit the vocabulary size if needed.

In order to extract the most relevant information to predict our target variable, vote average, the most effectively, we want to focus on a few derived features like profitability ```(revenue - budget) / budget``` and popularity-to-vote ratio ```popularity / vote_count```. However, one thing we noticed was how prominent the outliers/ extreme values were in the numerical features like budget, revenue, and runtime. If this affects our analysis, we decided we’d cap the values at a reasonable threshold (e.g. the 1st and 99th percentiles) to limit their influence while preserving the overall distribution. Also, we’d remove or ignore rows with impossible or too extreme entries (e.g. budgets or runtimes outside realistic bounds) to avoid distortions in the model.

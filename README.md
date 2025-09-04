# Cse151A - Group Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pX_-QEuy3mVDzJt_kRWsYxMzv_V7c8IX?usp=sharing)

*Please use the above link to access the final project notebook.*

*Our Final report is contained in the document titled "finalreport.md" for readability and ease of access.*

***

This project explores the [Wykonos Movies Dataset](https://huggingface.co/datasets/wykonos/movies) to perform data preprocessing and predictive modeling. We focus on analyzing attributes like genre, runtime, budget, revenue, popularity, and vote averages to build models that predict movie performance.

## Environment Setup
### Requirements
```
- python>=3.9
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- datasets
```

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
- *See finalreport.md for detailed collaboration information*

## Changelog
#### 09/03/2025
- Model conclusion - *Richenda*
- Written report (Discussion, Conclusion, Statement of Collaboration) - *Richenda*
#### 09/01/2025
- Written report (Intro, Figures, Methods, Results) - *Gayane*
#### 08/30/2025
- KNN on reduced data - *Ethan*
- Model evaluation and predictions - *Ethan*
#### 08/29/2025
- t-SNE - *Shay*
#### 08/27/2025
- Milestone 4 Meeting; *Shay, Ethan, Gayane*
- PCA - *Ethan*
#### 08/24/2025
- Random Forest Regressor added, Conclusion written - *Shay*
- Milestone 3 Q3 and Q4 written - *Ethan*
#### 08/23/2025
- Plots added to visualize log transform - *Gayane*
- KNN model added - *Ethan*
#### 08/22/2025
- Milestone 3 meeting - *Shay, Richenda, Gayane, Ethan*
- Data preprocessing step complete (duplicates, missing data, zeros, release_year to year only, one-hot encoding, log transforms) - *Gayane, Richenda*
#### 08/21/2025
- Milestone 3 outlined added to notebook - *Richenda*
#### 08/17/2025
- Data exploration written - *Ethan* 
- Data plots added - *Richenda, Gayane, Ethan* 
- Preprocessing outline - *Gayane*
#### 08/13/2025
- Milestone 2 created - *Gayane*
#### 08/09/2025
- Milestone 1 abstract written - *Richenda*
- Edited - *Ethan, Gayane* 
- Submitted - *Shay*
#### 08/07/2025
- Meeting for potential datasets for exploration. wykonos/movies dataset chosen - *Ethan, Shay, Gayane, Richenda*

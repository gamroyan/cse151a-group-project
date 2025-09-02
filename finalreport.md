# Final Report
## Introduction
Have you ever sat down to watch a movie and struggled to decide which one might actually be worth your time? This everyday dilemma reflects a much larger challenge in the film industry: predicting which movies will perform well and resonate with audiences. That's why we chose to use the [Wykonos Movies Dataset](https://huggingface.co/datasets/wykonos/movies) to develop a predictive model that predicts movie performance based on 20 attributes that ranged from financial measures like budget and revenue to cultural markers like genre and runtime. With over 700k movies in this dataset, the variety of films mirrors the real-world diversity of the film industry. 

Not only does predicting movie performance connect to a topic lots of people are passionate about, it also generates data-driven insights with practical value for the entertainment and media industries. For film studios and streaming platforms, predictive insights can guide decisions on how to target marketing and how to anticipate audience preferences based on the types of movies they make/sell. For audiences, this predictive model highlights the quantitative factors that influence success in a very subjective industry. On a smaller scale, this kind of model could help everyday viewers decide which movie is worth watching on a given night.

## Figures
**(TO-DO)** Your report should include relevant figures of your choosing to help with the narration of your story, including legends (similar to a scientific paper). For reference you search machine learning and your model in google scholar for reference examples.  (3 points)

## Methods

### Data Exploration
We began by exploring the movies dataset, which contains over 700,000 entries with both numerical and categorical attributes. Numerical attributes include runtime, budget, revenue, popularity, vote count, and release date. Categorical attributes include original language, status, genre, and overview. Initial exploration of this dataset involved plotting histograms and distributions to understand ranges and skewness and help us figure out how to address missing data. **(TO-DO: ADD figure of plots)** After exploring, we realized we had a lot of cleaning up to do, especially when it came to the skew in some of the attributes and missing data points.


### Preprocessing
We began preprocessing by filtering the dataset to only include movies with ```status = "Released"```. Unrealeased/planned movies don't provide us much relevant information when training a model to predict how a movie will perform. From the full list of attributes, we kept the most relevant features for prediction: ```original_language```, ```popularity```, ```release_date```, ```budget```, ```revenue```, ```runtime```, ```vote_average```, and ```vote_count```. After feature selection, we prepared the dataset for modeling by:

1. Handling Missing Data
   - For categorical attributes, missing values were filled with placeholders like "other" for genre, "zz" for original_language.
   - For numerical attributes (popularity, budget, revenue, runtime, vote_average, vote_count) missing entries were filled with 0.
     
2. Feature Transformation
   - Applied a log transform (```log1p```) to skewed numerical features (budget, revenue, vote_count, popularity) to normalize their distributions. We generated histograms to compare raw and log-transformed distributions for budget and revenue **(TO-DO: ADD figure)**.
   - Runtime values were clipped to a range between 5 and 300 minutes to remove impossible/unrealistic outliers, like so:

      ```python
     moviesFiltered["runtime"] = moviesFiltered["runtime"].clip(5, 300)

4. Encoding Categorical Values
   - Genres were split into multiple categories and one-hot encoded using ```pandas.get_dummies```. Duplicate IDs were collapsed by taking the maximum value across duplicates, like so:

     ```python
     moviesFiltered = pd.get_dummies(moviesFiltered.explode('genres'), columns=['genres'], dtype=int).groupby('id').max()
   - Original language was reduced to the top 10 most frequent languages, with all others grouped into "zz", then one-hot encoded.
   - Release dates were converted into int years, with missing values filled as 0.
  
These preprocessing steps ensured that the data was consistent and normalized among all the attributes.


### Model 1: Decision Trees, Random Forest Regressor
For our first predictive model, we implemented a Random Forest Regressor. The dataset was split into 80% training and 20% testing, with random shuffling (```random_state = 14```) applied before the split. Features were standardized so that numerical attributes had mean = 0 and standard deviation = 1.

The Random Forest was trained with the hyperparameters:
- n_estimators = 150
- max_depth = 12
- max_features = 0.5
- min_samples_leaf = 5
- bootstrap = True
- max_samples = 0.7

Model evaluation was performed on the test split using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R^2, and the percentage of predictions within +/- 1.0 of the true vote_average.

Model 1 Test Results:
- Test MSE = 1.2107
- Test RMSE ≈ 1.1003
- Test R^2 = 0.8787
- +/- 1.0 accuracy = 82.83%

On the fitting curve, the Random Forest with depth 12 and minimum samples per leaf = 5 hit the “sweet spot.” Shallower forests (lower depth) tended to underfit, showing higher test error, while deeper forests lowered training error but began to overfit. This was a strong baseline, but we wanted to improve performance by clustering the dataset and using KNN within our clusters for our second model.


### Unsupervised Learning: PCA
Before building our second model, we applied **t-distributed Stochastic Neighbor Embedding (t-SNE)** to project the high-dimensional feature space into two dimensions for visualization. This allowed us to see whether clusters of “good” and “bad” movies (based on ```vote_average```) could be separated in a lower-dimensional space. 

20,000 movies were randomly selected from the dataset, and we applied TSNE with parameters:
- n_components=2,
- perplexity=30,
- learning_rate='auto',
- init='random',
- max_iter=1000,
- random_state=42,
- metric='euclidean',

The resulting 2D scatterplot **(TO-DO: ADD figure)** revealed local clustering patterns where “good” movies (```vote_average >= mean(vote_average)```) were labeled green and “bad” movies (```vote_average < mean(vote_average```) were labeled red. While the boundaries weren't perfectly separate, the visualization supported the idea that neighborhood-based models like KNN could be effective at capturing local structure in this dataset.


### Model 2: K-Nearest-Neighbors (KNN)
Based on the clustering patterns revealed in the t-SNE visualization, we implemented a K-Nearest Neighbors (KNN) regressor for our second model to test whether neighborhood-based predictions could outperform the global Random Forest model.

To prepare the data, we standardized all numerical features so they had a mean of zero and unit variance. The dataset was again split into an 80/20 train-test split, which was consistent with Model 1. Like model 1, we trained on the standardized training set and evaluated on the test set, but this time around we experimented with multiple values of K (the number of neighbors), specifically testing K = 2, K = 5, and K = 8. Performance was measured using the same metrics as we did in model 1, MSE, RMSE, R^2, and the percentage of predictions within +/- 1.0 rating of the ground truth.

Model 2 Test Results:
- K = 2: MSE = 1.424, RMSE = 1.1931, R^2 = 0.8545, +/- 1.0 accuracy = 84.68%
- K = 5: MSE = 1.334, RMSE = 1.1551, R^2 = 0.8636, +/- 1.0 accuracy = 83.73%
- K = 8: MSE = 1.385, RMSE = 1.1769, R^2 = 0.8585, +/- 1.0 accuracy = 81.90%

Between these K values, K = 5 provided the best balance. The K = 2 model tended to overfit by making predictions too sensitive to local noise while K = 8 underfit the data.

We also evaluated KNN as a classifier by converting the continuous target variable into binary classes (good” movies and “bad” movies) like we had in model 1. With K = 5, the classification results were strong, with overall accuracy of 95%. Precision, recall, and F1-scores were well balanced across both classes **(TO-DO: ADD figure of classification report)**. This confirmed that the KNN approach was not only effective when predicting exact ratings but also in distinguishing between high- and low-quality films (good v. bad predictions).
